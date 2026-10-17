"""Poll PostgreSQL for queued AI runs and execute their registered async handler.

Keep reservations alive while handlers run, save valid answers through RunStore,
and stop accepting new work during shutdown. Without a handler, only recover
expired runs; do not remove queued work or pretend inference has run.
"""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
import logging
from typing import TypeVar
from uuid import uuid4

from src.backend.execution.config import ExecutorSettings
from src.backend.execution.store import RunClaim, RunResult, RunStore

RunHandler = Callable[[RunClaim], Awaitable[RunResult]]
OperationResult = TypeVar("OperationResult")
logger = logging.getLogger("uvicorn.error")


class ComputeStateUnknown(Exception):
    """Report that a handler cannot confirm its model computation has stopped.

    Raise this instead of an ordinary failure when a lost connection or cancellation
    leaves external/GPU work uncertain. The executor marks the run interrupted and
    keeps its slot occupied until stoppage is confirmed separately.
    """


class RunExecutor:
    """Run queued jobs with a concurrency limit and periodic reservation renewal.

    store reads/writes run records. handler receives a RunClaim and returns RunResult;
    if omitted, queued work stays untouched. settings controls polling, deadlines and
    shutdown waits. Constructing this object starts no background tasks; call start.

    Handlers must allow other async tasks to run, must stop their external work before
    returning or raising an ordinary error, and must not save assistant messages
    themselves. Raise ComputeStateUnknown when external work may still be running.
    """

    def __init__(
        self,
        store: RunStore,
        *,
        handler: RunHandler | None = None,
        settings: ExecutorSettings | None = None,
    ) -> None:
        """Save the store, optional handler and settings; prepare empty task tracking.

        Generate a worker ID, but do not reserve work or start tasks until start is called.
        """
        self.store = store
        self.handler = handler
        self.settings = settings or ExecutorSettings()
        self.worker_id = uuid4().hex
        self._stop_requested = asyncio.Event()
        self._dispatcher: asyncio.Task[None] | None = None
        self._active: dict[asyncio.Task[None], RunClaim] = {}

    async def _database_call(self, operation: Awaitable[OperationResult]) -> OperationResult:
        """Await a store operation with the configured database timeout in seconds.

        Return the operation's result. Propagate its errors or asyncio.TimeoutError
        instead of letting a database call block dispatch/heartbeat indefinitely.
        """
        return await asyncio.wait_for(operation, timeout=self.settings.database_timeout_seconds)

    async def start(self) -> None:
        """Recover expired runs, then start the background polling task.

        Return None once polling is scheduled; do not wait for queued runs to finish.
        Without a handler, polling only checks expired reservations. Raise RuntimeError
        if already started or old handlers remain, and propagate initial recovery errors.
        """
        if self._dispatcher is not None or self._active:
            raise RuntimeError("Executor has already started or still has unfinished handlers")
        await self._database_call(self.store.recover_expired_leases())
        self._stop_requested.clear()
        self._dispatcher = asyncio.create_task(self._dispatch(), name="localmind-run-dispatcher")
        if self.handler is None:
            logger.info("Run executor started without a handler; queued work will not be claimed")

    async def _dispatch(self) -> None:
        """Poll for eligible runs until shutdown requests a stop.

        Recover expired reservations, reserve work through the store, and start handler
        tasks up to the configured limit. Log database error types and try again after
        the poll interval. Keep each task's claim until its completion callback removes it.
        """
        while not self._stop_requested.is_set():
            try:
                await self._database_call(self.store.recover_expired_leases())
                while (
                    self.handler is not None
                    and len(self._active) < self.settings.max_concurrency
                    and not self._stop_requested.is_set()
                ):
                    claim = await self._database_call(
                        self.store.claim_next(
                            worker_id=self.worker_id,
                            lease_seconds=self.settings.lease_seconds,
                            max_concurrency=self.settings.max_concurrency,
                        )
                    )
                    if claim is None:
                        break
                    task = asyncio.create_task(self._execute(claim), name=f"run-{claim.run_id}")
                    self._active[task] = claim
                    task.add_done_callback(self._handler_finished)
            except Exception as exception:
                logger.error(
                    "Run dispatcher database error exception_type=%s", type(exception).__name__
                )
            try:
                await asyncio.wait_for(
                    self._stop_requested.wait(), timeout=self.settings.poll_seconds
                )
            except asyncio.TimeoutError:
                pass

    def _handler_finished(self, task: asyncio.Task[None]) -> None:
        """Remove a finished execution task from local tracking and log unexpected errors.

        The task must be one registered in _active. Do not release a database slot here;
        RunStore decides that from the run's status and whether compute is confirmed stopped.
        """
        claim = self._active.pop(task)
        if not task.cancelled():
            exception = task.exception()
            if exception is not None:
                logger.error(
                    "Run task stopped run_id=%s exception_type=%s",
                    claim.run_id,
                    type(exception).__name__,
                )

    async def _interrupt_safely(self, claim: RunClaim, code: str) -> None:
        """Try to mark the claim interrupted with the supplied public error code.

        Log database failures without raising them to the caller. If PostgreSQL cannot
        be reached, the saved reservation will expire and later recovery can mark the
        run interrupted. This method does not release the slot or stop external compute.
        """
        try:
            await self._database_call(self.store.interrupt(claim, code=code))
        except Exception as exception:
            # Lease expiry remains durable even if PostgreSQL is currently unreachable.
            logger.error(
                "Could not record interruption run_id=%s exception_type=%s",
                claim.run_id,
                type(exception).__name__,
            )

    async def _renew_lease(
        self, claim: RunClaim, handler_task: asyncio.Task[RunResult], stop_heartbeat: asyncio.Event
    ) -> None:
        """Keep the claim's reservation alive while its handler task is running.

        Args:
            claim: Current run reservation, including owner token and attempt.
            handler_task: Generation task to cancel if its reservation is lost.
            stop_heartbeat: Set when execution ends so an in-flight DB call can finish.

        Wait heartbeat_seconds between updates. On database failure, lost ownership or
        a cancellation request, try to record interruption and cancel handler_task.
        Return after that cancellation request; it does not prove model compute stopped.
        """
        while not stop_heartbeat.is_set():
            try:
                await asyncio.wait_for(stop_heartbeat.wait(), self.settings.heartbeat_seconds)
                return
            except asyncio.TimeoutError:
                pass
            try:
                state = await self._database_call(
                    self.store.heartbeat(claim, lease_seconds=self.settings.lease_seconds)
                )
            except Exception as exception:
                logger.error(
                    "Lease heartbeat failed run_id=%s exception_type=%s",
                    claim.run_id,
                    type(exception).__name__,
                )
                if not stop_heartbeat.is_set():
                    await self._interrupt_safely(claim, "HEARTBEAT_FAILED")
                    handler_task.cancel()
                return
            if state != "active" and not stop_heartbeat.is_set():
                code = "CANCEL_REQUESTED" if state == "cancel_requested" else "LEASE_LOST"
                await self._interrupt_safely(claim, code)
                handler_task.cancel()
                return

    async def _execute(self, claim: RunClaim) -> None:
        """Run one claimed job, renew its reservation and save its result if still allowed.

        claim identifies the job and this execution attempt. Start the handler and its
        heartbeat together. Save a RunResult through the store; mark ordinary failures
        failed, and uncertain compute/cancellation interrupted. Signal and await
        heartbeat shutdown when the handler path ends. Do not retry inference automatically.
        """

        async def invoke_handler() -> RunResult:
            """Call the registered async handler with this claim and return its RunResult.

            Raise RuntimeError if no handler is registered. Handler errors pass to _execute.
            """
            if self.handler is None:
                raise RuntimeError("No run handler is registered")
            return await self.handler(claim)

        handler_task = asyncio.create_task(invoke_handler(), name=f"handler-{claim.run_id}")
        stop_heartbeat = asyncio.Event()
        heartbeat_task = asyncio.create_task(self._renew_lease(claim, handler_task, stop_heartbeat))
        try:
            result = await handler_task
            if not isinstance(result, RunResult):
                raise TypeError("Handler must return RunResult")
            completed = await self._database_call(self.store.complete(claim, result))
            if not completed:
                await self._interrupt_safely(claim, "LEASE_LOST_OR_CANCELLED")
        except asyncio.CancelledError:
            await self._interrupt_safely(claim, "EXECUTION_STOPPED")
            raise
        except ComputeStateUnknown:
            await self._interrupt_safely(claim, "COMPUTE_STATE_UNKNOWN")
        except Exception as exception:
            logger.error(
                "Run handler failed run_id=%s exception_type=%s",
                claim.run_id,
                type(exception).__name__,
            )
            try:
                if not await self._database_call(self.store.fail(claim)):
                    await self._interrupt_safely(claim, "LEASE_LOST_OR_CANCELLED")
            except Exception as database_error:
                logger.error(
                    "Could not record failure run_id=%s exception_type=%s",
                    claim.run_id,
                    type(database_error).__name__,
                )
        finally:
            stop_heartbeat.set()
            with suppress(asyncio.CancelledError):
                await asyncio.shield(heartbeat_task)

    async def stop(self) -> None:
        """Stop taking new work, then give active handlers time to finish.

        Wait shutdown_grace_seconds for current jobs. Mark unfinished jobs interrupted,
        request task cancellation, and wait one more second. Keep tracking tasks that
        ignore cancellation and keep their database slots occupied. Return without
        waiting forever; Python task cancellation cannot prove external/GPU work stopped.
        """
        self._stop_requested.set()
        if self._dispatcher is not None:
            dispatcher = self._dispatcher
            done, _ = await asyncio.wait(
                {dispatcher}, timeout=self.settings.database_timeout_seconds + 1
            )
            if dispatcher not in done:
                dispatcher.cancel()
                logger.error("Dispatcher did not stop within the database timeout")
                done, _ = await asyncio.wait({dispatcher}, timeout=1.0)
            if dispatcher in done:
                self._dispatcher = None
                with suppress(asyncio.CancelledError):
                    await dispatcher
        tasks = set(self._active)
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=self.settings.shutdown_grace_seconds)
        if not pending:
            return
        await asyncio.gather(
            *(
                self._interrupt_safely(self._active[task], "SHUTDOWN_TIMEOUT")
                for task in pending
                if task in self._active
            )
        )
        for task in pending:
            task.cancel()
        _, still_running = await asyncio.wait(pending, timeout=1.0)
        if still_running:
            logger.error(
                "%s handler(s) did not stop; interrupted admission slots remain held",
                len(still_running),
            )
