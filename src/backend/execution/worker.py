"""Bounded in-process dispatcher with durable claims and cooperative shutdown."""

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
    """A handler cannot confirm external/native compute has stopped.

    Adapters use this instead of an ordinary failure for ambiguous transport or
    cancellation outcomes, retaining admission until compute stoppage is verified.
    """


class RunExecutor:
    """Execute queued runs without holding a DB session across handler execution.

    Handlers must yield to the event loop and finish external compute before
    returning/raising. They must not publish assistant messages themselves;
    publication is fenced in RunStore.complete. Native/GPU work which outlives
    cancellation requires explicit interruption acknowledgement by its adapter.
    """

    def __init__(
        self,
        store: RunStore,
        *,
        handler: RunHandler | None = None,
        settings: ExecutorSettings | None = None,
    ) -> None:
        self.store = store
        self.handler = handler
        self.settings = settings or ExecutorSettings()
        self.worker_id = uuid4().hex
        self._stop_requested = asyncio.Event()
        self._dispatcher: asyncio.Task[None] | None = None
        self._active: dict[asyncio.Task[None], RunClaim] = {}

    async def _database_call(self, operation: Awaitable[OperationResult]) -> OperationResult:
        return await asyncio.wait_for(operation, timeout=self.settings.database_timeout_seconds)

    async def start(self) -> None:
        if self._dispatcher is not None or self._active:
            raise RuntimeError("Executor has already started or still has unfinished handlers")
        await self._database_call(self.store.recover_expired_leases())
        self._stop_requested.clear()
        self._dispatcher = asyncio.create_task(self._dispatch(), name="localmind-run-dispatcher")
        if self.handler is None:
            logger.info("Run executor started without a handler; queued work will not be claimed")

    async def _dispatch(self) -> None:
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
        try:
            await self._database_call(self.store.interrupt(claim, code=code))
        except Exception as exception:
            # Lease expiry remains durable even if PostgreSQL is currently unreachable.
            logger.error(
                "Could not record interruption run_id=%s exception_type=%s",
                claim.run_id,
                type(exception).__name__,
            )

    async def _renew_lease(self, claim: RunClaim, handler_task: asyncio.Task[RunResult]) -> None:
        while True:
            await asyncio.sleep(self.settings.heartbeat_seconds)
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
                await self._interrupt_safely(claim, "HEARTBEAT_FAILED")
                handler_task.cancel()
                return
            if state != "active":
                code = "CANCEL_REQUESTED" if state == "cancel_requested" else "LEASE_LOST"
                await self._interrupt_safely(claim, code)
                handler_task.cancel()
                return

    async def _execute(self, claim: RunClaim) -> None:
        async def invoke_handler() -> RunResult:
            if self.handler is None:
                raise RuntimeError("No run handler is registered")
            return await self.handler(claim)

        handler_task = asyncio.create_task(invoke_handler(), name=f"handler-{claim.run_id}")
        heartbeat_task = asyncio.create_task(self._renew_lease(claim, handler_task))
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
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task

    async def stop(self) -> None:
        """Stop dispatch, drain briefly, then fence and cancel unfinished handlers.

        Python cannot forcibly stop a cancellation-resistant coroutine or native
        compute. Such handlers remain tracked and keep their DB admission slot.
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
