"""Check worker reservations, restart and shutdown with real PostgreSQL and fake AI handlers."""

import asyncio
from dataclasses import replace
from datetime import timedelta
import os
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.backend.database.migration import upgrade_database
from src.backend.database.models import Conversation, Message, Run, Submission
from src.backend.execution.config import ExecutorSettings
from src.backend.execution.store import RunResult, RunStore
from src.backend.execution.worker import ComputeStateUnknown, RunExecutor

TEST_URL = os.getenv("LOCALMIND_MIGRATION_TEST_URL")
FAST_SETTINGS = ExecutorSettings(
    poll_seconds=0.03,
    lease_seconds=2,
    heartbeat_seconds=0.1,
    shutdown_grace_seconds=0.1,
    database_timeout_seconds=0.3,
)


@unittest.skipUnless(TEST_URL, "LOCALMIND_MIGRATION_TEST_URL is not configured")
class ExecutorIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Verify queued work and late results without making any model or GPU calls."""

    @classmethod
    def setUpClass(cls) -> None:
        """Open the disposable test database, rejecting any other database name.

        Require localmind_migration_tests because individual cases drop its public schema.
        Create a synchronous engine for fixtures; never use the application's database.
        """
        url = make_url(TEST_URL)
        if url.database != "localmind_migration_tests":
            raise RuntimeError("Tests require disposable database localmind_migration_tests")
        cls.engine = sa.create_engine(url.set(drivername="postgresql+psycopg2"))

    @classmethod
    def tearDownClass(cls) -> None:
        """Dispose the synchronous fixture engine after all cases finish."""
        cls.engine.dispose()

    async def asyncSetUp(self) -> None:
        """Recreate the disposable schema and prepare a run store plus executors to clean up
        after the case.
        """
        with self.engine.begin() as connection:
            connection.execute(sa.text("DROP SCHEMA public CASCADE"))
            connection.execute(sa.text("CREATE SCHEMA public"))
            upgrade_database(connection)
        self.async_engine = create_async_engine(TEST_URL, poolclass=NullPool)
        self.sessions = async_sessionmaker(self.async_engine, expire_on_commit=False)
        self.store = RunStore(self.sessions)
        self.executors = []

    async def asyncTearDown(self) -> None:
        """Stop every executor started by the case and dispose its async test engine."""
        for executor in self.executors:
            await executor.stop()
        await self.async_engine.dispose()

    def seed_run(self, *, status="queued", submission_status="ready", archived=False):
        """Insert a chat, send, user message and run; return the run UUID.

        status/submission_status choose eligibility and archived hides the fixture chat.
        Use only the disposable DB and never invoke a model.
        """
        with self.engine.begin() as connection:
            conversation_id = connection.execute(
                sa.insert(Conversation).values(
                    title="Executor test",
                    deleted_at=sa.func.now() if archived else None,
                )
            ).inserted_primary_key[0]
            submission_id = connection.execute(
                sa.insert(Submission).values(
                    conversation_id=conversation_id,
                    client_submission_id=uuid4(),
                    fingerprint="a" * 64,
                    settings_snapshot={"model_name": "fake"},
                    status=submission_status,
                )
            ).inserted_primary_key[0]
            message_id = connection.execute(
                sa.insert(Message).values(
                    conversation_id=conversation_id,
                    submission_id=submission_id,
                    role="user",
                    content="Question",
                )
            ).inserted_primary_key[0]
            return connection.execute(
                sa.insert(Run).values(
                    conversation_id=conversation_id,
                    submission_id=submission_id,
                    message_id=message_id,
                    settings_snapshot={"model_name": "fake"},
                    status=status,
                )
            ).inserted_primary_key[0]

    async def run_row(self, run_id):
        """Read the supplied run UUID as a mapping so the case can inspect its stored state."""
        async with self.sessions() as session:
            return (
                (await session.execute(sa.select(Run.__table__).where(Run.id == run_id)))
                .mappings()
                .one()
            )

    async def wait_status(self, run_id, status):
        """Poll run_id until it reaches status, returning its row or raising TimeoutError after
        four seconds.
        """

        async def poll():
            """Read the fixture run every 20 milliseconds and return once its status matches."""
            while True:
                row = await self.run_row(run_id)
                if row["status"] == status:
                    return row
                await asyncio.sleep(0.02)

        return await asyncio.wait_for(poll(), timeout=4)

    async def claim(self, worker="worker", limit=1):
        """Reserve one fixture job for worker with a two-second lease and the supplied
        concurrency limit.
        """
        return await self.store.claim_next(worker_id=worker, lease_seconds=2, max_concurrency=limit)

    def expire(self, run_id):
        """Move run_id's saved expiry into the past to simulate missed heartbeats without
        waiting.
        """
        with self.engine.begin() as connection:
            connection.execute(
                sa.update(Run)
                .where(Run.id == run_id)
                .values(lease_expires_at=sa.func.clock_timestamp() - timedelta(seconds=1))
            )

    async def start_executor(self, handler=None, settings=FAST_SETTINGS):
        """Start a tracked executor using the fake handler/settings and return it for this case."""
        executor = RunExecutor(self.store, handler=handler, settings=settings)
        self.executors.append(executor)
        await executor.start()
        return executor

    async def test_claim_skips_pending_indexing_and_archived_conversations(self):
        """Reserve only queued work whose submission is ready and conversation is visible."""
        self.seed_run(status="pending")
        self.seed_run(submission_status="indexing")
        self.seed_run(archived=True)
        ready_id = self.seed_run()
        claim = await self.claim()
        self.assertEqual(claim.run_id, ready_id)
        self.assertEqual(claim.attempt, 1)
        self.assertEqual((await self.run_row(ready_id))["status"], "running")

    async def test_two_workers_cannot_claim_the_same_run_or_exceed_one_global_slot(self):
        """Race two workers and allow only one claim under the shared one-slot limit."""
        first_id = self.seed_run()
        self.seed_run()
        claims = await asyncio.gather(self.claim("first"), self.claim("second"))
        claimed = [claim for claim in claims if claim is not None]
        self.assertEqual(len(claimed), 1)
        self.assertEqual(claimed[0].run_id, first_id)
        self.assertIsNone(await self.claim("third"))

    async def test_shared_concurrency_limit_and_skip_locked_rows(self):
        """Respect a larger shared slot limit and skip run rows locked by another transaction."""
        first_id, second_id, third_id = (self.seed_run() for _ in range(3))
        with self.engine.begin() as connection:
            connection.execute(sa.select(Run.id).where(Run.id == first_id).with_for_update())
            claim = await asyncio.wait_for(self.claim(limit=2), timeout=1)
            self.assertEqual(claim.run_id, second_id)
        second = await self.claim("another-worker", limit=2)
        self.assertEqual(second.run_id, first_id)
        self.assertIsNone(await self.claim(limit=2))
        self.assertEqual((await self.run_row(third_id))["status"], "queued")

    async def test_heartbeat_and_publication_require_current_owner_and_attempt(self):
        """Reject reservation renewal or result saving with a stale owner token or attempt."""
        run_id = self.seed_run()
        claim = await self.claim()
        for stale in (
            replace(claim, lease_owner="different"),
            replace(claim, attempt=claim.attempt + 1),
        ):
            self.assertEqual(await self.store.heartbeat(stale, lease_seconds=2), "lost")
            self.assertFalse(await self.store.complete(stale, RunResult("Stale answer")))
            self.assertFalse(await self.store.fail(stale))
        self.assertEqual(await self.store.heartbeat(claim, lease_seconds=5), "active")
        self.assertEqual((await self.run_row(run_id))["attempt"], 1)

    async def test_expired_run_is_interrupted_and_keeps_slot_until_acknowledged(self):
        """Keep an expired run occupying its slot until the matching owner is explicitly
        acknowledged.
        """
        run_id = self.seed_run()
        queued_id = self.seed_run()
        claim = await self.claim()
        self.expire(run_id)
        self.assertEqual(await self.store.heartbeat(claim, lease_seconds=2), "lost")
        self.assertEqual(await self.store.recover_expired_leases(), 1)
        row = await self.run_row(run_id)
        self.assertEqual(row["status"], "interrupted")
        self.assertEqual(row["lease_owner"], claim.lease_owner)
        self.assertFalse(await self.store.complete(claim, RunResult("Late answer")))
        self.assertIsNone(await self.claim())
        self.assertFalse(await self.store.acknowledge_interruption(run_id, expected_owner="wrong"))
        self.assertTrue(
            await self.store.acknowledge_interruption(run_id, expected_owner=claim.lease_owner)
        )
        self.assertEqual((await self.claim()).run_id, queued_id)
        self.assertEqual((await self.run_row(run_id))["status"], "interrupted")

    async def test_complete_is_atomic_and_cannot_publish_twice(self):
        """Save one answer together with completion and refuse a second result from the same
        claim.
        """
        run_id = self.seed_run()
        claim = await self.claim()
        self.assertTrue(await self.store.complete(claim, RunResult("Winner answer")))
        self.assertFalse(await self.store.complete(claim, RunResult("Duplicate answer")))
        row = await self.run_row(run_id)
        self.assertEqual(row["status"], "completed")
        self.assertIsNone(row["lease_owner"])
        async with self.sessions() as session:
            messages = list(
                (await session.scalars(sa.select(Message).where(Message.run_id == run_id))).all()
            )
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].id, row["result_message_id"])
        self.assertEqual(messages[0].content, "Winner answer")

    async def test_handlerless_startup_recovers_orphans_without_claiming_queued_work(self):
        """Recover expired work without taking queued jobs when no handler is installed."""
        orphan_id = self.seed_run(status="running")
        queued_id = self.seed_run()
        executor = await self.start_executor()
        self.assertEqual((await self.run_row(orphan_id))["status"], "interrupted")
        self.assertIsNotNone((await self.run_row(orphan_id))["lease_owner"])
        self.assertEqual((await self.run_row(queued_id))["attempt"], 0)
        await executor.stop()

    async def test_executor_completes_handler_and_restart_does_not_replay_result(self):
        """Complete a fake answer, restart, and execute only the new job rather than replaying
        the old one.
        """
        run_id = self.seed_run()
        handled = []

        async def handler(claim):
            """Record the claimed run ID and return a fixed answer without model IO."""
            handled.append(claim.run_id)
            return RunResult("Fake answer, no LLM call")

        executor = await self.start_executor(handler)
        await self.wait_status(run_id, "completed")
        await executor.stop()
        next_id = self.seed_run()
        restarted = await self.start_executor(handler)
        await self.wait_status(next_id, "completed")
        await restarted.stop()
        self.assertEqual(handled, [run_id, next_id])

    async def test_handler_failure_is_terminal_and_error_does_not_contain_input(self):
        """Save a generic failed state without retrying or leaking the handler's private
        exception text.
        """
        run_id = self.seed_run()

        async def handler(claim):
            """Raise a private error string that must never reach saved errors or logs."""
            raise RuntimeError("private input must not reach DB errors")

        with self.assertLogs("uvicorn.error", level="ERROR") as logs:
            executor = await self.start_executor(handler)
            row = await self.wait_status(run_id, "failed")
            await executor.stop()
        self.assertEqual(row["error"]["code"], "HANDLER_FAILED")
        self.assertNotIn("private input", str(row["error"]))
        self.assertFalse(any("private input" in entry for entry in logs.output))
        self.assertEqual(row["attempt"], 1)
        self.assertIsNone(await self.claim())

    async def test_ambiguous_compute_failure_retains_slot_instead_of_releasing_admission(self):
        """Mark uncertain external work interrupted and keep the next job from using its slot."""
        run_id = self.seed_run()
        self.seed_run()

        async def handler(claim):
            """Raise ComputeStateUnknown to simulate a lost connection to possibly running
            inference.
            """
            raise ComputeStateUnknown("External compute may still be active")

        executor = await self.start_executor(handler)
        row = await self.wait_status(run_id, "interrupted")
        self.assertEqual(row["error"]["code"], "COMPUTE_STATE_UNKNOWN")
        self.assertIsNotNone(row["lease_owner"])
        self.assertIsNone(await self.claim())
        await executor.stop()

    async def test_heartbeat_keeps_a_long_running_handler_leased(self):
        """Keep a blocked fake handler running longer than its original expiry through heartbeat
        renewal.
        """
        run_id = self.seed_run()
        release, started = asyncio.Event(), asyncio.Event()

        async def handler(claim):
            """Signal startup, wait for release and return an answer while heartbeats keep
            running.
            """
            started.set()
            await release.wait()
            return RunResult("Done")

        settings = replace(
            FAST_SETTINGS, lease_seconds=0.6, heartbeat_seconds=0.05, database_timeout_seconds=0.15
        )
        executor = await self.start_executor(handler, settings)
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            await asyncio.sleep(0.8)
            self.assertEqual(await self.store.recover_expired_leases(), 0)
            self.assertEqual((await self.run_row(run_id))["status"], "running")
            release.set()
            await self.wait_status(run_id, "completed")
        finally:
            release.set()
            await executor.stop()

    async def test_shutdown_drains_in_flight_work_and_leaves_queue_durable(self):
        """Let current work finish during shutdown without claiming the next queued job."""
        run_id = self.seed_run()
        queued_id = self.seed_run()
        release, started = asyncio.Event(), asyncio.Event()

        async def handler(claim):
            """Wait for the shutdown case to release the active job, then return its fake answer."""
            started.set()
            await release.wait()
            return RunResult("Drained")

        executor = await self.start_executor(
            handler, replace(FAST_SETTINGS, shutdown_grace_seconds=1)
        )
        await asyncio.wait_for(started.wait(), timeout=2)
        stopping = asyncio.create_task(executor.stop())
        await asyncio.sleep(0)
        release.set()
        await stopping
        self.assertEqual((await self.run_row(run_id))["status"], "completed")
        self.assertEqual((await self.run_row(queued_id))["status"], "queued")

    async def test_shutdown_timeout_interrupts_without_claiming_native_compute_stopped(self):
        """Mark overdue work interrupted on shutdown and retain its owner rather than assert GPU
        stoppage.
        """
        run_id = self.seed_run()
        started, stopped = asyncio.Event(), asyncio.Event()

        async def handler(claim):
            """Wait until cancelled and signal that the Python handler exited through finally."""
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        executor = await self.start_executor(handler)
        await asyncio.wait_for(started.wait(), timeout=2)
        await executor.stop()
        await asyncio.wait_for(stopped.wait(), timeout=1)
        row = await self.run_row(run_id)
        self.assertEqual(row["status"], "interrupted")
        self.assertEqual(row["error"]["code"], "SHUTDOWN_TIMEOUT")
        self.assertIsNotNone(row["lease_owner"])
        self.assertIsNone(row["result_message_id"])

    async def test_late_handler_cannot_publish_after_lease_loss(self):
        """Reject a late answer from a handler that ignored its first cancellation after expiry."""
        run_id = self.seed_run()
        started, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def handler(claim):
            """Delay an answer, acknowledge but ignore cancellation once, then return after
            explicit release.
            """
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
            return RunResult("Late answer")

        executor = await self.start_executor(handler)
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            self.expire(run_id)
            await self.wait_status(run_id, "interrupted")
            await asyncio.wait_for(cancelled.wait(), timeout=2)
            release.set()
            await executor.stop()
            self.assertIsNone((await self.run_row(run_id))["result_message_id"])
        finally:
            release.set()

    async def test_heartbeat_database_failure_stops_handler_conservatively(self):
        """Interrupt/cancel on heartbeat DB failure while retaining the slot and hiding private
        connection text.
        """
        run_id = self.seed_run()
        started, stopped = asyncio.Event(), asyncio.Event()

        async def handler(claim):
            """Wait indefinitely until cancellation and signal handler exit in finally."""
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        with patch.object(
            self.store, "heartbeat", AsyncMock(side_effect=ConnectionError("private DSN"))
        ):
            with self.assertLogs("uvicorn.error", level="ERROR") as logs:
                executor = await self.start_executor(handler)
                await asyncio.wait_for(started.wait(), timeout=2)
                await asyncio.wait_for(stopped.wait(), timeout=2)
                row = await self.wait_status(run_id, "interrupted")
                await executor.stop()
        self.assertIsNotNone(row["lease_owner"])
        self.assertFalse(any("private DSN" in entry for entry in logs.output))

    async def test_cancel_flag_interrupts_without_pretending_full_cancel_api_exists(self):
        """Observe a saved cancellation flag, stop the Python handler and save no final answer."""
        run_id = self.seed_run()
        started, stopped = asyncio.Event(), asyncio.Event()

        async def handler(claim):
            """Signal startup and wait for cancellation, then signal exit without producing a
            result.
            """
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        executor = await self.start_executor(handler)
        await asyncio.wait_for(started.wait(), timeout=2)
        with self.engine.begin() as connection:
            connection.execute(
                sa.update(Run)
                .where(Run.id == run_id)
                .values(
                    status="cancel_requested",
                    cancel_requested=True,
                )
            )
        await asyncio.wait_for(stopped.wait(), timeout=2)
        row = await self.wait_status(run_id, "interrupted")
        self.assertIsNone(row["result_message_id"])
        await executor.stop()

    async def test_shutdown_returns_when_handler_resists_cancellation_and_fences_its_late_result(
        self,
    ):
        """Bound shutdown waiting and refuse a later answer even when the handler repeatedly
        ignores cancellation.
        """
        run_id = self.seed_run()
        started, release = asyncio.Event(), asyncio.Event()

        async def handler(claim):
            """Ignore task cancellation until release is set, then return a deliberately too-late
            answer.
            """
            started.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue
            return RunResult("Too late")

        executor = await self.start_executor(handler)
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            with self.assertLogs("uvicorn.error", level="ERROR"):
                await asyncio.wait_for(executor.stop(), timeout=3)
            row = await self.run_row(run_id)
            self.assertEqual(row["status"], "interrupted")
            self.assertIsNotNone(row["lease_owner"])
            release.set()
            await executor.stop()
            self.assertIsNone((await self.run_row(run_id))["result_message_id"])
        finally:
            release.set()
