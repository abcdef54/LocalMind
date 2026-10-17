"""Check actual event ordering, atomic snapshots, cooperative cancellation and replay cleanup."""

import asyncio
from copy import deepcopy
import json
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import sqlalchemy as sa
from langchain_core.messages import AIMessage, HumanMessage

from src.backend.ai.langgraph.graph_builder import LocalMindAI
from src.backend.ai.langgraph.graph_nodes import generate
from src.backend.execution.tests import test_executor as executor_tests
from src.backend.execution.control import RunCancellationRequested
from src.backend.execution.events import RunProgressRecorder, update_state_and_add_run_event
from src.backend.execution.handler import GenerateReply
from src.backend.execution.worker import RunExecutor
from src.backend.execution.replay import stream_events
from src.backend.database.models import Message, Run, RunEvent
from src.backend.services import runs


class InMemoryProgressRecorder:
    """Collect safe metadata in memory for graph benchmarks without a DB or HTTP client."""

    def __init__(self):
        self.events = []

    async def check_run_ownership_and_save_progress_event(self, event_type, payload):
        """Record an independent payload copy so later mutations cannot alter past events."""
        self.events.append((event_type, deepcopy(payload)))


class GraphEventTests(unittest.IsolatedAsyncioTestCase):
    """Verify graph events expose real completion order without altering pool order."""

    async def test_candidate_completion_is_emitted_before_the_whole_pool_finishes(self):
        """Candidate 2 may finish first while retaining display index 2 and original pool position."""
        progress_recorder = InMemoryProgressRecorder()
        release_first = asyncio.Event()
        second_finished = asyncio.Event()
        calls = 0

        async def sample(*args):
            """Hold the first candidate while the second is already complete."""
            nonlocal calls
            calls += 1
            index = calls
            if index == 1:
                await release_first.wait()
            else:
                second_finished.set()
            return [AIMessage(content=f"answer-{index}")], []

        config = {"configurable": {"tools_enabled": False, "progress_recorder": progress_recorder}}
        with patch("src.backend.ai.langgraph.graph_nodes.get_llm", return_value=object()), patch(
            "src.backend.ai.langgraph.graph_nodes._seeded", side_effect=lambda model, seed: model
        ), patch(
            "src.backend.ai.langgraph.graph_nodes._generate_sample_with_tools", side_effect=sample
        ), patch(
            "src.backend.ai.langgraph.graph_nodes._log_token_usage"
        ), patch(
            "builtins.print"
        ):
            task = asyncio.create_task(
                generate(
                    {
                        "width": 2,
                        "messages": [HumanMessage(content="question")],
                        "user_query": "question",
                    },
                    config,
                )
            )
            await second_finished.wait()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            completed = [
                payload
                for kind, payload in progress_recorder.events
                if kind == "candidate.completed"
            ]
            self.assertEqual(completed[0]["display_index"], 2)
            self.assertNotIn("generation.completed", [kind for kind, _ in progress_recorder.events])
            release_first.set()
            result = await task
        self.assertEqual(
            [candidate.answer for candidate in result["pool"]], ["answer-1", "answer-2"]
        )
        self.assertEqual(result["pool"][1].id, completed[0]["candidate_id"])
        self.assertEqual(progress_recorder.events[-1][0], "generation.completed")

    async def test_high_bracket_events_match_real_groups_byes_and_fallback(self):
        """Emit actual group membership and fallback winners with unchanged 5+2 model call counts."""
        progress_recorder = InMemoryProgressRecorder()
        model, ranker = AsyncMock(), AsyncMock()
        model.ainvoke.return_value = AIMessage(content="SECRET_CANDIDATE_ANSWER")
        ranker.ainvoke.return_value = AIMessage(content="???")
        model.bind = lambda **kwargs: ranker
        with patch("src.backend.ai.langgraph.graph_nodes.get_llm", return_value=model), patch(
            "src.backend.ai.langgraph.graph_nodes._seeded", side_effect=lambda model, seed: model
        ), patch("src.backend.ai.langgraph.graph_nodes._log_token_usage"), patch("builtins.print"):
            result = await LocalMindAI().chat(
                [{"role": "user", "content": "question"}],
                "fake",
                "general",
                "high",
                0.7,
                tools_enabled=False,
                progress_recorder=progress_recorder,
            )
        self.assertEqual(model.ainvoke.await_count, 5)
        self.assertEqual(ranker.ainvoke.await_count, 2)
        types = [kind for kind, _ in progress_recorder.events]
        self.assertEqual(types.count("candidate.completed"), 5)
        self.assertEqual(types.count("candidate.bye"), 2)
        self.assertEqual(types.count("group.started"), 2)
        self.assertEqual(types.count("ranking.fallback"), 2)
        for kind, payload in progress_recorder.events:
            if kind == "group.completed":
                self.assertEqual(
                    set(payload["eliminated_ids"]),
                    set(payload["candidate_ids"]) - {payload["winner_id"]},
                )
                self.assertTrue(payload["parse_failed"])
                self.assertEqual(payload["selection_method"], "fallback_first_candidate")
        self.assertEqual(
            progress_recorder.events[-1][1]["winner_id"],
            result["generation_metadata"]["winner_candidate_id"],
        )
        self.assertNotIn("SECRET_CANDIDATE_ANSWER", json.dumps(progress_recorder.events))

    async def test_low_has_no_fake_matches_and_disabled_sink_preserves_calls(self):
        """A single candidate selects its answer with zero tournament model calls."""
        progress_recorder = InMemoryProgressRecorder()
        model = AsyncMock()
        model.ainvoke.return_value = AIMessage(content="answer")
        with patch("src.backend.ai.langgraph.graph_nodes.get_llm", return_value=model), patch(
            "src.backend.ai.langgraph.graph_nodes._seeded", side_effect=lambda model, seed: model
        ), patch("src.backend.ai.langgraph.graph_nodes._log_token_usage"), patch("builtins.print"):
            await LocalMindAI().chat(
                [{"role": "user", "content": "q"}],
                "fake",
                "general",
                "low",
                0,
                tools_enabled=False,
                progress_recorder=progress_recorder,
            )
        self.assertEqual(model.ainvoke.await_count, 1)
        self.assertNotIn("group.started", [kind for kind, _ in progress_recorder.events])
        self.assertTrue(progress_recorder.events[-1][1]["ranking_skipped"])


@unittest.skipUnless(executor_tests.TEST_URL, "LOCALMIND_MIGRATION_TEST_URL is not configured")
class PersistedEventTests(unittest.IsolatedAsyncioTestCase):
    """Use real PostgreSQL locks and fake inference to check lifecycle races."""

    setUpClass = classmethod(executor_tests.ExecutorIntegrationTests.setUpClass.__func__)
    tearDownClass = classmethod(executor_tests.ExecutorIntegrationTests.tearDownClass.__func__)
    asyncSetUp = executor_tests.ExecutorIntegrationTests.asyncSetUp
    asyncTearDown = executor_tests.ExecutorIntegrationTests.asyncTearDown
    seed_run = executor_tests.ExecutorIntegrationTests.seed_run

    async def claimed_progress_recorder(self):
        """Reserve one run and initialize its candidate projection."""
        self.seed_run()
        claim = await self.store.reserve_next_queued_run_for_worker(
            worker_id="events", lease_seconds=60, max_concurrency=1
        )
        progress_recorder = RunProgressRecorder(self.sessions, claim)
        await progress_recorder.check_run_ownership_and_save_progress_event(
            "generation.started",
            {
                "candidate_count": 3,
                "ranking_group_size": 3,
                "candidates": [
                    {"candidate_id": str(index), "display_index": index + 1} for index in range(3)
                ],
            },
        )
        return claim, progress_recorder

    async def test_concurrent_completions_have_gap_free_sequence_and_atomic_projection(self):
        """Three separate transactions serialize completion counters on the same run row."""
        claim, progress_recorder = await self.claimed_progress_recorder()
        await asyncio.gather(
            *(
                progress_recorder.check_run_ownership_and_save_progress_event(
                    "candidate.completed", {"candidate_id": str(index), "display_index": index + 1}
                )
                for index in range(3)
            )
        )
        async with self.sessions() as session:
            snapshot = await runs.snapshot(session, claim.conversation_id, claim.run_id)
            rows = list(
                (
                    await session.scalars(
                        sa.select(RunEvent)
                        .where(RunEvent.run_id == claim.run_id)
                        .order_by(RunEvent.sequence)
                    )
                ).all()
            )
        self.assertEqual([row.sequence for row in rows], [1, 2, 3, 4, 5])
        self.assertEqual(snapshot["last_event_sequence"], 5)
        self.assertEqual(snapshot["visualization"]["completed_candidates"], 3)

    async def test_cancelled_or_replaced_owner_cannot_append_late_progress(self):
        """Reject late progress after cancel and keep terminal visualization stable."""
        claim, progress_recorder = await self.claimed_progress_recorder()
        async with self.sessions() as session:
            await runs.request_cancel(session, claim.conversation_id, claim.run_id)
        with self.assertRaises(RunCancellationRequested):
            await progress_recorder.check_run_ownership_and_save_progress_event(
                "candidate.completed", {"candidate_id": "0", "display_index": 1}
            )
        self.assertTrue(await self.store.confirm_cancellation(claim))
        with self.assertRaises(ValueError):
            await progress_recorder.check_run_ownership_and_save_progress_event(
                "candidate.completed", {"candidate_id": "1", "display_index": 2}
            )
        async with self.sessions() as session:
            snapshot = await runs.snapshot(session, claim.conversation_id, claim.run_id)
        self.assertEqual(snapshot["status"], "cancelled")
        self.assertIsNone(snapshot["visualization"]["winner_id"])
        self.assertEqual(snapshot["visualization"]["completed_candidates"], 0)

    async def test_stream_disconnect_releases_slot_without_cancelling_run(self):
        """Closing an SSE iterator releases its read transaction and leaves the task running."""
        claim, progress_recorder = await self.claimed_progress_recorder()
        request = AsyncMock()
        request.is_disconnected.return_value = False
        released = []
        async with self.sessions() as session:
            stream = stream_events(
                request,
                session,
                claim.conversation_id,
                claim.run_id,
                0,
                lambda: released.append(True),
            )
            first = await anext(stream)
            self.assertIn("run.started", first)
            self.assertFalse(session.in_transaction())
            await stream.aclose()
        self.assertEqual(released, [True])
        async with self.sessions() as session:
            snapshot = await runs.snapshot(session, claim.conversation_id, claim.run_id)
        self.assertEqual(snapshot["status"], "running")
        self.assertFalse(snapshot["cancel_requested"])

    async def test_cooperative_cancel_waits_for_all_candidate_requests(self):
        """Cancel during inference must not mark cancelled until both outstanding calls return."""
        self.seed_run()
        claim = await self.store.reserve_next_queued_run_for_worker(
            worker_id="events", lease_seconds=60, max_concurrency=1
        )
        with self.engine.begin() as connection:
            connection.execute(
                sa.update(Run)
                .where(Run.id == claim.run_id)
                .values(
                    settings_snapshot={
                        "model_name": "fake",
                        "reasoning_mode": "medium",
                        "tools_enabled": False,
                    }
                )
            )
        entered = 0
        all_started, release = asyncio.Event(), asyncio.Event()

        async def inference(messages):
            """Represent real remote compute that cannot be confirmed stopped until it returns."""
            nonlocal entered
            entered += 1
            if entered == 3:
                all_started.set()
            await release.wait()
            return AIMessage(content="answer to discard")

        model = AsyncMock()
        model.ainvoke.side_effect = inference
        with patch("src.backend.ai.langgraph.graph_nodes.get_llm", return_value=model), patch(
            "src.backend.ai.langgraph.graph_nodes._seeded", side_effect=lambda model, seed: model
        ), patch("src.backend.ai.langgraph.graph_nodes._log_token_usage"), patch("builtins.print"):
            executor = RunExecutor(
                self.store,
                handler=GenerateReply(self.sessions, LocalMindAI()),
                settings=executor_tests.FAST_SETTINGS,
            )
            self.executors.append(executor)
            task = asyncio.create_task(executor._execute(claim))
            await asyncio.wait_for(all_started.wait(), 3)
            async with self.sessions() as session:
                await runs.request_cancel(session, claim.conversation_id, claim.run_id)
            await asyncio.sleep(0.15)
            async with self.sessions() as session:
                state = await runs.snapshot(session, claim.conversation_id, claim.run_id)
            self.assertEqual(state["status"], "cancel_requested")
            self.assertFalse(state["compute_stopped"])
            release.set()
            await asyncio.wait_for(task, 3)
        async with self.sessions() as session:
            state = await runs.snapshot(session, claim.conversation_id, claim.run_id)
        self.assertEqual(state["status"], "cancelled")
        self.assertTrue(state["compute_stopped"])
        self.assertIsNone(state["result_message_id"])

    async def test_cancelled_inflight_transport_failure_still_keeps_unknown_slot(self):
        """Cancellation plus HTTP timeout cannot be reported as confirmed GPU stoppage."""
        import httpx

        self.seed_run()
        claim = await self.store.reserve_next_queued_run_for_worker(
            worker_id="events", lease_seconds=60, max_concurrency=1
        )
        started, release = asyncio.Event(), asyncio.Event()

        async def inference(messages):
            """Lose the connection after cancellation while remote work may remain active."""
            started.set()
            await release.wait()
            raise httpx.ReadTimeout("private server address")

        model = AsyncMock()
        model.bind_tools = lambda tools: model
        model.ainvoke.side_effect = inference
        with patch("src.backend.ai.langgraph.graph_nodes.get_llm", return_value=model), patch(
            "src.backend.ai.langgraph.graph_nodes._seeded", side_effect=lambda model, seed: model
        ), patch("src.backend.ai.langgraph.graph_nodes._log_token_usage"), patch("builtins.print"):
            executor = RunExecutor(
                self.store,
                handler=GenerateReply(self.sessions, LocalMindAI()),
                settings=executor_tests.FAST_SETTINGS,
            )
            task = asyncio.create_task(executor._execute(claim))
            await asyncio.wait_for(started.wait(), 3)
            async with self.sessions() as session:
                await runs.request_cancel(session, claim.conversation_id, claim.run_id)
            release.set()
            await asyncio.wait_for(task, 3)
        async with self.sessions() as session:
            state = await runs.snapshot(session, claim.conversation_id, claim.run_id)
        self.assertEqual(state["status"], "interrupted")
        self.assertFalse(state["compute_stopped"])
        self.assertTrue(state["cancel_requested"])
        self.assertIsNone(state["result_message_id"])

    async def test_event_and_projection_roll_back_together_on_write_failure(self):
        """A failed progress transaction must leave neither a new sequence nor projected completion."""
        claim, progress_recorder = await self.claimed_progress_recorder()

        async def fail_append(session, run, event_type, payload):
            """Change both event/projection, then simulate a failure before commit."""
            await update_state_and_add_run_event(session, run, event_type, payload)
            raise RuntimeError("simulated persistence failure")

        with patch(
            "src.backend.execution.events.update_state_and_add_run_event", side_effect=fail_append
        ):
            with self.assertRaises(RuntimeError):
                await progress_recorder.check_run_ownership_and_save_progress_event(
                    "candidate.completed", {"candidate_id": "0", "display_index": 1}
                )
        async with self.sessions() as session:
            state = await runs.snapshot(session, claim.conversation_id, claim.run_id)
        self.assertEqual(state["last_event_sequence"], 2)
        self.assertEqual(state["visualization"]["completed_candidates"], 0)

    async def test_expired_progress_and_retry_never_reuse_candidate_state(self):
        """An expired attempt cannot append; confirmed retry resets projection but keeps sequence."""
        from datetime import timedelta

        claim, progress_recorder = await self.claimed_progress_recorder()
        async with self.sessions() as session, session.begin():
            await session.execute(
                sa.update(Run)
                .where(Run.id == claim.run_id)
                .values(lease_expires_at=sa.func.clock_timestamp() - timedelta(seconds=1))
            )
        with self.assertRaises(ValueError):
            await progress_recorder.check_run_ownership_and_save_progress_event(
                "candidate.completed", {"candidate_id": "0", "display_index": 1}
            )
        await self.store.recover_expired_leases()
        async with self.sessions() as session:
            before = await runs.snapshot(session, claim.conversation_id, claim.run_id)
        await self.store.release_interrupted_run_slot_after_compute_stop_confirmed(
            claim.run_id, expected_owner=claim.lease_owner
        )
        async with self.sessions() as session:
            await runs.retry_run(session, claim.conversation_id, claim.run_id, expected_attempt=1)
        async with self.sessions() as session:
            after = await runs.snapshot(session, claim.conversation_id, claim.run_id)
        self.assertGreater(after["last_event_sequence"], before["last_event_sequence"])
        self.assertEqual(after["visualization"], {"phase": "queued"})
        self.assertEqual(after["attempt"], 1)
        next_claim = await self.store.reserve_next_queued_run_for_worker(
            worker_id="retry", lease_seconds=60, max_concurrency=1
        )
        self.assertEqual(next_claim.attempt, 2)
