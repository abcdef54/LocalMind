"""Check incremental summary coverage, failed batches and bounded input on PostgreSQL."""

from datetime import datetime, timedelta
from dataclasses import replace
import os
import unittest
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import sqlalchemy as sa

from src.backend.execution.tests import test_executor as executor_tests
from src.backend.database.models import Conversation, Message, Run
from src.backend.api.v1.schemas import ChatSettings
from src.backend.memory.service import ConversationMemory
from src.backend.memory.config import MemorySettings
from src.backend.memory.budget import estimate_messages_tokens, split_text_for_summary
from src.backend.ai.errors import InferenceStateUnknown
from src.backend.execution.handler import GenerateReply
from src.backend.execution.worker import ComputeStateUnknown

TEST_URL = os.getenv("LOCALMIND_MIGRATION_TEST_URL")


@unittest.skipUnless(TEST_URL, "LOCALMIND_MIGRATION_TEST_URL is not configured")
class ConversationMemoryTests(unittest.IsolatedAsyncioTestCase):
    """Use real summary checkpoints and fake inference; never call a production model."""

    setUpClass = classmethod(executor_tests.ExecutorIntegrationTests.setUpClass.__func__)
    tearDownClass = classmethod(executor_tests.ExecutorIntegrationTests.tearDownClass.__func__)
    asyncSetUp = executor_tests.ExecutorIntegrationTests.asyncSetUp
    asyncTearDown = executor_tests.ExecutorIntegrationTests.asyncTearDown
    seed_run = executor_tests.ExecutorIntegrationTests.seed_run

    async def make_memory(self, contents):
        """Create a claimed send with ordered prior messages; return service and IDs."""
        self.seed_run()
        claim = await self.store.claim_next(
            worker_id="memory", lease_seconds=120, max_concurrency=1
        )
        now = datetime(2026, 10, 17, 12)
        with self.engine.begin() as connection:
            connection.execute(
                sa.update(Message).where(Message.id == claim.message_id).values(created_at=now)
            )
            for index, content in enumerate(contents):
                connection.execute(
                    sa.insert(Message).values(
                        id=UUID(int=index + 1),
                        conversation_id=claim.conversation_id,
                        role="user" if index % 2 == 0 else "assistant",
                        content=content,
                        created_at=now - timedelta(minutes=len(contents) - index),
                    )
                )
        ai = AsyncMock()
        ai.summarize_conversation.return_value = "compact facts"
        settings = MemorySettings(
            context_tokens=12000,
            reply_tokens=512,
            tool_tokens=1024,
            summary_tokens=512,
            recent_messages=2,
            batch_messages=2,
        )
        memory = ConversationMemory(self.sessions, ai, settings)
        return memory, ai, claim, (now, claim.message_id)

    async def test_summary_advances_once_and_raw_history_never_overlaps(self):
        """Summarize only the oldest five messages and reuse their saved coverage on reread."""
        memory, ai, claim, position = await self.make_memory(
            [f"history-{index}" for index in range(7)]
        )
        settings = ChatSettings(model_name="fake")
        messages = await memory.prepare(claim, settings, "current", position)
        summarized = [
            message["content"]
            for call in ai.summarize_conversation.call_args_list
            for message in call.args[1]
        ]
        self.assertEqual(summarized, [f"history-{index}" for index in range(5)])
        self.assertEqual(
            [message["content"] for message in messages[1:]], ["history-5", "history-6", "current"]
        )
        self.assertEqual(ai.summarize_conversation.await_count, 3)
        again = await memory.prepare(claim, settings, "current", position)
        self.assertEqual(again, messages)
        self.assertEqual(ai.summarize_conversation.await_count, 3)
        with self.engine.connect() as connection:
            row = connection.execute(
                sa.select(Conversation).where(Conversation.id == claim.conversation_id)
            ).one()
            self.assertEqual(row.summary_through_message_id, UUID(int=5))
            self.assertEqual(row.summary_version, 3)
            self.assertEqual(row.unsummarized_message_count, 3)

    async def test_failed_batch_retains_last_successful_boundary_and_resumes(self):
        """A failed second batch must not skip history when the same input is retried."""
        memory, ai, claim, position = await self.make_memory(
            [f"history-{index}" for index in range(7)]
        )
        ai.summarize_conversation.side_effect = ["first batch", ValueError("model failure")]
        with self.assertRaises(ValueError):
            await memory.prepare(claim, ChatSettings(model_name="fake"), "current", position)
        with self.engine.connect() as connection:
            row = connection.execute(
                sa.select(Conversation).where(Conversation.id == claim.conversation_id)
            ).one()
            self.assertEqual(row.summary_through_message_id, UUID(int=2))
            self.assertEqual(row.summary, "first batch")
        ai.summarize_conversation.reset_mock()
        ai.summarize_conversation.side_effect = None
        ai.summarize_conversation.return_value = "new facts"
        await memory.prepare(claim, ChatSettings(model_name="fake"), "current", position)
        contents = [
            message["content"]
            for call in ai.summarize_conversation.call_args_list
            for message in call.args[1]
        ]
        self.assertEqual(contents, ["history-2", "history-3", "history-4"])

    async def test_unknown_legacy_summary_is_not_mixed_with_raw_history(self):
        """Ignore legacy text without a trustworthy coverage boundary; retain original DB text."""
        memory, ai, claim, position = await self.make_memory(["old", "recent"])
        with self.engine.begin() as connection:
            connection.execute(
                sa.update(Conversation)
                .where(Conversation.id == claim.conversation_id)
                .values(summary="unknown coverage")
            )
        messages = await memory.prepare(claim, ChatSettings(model_name="fake"), "current", position)
        self.assertEqual([row["content"] for row in messages], ["old", "recent", "current"])
        ai.summarize_conversation.assert_not_awaited()

    async def test_large_recent_messages_move_into_summary_to_fit_budget(self):
        """Token estimate, not just message count, determines how much raw history fits."""
        memory, ai, claim, position = await self.make_memory(["a" * 5000, "b" * 5000])
        messages = await memory.prepare(claim, ChatSettings(model_name="fake"), "current", position)
        self.assertEqual(messages[-2]["content"], "b" * 5000)
        self.assertEqual(ai.summarize_conversation.call_args.args[1][0]["content"], "a" * 5000)
        self.assertLess(
            estimate_messages_tokens(messages),
            memory.settings.context_tokens
            - memory.settings.reply_tokens
            - memory.settings.tool_tokens,
        )
        with self.assertRaises(ValueError):
            await memory.prepare(claim, ChatSettings(model_name="fake"), "x" * 20000, position)

    async def test_oversized_message_is_fully_processed_before_advancing_boundary(self):
        """Splitting old Unicode text must preserve all content and never checkpoint a partial row."""
        content = "Việt Nam 🌊 " * 1500
        memory, ai, claim, position = await self.make_memory([content])
        ai.summarize_conversation.side_effect = ["partial", ValueError("fail")]
        with self.assertRaises(ValueError):
            await memory.prepare(claim, ChatSettings(model_name="fake"), "current", position)
        with self.engine.connect() as connection:
            self.assertIsNone(
                connection.scalar(
                    sa.select(Conversation.summary_through_message_id).where(
                        Conversation.id == claim.conversation_id
                    )
                )
            )
        ai.summarize_conversation.reset_mock()
        ai.summarize_conversation.side_effect = None
        ai.summarize_conversation.return_value = "all facts"
        messages = await memory.prepare(claim, ChatSettings(model_name="fake"), "current", position)
        pieces = [
            message["content"]
            for call in ai.summarize_conversation.call_args_list
            for message in call.args[1]
        ]
        self.assertEqual("".join(pieces), content)
        self.assertEqual(len(messages), 2)

    async def test_stale_run_cannot_commit_summary_after_inference(self):
        """Replacing the reservation during summarization must reject its late checkpoint."""
        memory, ai, claim, position = await self.make_memory(["one", "two", "three"])

        async def lose_lease(*args, **kwargs):
            """Simulate another owner before this summary request completes."""
            with self.engine.begin() as connection:
                connection.execute(
                    sa.update(Run).where(Run.id == claim.run_id).values(lease_owner="replacement")
                )
            return "late summary"

        ai.summarize_conversation.side_effect = lose_lease
        with self.assertRaises(ValueError):
            await memory.prepare(claim, ChatSettings(model_name="fake"), "current", position)
        with self.engine.connect() as connection:
            self.assertIsNone(
                connection.scalar(
                    sa.select(Conversation.summary_through_message_id).where(
                        Conversation.id == claim.conversation_id
                    )
                )
            )

    async def test_version_conflict_rejects_late_summary(self):
        """Do not overwrite a summary changed by another operation while the model ran."""
        memory, ai, claim, position = await self.make_memory(["one", "two", "three"])

        async def change_summary(*args, **kwargs):
            """Increment stored version to simulate a concurrent summary update."""
            with self.engine.begin() as connection:
                connection.execute(
                    sa.update(Conversation)
                    .where(Conversation.id == claim.conversation_id)
                    .values(summary_version=1, summary="other result")
                )
            return "stale result"

        ai.summarize_conversation.side_effect = change_summary
        with self.assertRaises(ValueError):
            await memory.prepare(claim, ChatSettings(model_name="fake"), "current", position)
        with self.engine.connect() as connection:
            self.assertEqual(
                connection.scalar(
                    sa.select(Conversation.summary).where(Conversation.id == claim.conversation_id)
                ),
                "other result",
            )

    async def test_summary_transport_uncertainty_is_reported_to_executor(self):
        """Summary inference uses the same conservative slot-retention contract as answers."""
        memory, ai, claim, position = await self.make_memory(["one", "two", "three"])
        ai.summarize_conversation.side_effect = InferenceStateUnknown()
        handler = GenerateReply(self.sessions, ai, memory.settings)
        with self.assertRaises(ComputeStateUnknown):
            await handler(claim)
        ai.chat.assert_not_awaited()

    async def test_equal_timestamp_boundary_excludes_covered_ids(self):
        """Use UUID as the tie-breaker so equal timestamps do not cause gaps or duplicates."""
        memory, ai, claim, position = await self.make_memory(["one", "two", "three", "four"])
        timestamp = position[0] - timedelta(minutes=1)
        with self.engine.begin() as connection:
            connection.execute(
                sa.update(Message)
                .where(
                    Message.conversation_id == claim.conversation_id, Message.id != claim.message_id
                )
                .values(created_at=timestamp)
            )
        await memory.prepare(claim, ChatSettings(model_name="fake"), "current", position)
        covered = [
            row["content"]
            for call in ai.summarize_conversation.call_args_list
            for row in call.args[1]
        ]
        self.assertEqual(covered, ["one", "two"])

    async def test_schema_rejects_partial_summary_boundary_and_negative_version(self):
        """Both boundary fields must be present together and versions cannot be negative."""
        memory, ai, claim, position = await self.make_memory([])
        for changes in ({"summary_through_message_id": uuid4()}, {"summary_version": -1}):
            with self.subTest(changes=changes), self.assertRaises(sa.exc.IntegrityError):
                with self.engine.begin() as connection:
                    connection.execute(
                        sa.update(Conversation)
                        .where(Conversation.id == claim.conversation_id)
                        .values(**changes)
                    )

    async def test_migration_preserves_legacy_text_and_rebuilds_counters(self):
        """An old summary is preserved but gets no invented coverage boundary."""
        from alembic import command
        from alembic.config import Config
        from src.backend.constants import PROJECT_ROOT

        memory, ai, claim, position = await self.make_memory(["one", "two"])
        with self.engine.begin() as connection:
            connection.execute(
                sa.update(Conversation)
                .where(Conversation.id == claim.conversation_id)
                .values(summary="legacy summary", message_count=999, unsummarized_message_count=999)
            )
            config = Config(str(PROJECT_ROOT / "alembic.ini"))
            config.attributes["connection"] = connection
            command.downgrade(config, "0007_generation_metadata")
            command.upgrade(config, "head")
            row = connection.execute(
                sa.select(Conversation).where(Conversation.id == claim.conversation_id)
            ).one()
            self.assertEqual(row.summary, "legacy summary")
            self.assertIsNone(row.summary_through_message_id)
            self.assertEqual(row.message_count, 3)
            self.assertEqual(row.unsummarized_message_count, 3)


class MemoryBudgetTests(unittest.TestCase):
    """Validate conservative estimates and Unicode splitting without model/tokenizer downloads."""

    def test_unicode_chunks_preserve_text_and_respect_capacity(self):
        """Keep every original character when a multibyte character crosses the boundary."""
        text = "abc Việt 🌊" * 10
        chunks = split_text_for_summary(text, 13)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(chunk.encode()) <= 13 for chunk in chunks))

    def test_invalid_memory_settings_fail_before_startup(self):
        """Reject invalid integer limits and reserves that exhaust the context."""
        for changes in ({"recent_messages": 0}, {"batch_messages": True}, {"context_tokens": 500}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                MemorySettings(**changes)


class ModelBudgetTests(unittest.IsolatedAsyncioTestCase):
    """Reject tool-expanded requests before sending oversized context to inference."""

    async def test_tool_results_and_definitions_are_included_in_budget(self):
        """A large tool result/schema must not bypass the initial memory budget."""
        from unittest.mock import patch
        from src.backend.ai.langgraph.graph_utils import invoke_model

        model = AsyncMock()
        model.kwargs = {"tools": [{"description": "x" * 40000}]}
        with self.assertRaises(ValueError):
            await invoke_model(model, [{"role": "user", "content": "short question"}])
        model.kwargs = {}
        with self.assertRaises(ValueError):
            await invoke_model(model, [{"role": "tool", "content": "x" * 40000}])
        model.ainvoke.assert_not_awaited()

    async def test_summary_request_is_single_tool_free_call(self):
        """Summary inference uses its own instructions and no candidate tournament/tool binding."""
        from unittest.mock import patch
        from langchain_core.messages import AIMessage
        from src.backend.ai.langgraph.graph_builder import LocalMindAI

        model = AsyncMock()
        model.bind = lambda **kwargs: model
        model.kwargs = {}
        model.ainvoke.return_value = AIMessage(content="saved facts")
        with patch("src.backend.ai.langgraph.graph_utils.get_llm", return_value=model):
            summary = await LocalMindAI().summarize_conversation(
                "earlier facts", [{"role": "user", "content": "new fact"}], "fake"
            )
        self.assertEqual(summary, "saved facts")
        model.ainvoke.assert_awaited_once()
        self.assertIn("earlier facts", str(model.ainvoke.call_args.args))
        self.assertIn("new fact", str(model.ainvoke.call_args.args))
        model.bind_tools.assert_not_called()
