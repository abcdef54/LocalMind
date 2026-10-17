"""Verify durable generation, context selection, citations and failure behavior."""

from dataclasses import replace
from datetime import datetime, timedelta
import os
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import sqlalchemy as sa
from langchain_core.messages import AIMessage

from src.backend.execution.tests import test_executor as executor_tests
from src.backend.execution.handler import GenerateReply
from src.backend.execution.store import RunResult
from src.backend.execution.worker import RunExecutor
from src.backend.database.models import Attachment, Conversation, Message, Run, Submission
from src.backend.ai.errors import InferenceStateUnknown
from src.backend.ai.langgraph.graph_builder import LocalMindAI

TEST_URL = os.getenv("LOCALMIND_MIGRATION_TEST_URL")


@unittest.skipUnless(TEST_URL, "LOCALMIND_MIGRATION_TEST_URL is not configured")
class GenerateReplyTests(unittest.IsolatedAsyncioTestCase):
    """Exercise real PostgreSQL transactions with fake model calls and the real graph."""

    setUpClass = classmethod(executor_tests.ExecutorIntegrationTests.setUpClass.__func__)
    tearDownClass = classmethod(executor_tests.ExecutorIntegrationTests.tearDownClass.__func__)
    asyncSetUp = executor_tests.ExecutorIntegrationTests.asyncSetUp
    asyncTearDown = executor_tests.ExecutorIntegrationTests.asyncTearDown
    seed_run = executor_tests.ExecutorIntegrationTests.seed_run

    async def claim_run(self):
        """Seed one ready send and reserve its execution slot for this test."""
        self.seed_run()
        claim = await self.store.claim_next(
            worker_id="handler-test", lease_seconds=60, max_concurrency=1
        )
        self.assertIsNotNone(claim)
        return claim

    def seed_document(
        self, conversation_id, *, status="ready", submission_id=None, message_id=None
    ):
        """Insert a scoped attachment with a generation and return its UUID."""
        identity = uuid4()
        with self.engine.begin() as connection:
            connection.execute(
                sa.insert(Attachment).values(
                    id=identity,
                    conversation_id=conversation_id,
                    file_name="report.txt",
                    file_path="/unused/report.txt",
                    file_type="text/plain",
                    file_size=12,
                    status=status,
                    index_generation=uuid4(),
                    submission_id=submission_id,
                    message_id=message_id,
                    storage_key=f"blobs/{submission_id}/{identity}" if submission_id else None,
                    client_file_id="file-1" if submission_id else None,
                )
            )
        return identity

    async def test_context_is_bounded_ordered_and_current_message_once(self):
        """Read latest backend history, exclude future/other chats and use saved settings."""
        claim = await self.claim_run()
        now = datetime(2026, 10, 17, 12)
        settings = {
            "model_name": "saved-model",
            "personality": "researcher",
            "reasoning_mode": "high",
            "temperature": 0.4,
            "tools_enabled": True,
        }
        with self.engine.begin() as connection:
            connection.execute(
                sa.update(Run).where(Run.id == claim.run_id).values(settings_snapshot=settings)
            )
            connection.execute(
                sa.update(Conversation)
                .where(Conversation.id == claim.conversation_id)
                .values(summary="Known facts")
            )
            connection.execute(
                sa.update(Message)
                .where(Message.id == claim.message_id)
                .values(created_at=now, content="Current question")
            )
            for number in range(25):
                connection.execute(
                    sa.insert(Message).values(
                        conversation_id=claim.conversation_id,
                        role="user" if number % 2 == 0 else "assistant",
                        content=f"history-{number}",
                        created_at=now - timedelta(minutes=25 - number),
                    )
                )
            connection.execute(
                sa.insert(Message).values(
                    conversation_id=claim.conversation_id,
                    role="user",
                    content="future-secret",
                    created_at=now + timedelta(minutes=1),
                )
            )
            other = connection.execute(
                sa.insert(Conversation).values(title="Other")
            ).inserted_primary_key[0]
            connection.execute(
                sa.insert(Message).values(
                    conversation_id=other, role="user", content="other-secret"
                )
            )
        ai = AsyncMock()
        ai.chat.return_value = {"final_answer": "answer"}
        result = await GenerateReply(self.sessions, ai)(
            replace(claim, settings_snapshot={"model_name": "tampered"})
        )
        kwargs = ai.chat.call_args.kwargs
        self.assertEqual(kwargs["model_name"], "saved-model")
        self.assertEqual(kwargs["conversation_id"], claim.conversation_id)
        messages = kwargs["messages"]
        self.assertEqual(len(messages), 22)
        self.assertIn("Known facts", messages[0]["content"])
        self.assertEqual(
            [row["content"] for row in messages[1:-1]],
            [f"history-{number}" for number in range(5, 25)],
        )
        self.assertEqual(messages[-1]["content"], "Current question")
        self.assertNotIn("secret", str(messages))
        self.assertEqual(result.content, "answer")

    async def test_complete_saves_settings_metadata_and_only_live_scoped_citations(self):
        """Atomically save one answer and recheck citations changed during generation."""
        claim = await self.claim_run()
        valid = self.seed_document(claim.conversation_id)
        deleted = self.seed_document(claim.conversation_id)
        pending = self.seed_document(claim.conversation_id, status="pending")
        with self.engine.begin() as connection:
            other = connection.execute(
                sa.insert(Conversation).values(title="Other")
            ).inserted_primary_key[0]
            connection.execute(
                sa.update(Run)
                .where(Run.id == claim.run_id)
                .values(
                    settings_snapshot={
                        "model_name": "m" * 200,
                        "personality": "coder",
                        "reasoning_mode": "high",
                        "temperature": 0.7,
                    }
                )
            )
        foreign = self.seed_document(other)
        ai = AsyncMock()
        ai.chat.return_value = {
            "final_answer": "answer",
            "citations": [
                str(value) for value in (valid, valid, deleted, foreign, pending, uuid4())
            ],
            "generation_metadata": {"candidate_count": 5, "tournament_rounds": 2},
        }
        result = await GenerateReply(self.sessions, ai)(claim)
        with self.engine.begin() as connection:
            connection.execute(
                sa.update(Attachment)
                .where(Attachment.id == deleted)
                .values(status="deleted", deleted_at=sa.func.now())
            )
        self.assertTrue(await self.store.complete(claim, result))
        self.assertFalse(await self.store.complete(claim, result))
        from src.backend.database.models import message_attachments

        with self.engine.connect() as connection:
            message = connection.execute(
                sa.select(Message).where(Message.run_id == claim.run_id)
            ).one()
            self.assertEqual(message.content, "answer")
            self.assertEqual(message.model_name, "m" * 200)
            self.assertEqual(message.personality, "coder")
            self.assertEqual(message.generation_metadata["candidate_count"], 5)
            self.assertEqual(
                connection.scalars(
                    sa.select(message_attachments.c.attachment_id).where(
                        message_attachments.c.message_id == message.id
                    )
                ).all(),
                [valid],
            )
            self.assertEqual(
                connection.scalar(sa.select(Run.status).where(Run.id == claim.run_id)), "completed"
            )

    async def test_file_only_send_and_document_readiness_check(self):
        """Include uploaded filenames for empty-text sends; reject files no longer ready."""
        claim = await self.claim_run()
        document_id = self.seed_document(
            claim.conversation_id, submission_id=claim.submission_id, message_id=claim.message_id
        )
        with self.engine.begin() as connection:
            connection.execute(
                sa.update(Message).where(Message.id == claim.message_id).values(content="")
            )
        ai = AsyncMock()
        ai.chat.return_value = {"final_answer": "file answer", "citations": [str(document_id)]}
        await GenerateReply(self.sessions, ai)(claim)
        self.assertIn("report.txt", ai.chat.call_args.kwargs["messages"][-1]["content"])
        ai.reset_mock()
        with self.engine.begin() as connection:
            connection.execute(
                sa.update(Attachment)
                .where(Attachment.id == document_id)
                .values(status="deleted", deleted_at=sa.func.now())
            )
        with self.assertRaises(ValueError):
            await GenerateReply(self.sessions, ai)(claim)
        ai.chat.assert_not_awaited()

    async def test_model_failure_saves_no_fake_answer_and_releases_slot(self):
        """Ordinary generation failure creates no assistant row and releases admission."""
        claim = await self.claim_run()
        ai = AsyncMock()
        ai.chat.side_effect = ValueError("private model failure")
        await RunExecutor(self.store, handler=GenerateReply(self.sessions, ai))._execute(claim)
        with self.engine.connect() as connection:
            run = connection.execute(sa.select(Run).where(Run.id == claim.run_id)).one()
            self.assertEqual(run.status, "failed")
            self.assertIsNone(run.lease_owner)
            self.assertNotIn("private", str(run.error))
            self.assertEqual(
                connection.scalar(
                    sa.select(sa.func.count())
                    .select_from(Message)
                    .where(Message.role == "assistant")
                ),
                0,
            )

    async def test_uncertain_transport_keeps_interrupted_slot(self):
        """A lost model connection must not free admission for overlapping remote compute."""
        claim = await self.claim_run()
        ai = AsyncMock()
        ai.chat.side_effect = InferenceStateUnknown()
        await RunExecutor(self.store, handler=GenerateReply(self.sessions, ai))._execute(claim)
        with self.engine.connect() as connection:
            run = connection.execute(sa.select(Run).where(Run.id == claim.run_id)).one()
            self.assertEqual(run.status, "interrupted")
            self.assertEqual(run.lease_owner, claim.lease_owner)
            self.assertIsNone(run.result_message_id)

    async def test_real_graph_low_mode_to_persisted_answer(self):
        """Run the compiled graph with a fake model, then persist its real result metadata."""
        claim = await self.claim_run()
        model = AsyncMock()
        model.bind_tools = lambda tools: model
        model.ainvoke.return_value = AIMessage(content="Graph answer")
        ai = LocalMindAI()
        with patch("src.backend.ai.langgraph.graph_nodes.get_llm", return_value=model), patch(
            "src.backend.ai.langgraph.graph_nodes._seeded", side_effect=lambda model, seed: model
        ), patch("src.backend.ai.langgraph.graph_nodes._log_token_usage"):
            await RunExecutor(self.store, handler=GenerateReply(self.sessions, ai))._execute(claim)
        with self.engine.connect() as connection:
            answer = connection.execute(
                sa.select(Message).where(Message.run_id == claim.run_id)
            ).one()
            self.assertEqual(answer.content, "Graph answer")
            self.assertEqual(answer.generation_metadata["candidate_count"], 1)
            self.assertEqual(answer.generation_metadata["rank_calls"], 0)

    async def test_lost_lease_or_cancelled_run_cannot_publish_result(self):
        """Reject final writes after cancellation or lease loss, retaining no assistant row."""
        claim = await self.claim_run()
        with self.engine.begin() as connection:
            connection.execute(
                sa.update(Run).where(Run.id == claim.run_id).values(cancel_requested=True)
            )
        self.assertFalse(await self.store.complete(claim, RunResult("late answer")))
        ai = AsyncMock()
        with self.assertRaises(ValueError):
            await GenerateReply(self.sessions, ai)(claim)
        ai.chat.assert_not_awaited()

    async def test_send_index_retrieve_generate_and_persist_end_to_end(self):
        """Accept a real multipart send and run real indexing/RAG/graph with fake model IO."""
        import asyncio
        import json
        from pathlib import Path
        import tempfile
        import httpx
        from fastapi import FastAPI
        from src.backend.api.application import install_api_foundation
        from src.backend.api.v1.dependencies import get_session
        from src.backend.api.v1.submissions import get_upload_settings
        from src.backend.uploads.config import UploadSettings
        from src.backend.indexing.config import IndexingSettings
        from src.backend.indexing.store import IndexingStore
        from src.backend.indexing.worker import IndexingWorker
        from src.backend.indexing.vectors import DocumentVectorIndex
        from src.backend.indexing.tests.fixtures import TinyEmbeddings

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = DocumentVectorIndex(root / "vectors", TinyEmbeddings())
            app = FastAPI()
            api = install_api_foundation(app)

            async def session_dependency():
                """Supply sessions from the disposable test database."""
                async with self.sessions() as session:
                    yield session

            api.dependency_overrides[get_session] = session_dependency
            api.dependency_overrides[get_upload_settings] = lambda: UploadSettings(root=root)
            with self.engine.begin() as connection:
                conversation_id = connection.execute(
                    sa.insert(Conversation).values(title="Pipeline")
                ).inserted_primary_key[0]
            payload = {
                "client_submission_id": str(uuid4()),
                "content": "What is alpha?",
                "settings": {"model_name": "fake"},
                "files_manifest": [
                    {"client_file_id": "local-1", "name": "alpha.txt", "size_bytes": 17}
                ],
            }
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    f"/api/v1/conversations/{conversation_id}/submissions",
                    files=[
                        ("payload", (None, json.dumps(payload))),
                        ("files", ("alpha.txt", b"alpha means first", "text/plain")),
                    ],
                )
            self.assertEqual(response.status_code, 202, response.text)
            accepted = response.json()
            self.assertIsNone(
                await self.store.claim_next(
                    worker_id="too-early", lease_seconds=10, max_concurrency=1
                )
            )
            model = AsyncMock()
            model.bind_tools = lambda tools: model
            model.ainvoke.side_effect = [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "retrieve_context", "args": {"query": "alpha"}, "id": "rag-1"}
                    ],
                ),
                AIMessage(content="Alpha means first, according to alpha.txt."),
            ]
            executor = RunExecutor(
                self.store,
                handler=GenerateReply(self.sessions, LocalMindAI()),
                settings=executor_tests.FAST_SETTINGS,
            )
            self.executors.append(executor)
            indexer = IndexingWorker(
                IndexingStore(self.sessions),
                index,
                root / "indexing_locks",
                settings=IndexingSettings(poll_seconds=0.03),
            )
            with patch("src.backend.ai.langgraph.graph_nodes.get_llm", return_value=model), patch(
                "src.backend.ai.langgraph.graph_nodes._seeded",
                side_effect=lambda model, seed: model,
            ), patch("src.backend.ai.langgraph.graph_nodes._log_token_usage"), patch(
                "src.backend.database.session.async_session_maker", self.sessions
            ), patch(
                "src.backend.indexing.resources.get_document_vector_index", return_value=index
            ):
                try:
                    await executor.start()
                    await indexer.start()
                    for _ in range(200):
                        async with self.sessions() as session:
                            status = await session.scalar(
                                sa.select(Run.status).where(Run.id == accepted["run_id"])
                            )
                        if status in ("completed", "failed", "interrupted"):
                            break
                        await asyncio.sleep(0.05)
                    self.assertEqual(status, "completed")
                finally:
                    await indexer.stop()
                    await executor.stop()
            from src.backend.database.models import message_attachments

            with self.engine.connect() as connection:
                answer = connection.execute(
                    sa.select(Message).where(Message.run_id == accepted["run_id"])
                ).one()
                self.assertIn("Alpha means first", answer.content)
                self.assertEqual(
                    str(
                        connection.scalar(
                            sa.select(message_attachments.c.attachment_id).where(
                                message_attachments.c.message_id == answer.id
                            )
                        )
                    ),
                    accepted["documents"][0]["id"],
                )
                self.assertEqual(answer.generation_metadata["candidate_count"], 1)
