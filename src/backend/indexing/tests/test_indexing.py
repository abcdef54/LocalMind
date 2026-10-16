"""Check saved-file indexing, restart/retry and scoped retrieval against PostgreSQL and Chroma."""

import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from fastapi import FastAPI
import httpx
from langchain_core.documents import Document
import sqlalchemy as sa
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.backend.api.application import install_api_foundation
from src.backend.api.v1.dependencies import get_session
from src.backend.api.v1.submissions import get_upload_settings
from src.backend.database.migration import upgrade_database
from src.backend.database.models import Attachment, Conversation, Message, Run, Submission
from src.backend.execution.store import RunStore
from src.backend.indexing.config import IndexingSettings
from src.backend.indexing.parser_process import DocumentSection
from src.backend.indexing.retrieval import ConversationRetriever
from src.backend.indexing.store import IndexingStore
from src.backend.indexing.tests.fixtures import TinyEmbeddings
from src.backend.indexing.vectors import DocumentVectorIndex
from src.backend.indexing.worker import IndexingWorker
from src.backend.uploads.config import UploadSettings

TEST_URL = os.getenv("LOCALMIND_MIGRATION_TEST_URL")
FAST_SETTINGS = IndexingSettings(
    poll_seconds=0.02,
    lease_seconds=2,
    heartbeat_seconds=0.1,
    database_timeout_seconds=0.3,
    shutdown_grace_seconds=0.1,
)


@unittest.skipUnless(TEST_URL, "LOCALMIND_MIGRATION_TEST_URL is not configured")
class IndexingIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Use real vector storage and HTTP sends, but deterministic CPU-only fake embeddings."""

    @classmethod
    def setUpClass(cls):
        """Open only the disposable DB and one shared temporary Chroma instance."""
        url = make_url(TEST_URL)
        if url.database != "localmind_migration_tests":
            raise RuntimeError("Tests require disposable database localmind_migration_tests")
        cls.engine = sa.create_engine(url.set(drivername="postgresql+psycopg2"))
        cls.vector_directory = tempfile.TemporaryDirectory()
        cls.index = DocumentVectorIndex(Path(cls.vector_directory.name), TinyEmbeddings())

    @classmethod
    def tearDownClass(cls):
        """Dispose the fixture DB engine and remove temporary Chroma files."""
        cls.engine.dispose()
        cls.vector_directory.cleanup()

    async def asyncSetUp(self):
        """Reset DB/vector records and wire a no-startup HTTP client to temporary upload storage."""
        with self.engine.begin() as connection:
            connection.execute(sa.text("DROP SCHEMA public CASCADE"))
            connection.execute(sa.text("CREATE SCHEMA public"))
            upgrade_database(connection)
        old_ids = self.index.collection.get(include=[])["ids"]
        if old_ids:
            self.index.collection.delete(ids=old_ids)
        self.temporary = tempfile.TemporaryDirectory()
        self.upload_settings = UploadSettings(root=Path(self.temporary.name))
        self.async_engine = create_async_engine(TEST_URL, poolclass=NullPool)
        self.sessions = async_sessionmaker(self.async_engine, expire_on_commit=False)
        self.store = IndexingStore(self.sessions)
        self.retriever = ConversationRetriever(self.sessions, self.index)
        self.workers = []

        async def test_session():
            """Yield a disposable DB session instead of touching the application DB."""
            async with self.sessions() as session:
                yield session

        self.app = FastAPI()
        api = install_api_foundation(self.app)
        api.dependency_overrides[get_session] = test_session
        api.dependency_overrides[get_upload_settings] = lambda: self.upload_settings
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        )
        self.conversation_id = self.new_conversation()

    async def asyncTearDown(self):
        """Stop test workers before deleting their temporary inputs or closing sessions."""
        for worker in self.workers:
            await worker.stop()
        await self.client.aclose()
        await self.async_engine.dispose()
        self.temporary.cleanup()

    def new_conversation(self):
        """Insert a live chat in the disposable database and return its UUID."""
        identity = uuid4()
        with self.engine.begin() as connection:
            connection.execute(sa.insert(Conversation).values(id=identity, title="Index test"))
        return identity

    async def send(self, files, conversation_id=None):
        """Accept named byte tuples through BE-07 and return its real HTTP response payload."""
        owner_id = conversation_id or self.conversation_id
        payload = {
            "client_submission_id": str(uuid4()),
            "content": "Read these documents",
            "settings": {"model_name": "fake"},
            "files_manifest": [
                {"client_file_id": str(i), "name": name, "size_bytes": len(content)}
                for i, (name, content) in enumerate(files)
            ],
        }
        parts = [("payload", (None, json.dumps(payload)))]
        parts.extend(("files", (name, content)) for name, content in files)
        response = await self.client.post(
            f"/api/v1/conversations/{owner_id}/submissions", files=parts
        )
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()

    async def row(self, model, identity):
        """Read one table row as a mapping, without keeping a transaction across awaits."""
        async with self.sessions() as session:
            return (
                (
                    await session.execute(
                        sa.select(model.__table__).where(model.id == UUID(str(identity)))
                    )
                )
                .mappings()
                .one()
            )

    async def wait_for_state(self, model, identity, expected):
        """Wait at most eight seconds for a stored status and return the matching row."""

        async def poll():
            """Read the short-lived DB snapshot until the expected state is visible."""
            while True:
                row = await self.row(model, identity)
                if row["status"] == expected:
                    return row
                await asyncio.sleep(0.02)

        return await asyncio.wait_for(poll(), 8)

    async def start_worker(self, **options):
        """Start a tracked worker; the caller may replace settings/parser for a controlled failure."""
        worker = IndexingWorker(
            self.store,
            self.index,
            self.upload_settings.root / "indexing_locks",
            settings=options.pop("settings", FAST_SETTINGS),
            **options,
        )
        self.workers.append(worker)
        await worker.start()
        return worker

    async def claim(self, worker_id="worker", limit=1):
        """Reserve one eligible file using the configured small test reservation."""
        return await self.store.claim_next(
            worker_id=worker_id, lease_seconds=2, max_concurrency=limit
        )

    def expire(self, attachment_id):
        """Simulate a crashed worker by moving its persisted reservation into the past."""
        with self.engine.begin() as connection:
            connection.execute(
                sa.update(Attachment)
                .where(Attachment.id == attachment_id)
                .values(index_expires_at=sa.func.clock_timestamp() - sa.text("INTERVAL '1 second'"))
            )

    async def test_real_parser_indexes_two_files_then_queues_one_run(self):
        """Move a complete batch to ready, expose its chunks and leave AI execution queued."""
        send = await self.send(
            [("notes.py", b'print("alpha")'), ("notes.cpp", b"int main() { return 0; }")]
        )
        worker = await self.start_worker()
        await self.wait_for_state(Submission, send["submission_id"], "ready")
        await worker.stop()
        run = await self.row(Run, send["run_id"])
        self.assertEqual(run["status"], "queued")
        documents = await self.retriever.search(self.conversation_id, "alpha", limit=10)
        self.assertEqual(
            {document.metadata["source"] for document in documents}, {"notes.py", "notes.cpp"}
        )
        history = (
            await self.client.get(f"/api/v1/conversations/{self.conversation_id}/messages")
        ).json()["items"]
        self.assertEqual([message["role"] for message in history], ["user"])
        self.assertTrue(all(file["chunk_count"] > 0 for file in history[0]["attachments"]))
        self.assertNotIn(self.temporary.name, json.dumps(history))

    async def test_first_ready_file_does_not_queue_until_all_inputs_ready(self):
        """Keep the run pending after the first file and queue it only after the final success."""
        send = await self.send([("a.txt", b"alpha"), ("b.txt", b"beta")])
        first = await self.claim()
        self.assertTrue(await self.store.finish(first, chunk_count=1))
        self.assertEqual((await self.row(Run, send["run_id"]))["status"], "pending")
        second = await self.claim()
        self.assertTrue(await self.store.finish(second, chunk_count=1))
        self.assertEqual((await self.row(Run, send["run_id"]))["status"], "queued")

    async def test_embedding_failure_cleans_partial_vectors_and_retry_preserves_ids(self):
        """Retry one partially indexed file without reindexing its already-ready sibling."""
        send = await self.send([("good.txt", b"good alpha"), ("retry.txt", b"retry beta")])
        original_write = self.index.upsert_batch
        failure_injected = False

        def fail_after_write(claim, chunks, start_index):
            """Write real vectors then raise once, simulating a partial external commit."""
            nonlocal failure_injected
            original_write(claim, chunks, start_index)
            if claim.file_name == "retry.txt" and not failure_injected:
                failure_injected = True
                raise RuntimeError("private embedding text must stay out of errors")

        with patch.object(self.index, "upsert_batch", fail_after_write):
            worker = await self.start_worker()
            await self.wait_for_state(Submission, send["submission_id"], "failed")
            good_id = next(
                file["id"] for file in send["documents"] if file["file_name"] == "good.txt"
            )
            await self.wait_for_state(Attachment, good_id, "ready")
            await worker.stop()
        self.assertEqual((await self.row(Run, send["run_id"]))["status"], "failed")
        self.assertIsNone(
            await RunStore(self.sessions).claim_next(
                worker_id="ai", lease_seconds=30, max_concurrency=1
            )
        )
        ready_generation = (await self.row(Attachment, good_id))["index_generation"]
        response = await self.client.post(
            f"/api/v1/conversations/{self.conversation_id}/submissions/{send['submission_id']}/retry-indexing"
        )
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()["message_id"], send["message_id"])
        self.assertEqual(response.json()["run_id"], send["run_id"])
        await worker.start()
        await self.wait_for_state(Submission, send["submission_id"], "ready")
        await worker.stop()
        self.assertEqual(
            (await self.row(Attachment, good_id))["index_generation"], ready_generation
        )
        self.assertEqual(self.index.collection.count(), 2)
        self.assertEqual(
            len(await self.retriever.search(self.conversation_id, "beta", limit=10)), 2
        )

    async def test_expired_attempt_rebuilds_and_late_result_cannot_replace_ready_generation(self):
        """Recover a partial old generation, delete its vectors and reject its late DB update."""
        send = await self.send([("old.txt", b"new safe text")])
        old_claim = await self.claim()
        self.index.upsert_batch(
            old_claim, [Document(page_content="stale content", metadata={"source": "old.txt"})], 0
        )
        self.assertEqual(await self.retriever.search(self.conversation_id, "stale"), [])
        self.expire(old_claim.attachment_id)
        worker = await self.start_worker()
        await self.wait_for_state(Submission, send["submission_id"], "ready")
        await worker.stop()
        current = await self.row(Attachment, old_claim.attachment_id)
        self.assertNotEqual(current["index_generation"], old_claim.generation)
        self.assertEqual(current["index_attempt"], 2)
        self.assertFalse(await self.store.finish(old_claim, chunk_count=1))
        self.index.delete_attempt(old_claim.attachment_id, old_claim.generation)
        self.assertEqual(self.index.collection.count(), 1)
        self.assertEqual(
            (await self.retriever.search(self.conversation_id, "text"))[0].page_content,
            "new safe text",
        )

    async def test_repeated_vector_batch_upsert_does_not_duplicate_chunks(self):
        """Repeat the exact same vector write and retain one stable chunk ID."""
        await self.send([("one.txt", b"alpha")])
        claim = await self.claim()
        chunks = [Document(page_content="alpha", metadata={"source": "one.txt"})]
        self.index.upsert_batch(claim, chunks, 0)
        self.index.upsert_batch(claim, chunks, 0)
        self.assertEqual(self.index.collection.count(), 1)

    async def test_retrieval_filters_chat_ready_generation_and_deletion(self):
        """Never return another chat, pending/failed/deleted file or obsolete generation."""
        own = await self.send([("own.txt", b"alpha own")])
        other_id = self.new_conversation()
        await self.send([("foreign.txt", b"alpha foreign")], other_id)
        worker = await self.start_worker()
        await self.wait_for_state(Submission, own["submission_id"], "ready")
        async with self.sessions() as session:
            other_send = await session.scalar(
                sa.select(Submission.id).where(Submission.conversation_id == other_id)
            )
        await self.wait_for_state(Submission, other_send, "ready")
        await worker.stop()
        documents = await self.retriever.search(self.conversation_id, "alpha", limit=10)
        self.assertEqual([document.metadata["source"] for document in documents], ["own.txt"])
        attachment_id = UUID(own["documents"][0]["id"])
        for changes in (
            {"status": "failed"},
            {"status": "pending"},
            {"status": "deleted"},
            {"status": "ready", "deleted_at": sa.func.now()},
        ):
            with self.subTest(changes=changes):
                with self.engine.begin() as connection:
                    connection.execute(
                        sa.update(Attachment)
                        .where(Attachment.id == attachment_id)
                        .values(**changes)
                    )
                self.assertEqual(await self.retriever.search(self.conversation_id, "alpha"), [])

    async def test_file_deleted_during_vector_query_is_filtered_by_fresh_db_read(self):
        """Recheck DB eligibility after similarity search instead of trusting an earlier snapshot."""
        send = await self.send([("one.txt", b"alpha")])
        worker = await self.start_worker()
        await self.wait_for_state(Submission, send["submission_id"], "ready")
        await worker.stop()
        original_search = self.index.search

        def delete_during_search(*args, **kwargs):
            """Return real query matches after deleting their document in a separate transaction."""
            matches = original_search(*args, **kwargs)
            with self.engine.begin() as connection:
                connection.execute(
                    sa.update(Attachment).values(deleted_at=sa.func.now(), status="deleted")
                )
            return matches

        with patch.object(self.index, "search", delete_during_search):
            self.assertEqual(await self.retriever.search(self.conversation_id, "alpha"), [])

    async def test_no_ready_files_never_triggers_embedding_or_unfiltered_search(self):
        """Return empty context immediately when the selected chat has no eligible documents."""
        with patch.object(self.index, "search", side_effect=AssertionError("must not query")):
            self.assertEqual(await self.retriever.search(self.conversation_id, "alpha"), [])
            self.assertEqual(await self.retriever.search(uuid4(), "alpha"), [])
        with self.assertRaises(ValueError):
            await self.retriever.search(None, "alpha")

    async def test_global_claim_limit_and_owner_checks(self):
        """Admit one lease across workers and reject a copied claim with a changed owner."""
        await self.send([("one.txt", b"a"), ("two.txt", b"b")])
        claims = await asyncio.gather(self.claim("first"), self.claim("second"))
        valid = [claim for claim in claims if claim is not None]
        self.assertEqual(len(valid), 1)
        forged = replace(valid[0], owner="wrong-owner")
        self.assertFalse(await self.store.heartbeat(forged, lease_seconds=2))
        self.assertFalse(await self.store.finish(forged, chunk_count=1))

    async def test_os_file_lock_keeps_replacement_from_overlapping_old_native_work(self):
        """Return a replacement claim to pending when the prior process still holds its file lock."""
        await self.send([("one.txt", b"a")])
        old_claim = await self.claim()
        worker = IndexingWorker(
            self.store,
            self.index,
            self.upload_settings.root / "indexing_locks",
            settings=FAST_SETTINGS,
        )
        lock = worker._acquire_file_lock(old_claim)
        try:
            self.expire(old_claim.attachment_id)
            await self.store.recover_expired()
            replacement = await self.claim()
            with patch.object(
                worker,
                "_parse_and_embed",
                AsyncMock(side_effect=AssertionError("must not compute")),
            ):
                await worker._index_one(replacement)
            self.assertEqual(
                (await self.row(Attachment, old_claim.attachment_id))["status"], "pending"
            )
        finally:
            lock.close()

    async def test_retry_checks_scope_busy_deleted_and_other_active_run(self):
        """Reject retry requests that cannot safely reuse this send's documents and run."""
        send = await self.send([("empty.txt", b"")])
        worker = await self.start_worker()
        await self.wait_for_state(Submission, send["submission_id"], "failed")
        await worker.stop()
        url = f"/api/v1/conversations/{self.conversation_id}/submissions/{send['submission_id']}/retry-indexing"
        foreign = await self.client.post(
            url.replace(str(self.conversation_id), str(self.new_conversation()))
        )
        self.assertEqual(foreign.status_code, 404)
        await self.send([("new.txt", b"new")])
        self.assertEqual((await self.client.post(url)).status_code, 409)
        with self.engine.begin() as connection:
            connection.execute(
                sa.update(Run).where(Run.id != UUID(send["run_id"])).values(status="failed")
            )
            connection.execute(
                sa.update(Attachment)
                .where(Attachment.id == UUID(send["documents"][0]["id"]))
                .values(deleted_at=sa.func.now(), status="deleted")
            )
        response = await self.client.post(url)
        self.assertEqual(response.json()["error"]["code"], "DOCUMENT_DELETED")

    async def test_shutdown_leaves_interrupted_indexing_recoverable(self):
        """Stop a blocked parser without saving an answer, then let a restarted worker finish it."""
        send = await self.send([("slow.txt", b"alpha")])
        started = asyncio.Event()

        async def blocked_parser(file_path, file_name, settings):
            """Signal parsing startup and wait for worker cancellation without doing model IO."""
            started.set()
            await asyncio.Event().wait()

        worker = await self.start_worker(parser=blocked_parser)
        await asyncio.wait_for(started.wait(), 3)
        await worker.stop()
        file = await self.row(Attachment, send["documents"][0]["id"])
        self.assertEqual(file["status"], "pending")
        restarted = await self.start_worker()
        await self.wait_for_state(Submission, send["submission_id"], "ready")
        await restarted.stop()

    async def test_uncertain_ready_commit_does_not_delete_published_vectors(self):
        """Keep searchable vectors if DB committed ready but the worker lost its acknowledgement."""
        send = await self.send([("one.txt", b"alpha")])
        original_finish = self.store.finish

        async def commit_then_fail(claim, **kwargs):
            """Commit the real ready state and simulate a lost database acknowledgement."""
            await original_finish(claim, **kwargs)
            raise ConnectionError("private DSN")

        with patch.object(self.store, "finish", commit_then_fail):
            worker = await self.start_worker()
            await self.wait_for_state(Submission, send["submission_id"], "ready")
            await worker.stop()
        self.assertEqual(self.index.collection.count(), 1)
        self.assertEqual(len(await self.retriever.search(self.conversation_id, "alpha")), 1)

    async def test_reindexing_historical_file_does_not_restart_completed_ai(self):
        """Rebuild an old document's vectors while preserving its completed answer and run."""
        send = await self.send([("old.txt", b"historical alpha")])
        answer_id = uuid4()
        with self.engine.begin() as connection:
            connection.execute(
                sa.insert(Message).values(
                    id=answer_id,
                    conversation_id=self.conversation_id,
                    run_id=UUID(send["run_id"]),
                    role="assistant",
                    content="Keep historical answer",
                )
            )
            connection.execute(
                sa.update(Run)
                .where(Run.id == UUID(send["run_id"]))
                .values(status="completed", result_message_id=answer_id)
            )
            connection.execute(sa.update(Submission).values(status="ready"))
        worker = await self.start_worker()
        await self.wait_for_state(Attachment, send["documents"][0]["id"], "ready")
        await worker.stop()
        self.assertEqual((await self.row(Run, send["run_id"]))["status"], "completed")
        self.assertEqual((await self.row(Message, answer_id))["content"], "Keep historical answer")
        self.assertEqual(len(await self.retriever.search(self.conversation_id, "alpha")), 1)

    async def test_cancellation_keeps_file_pending_and_never_publishes_partial_index(self):
        """Observe a run cancellation during parsing and leave the file unavailable to retrieval."""
        send = await self.send([("cancel.txt", b"alpha")])
        started = asyncio.Event()

        async def blocked_parser(file_path, file_name, settings):
            """Wait for the heartbeat to cancel this parser after the run flag changes."""
            started.set()
            await asyncio.Event().wait()

        worker = await self.start_worker(parser=blocked_parser)
        await asyncio.wait_for(started.wait(), 3)
        with self.engine.begin() as connection:
            connection.execute(
                sa.update(Run)
                .where(Run.id == UUID(send["run_id"]))
                .values(status="cancel_requested", cancel_requested=True)
            )
        await self.wait_for_state(Attachment, send["documents"][0]["id"], "pending")
        await worker.stop()
        self.assertEqual(await self.retriever.search(self.conversation_id, "alpha"), [])
        self.assertEqual(self.index.collection.count(), 0)
        self.assertIsNone(await self.claim())

    async def test_retry_waits_for_indexing_siblings_before_resetting_failed_send(self):
        """Do not reset a failed submission while another attachment still owns active indexing."""
        send = await self.send([("a.txt", b"alpha"), ("b.txt", b"beta")])
        first = await self.claim(limit=2)
        second = await self.claim(limit=2)
        await self.store.finish(first, chunk_count=0, error_code="UNREADABLE_DOCUMENT")
        url = f"/api/v1/conversations/{self.conversation_id}/submissions/{send['submission_id']}/retry-indexing"
        response = await self.client.post(url)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "INDEXING_BUSY")
        await self.store.finish(second, chunk_count=1)
        self.assertEqual((await self.client.post(url)).status_code, 202)

    async def test_shutdown_keeps_native_write_lock_until_thread_really_finishes(self):
        """Wait for a running embedding batch before releasing its file lock, even after repeated stop."""
        import threading

        send = await self.send([("native.txt", b"alpha")])
        started, released = threading.Event(), threading.Event()
        original_write = self.index.upsert_batch

        def delayed_write(claim, chunks, offset):
            """Hold a native write until the case verifies cancellation did not release the lock."""
            started.set()
            if not released.wait(timeout=5):
                raise RuntimeError("fixture did not release native write")
            original_write(claim, chunks, offset)

        async def local_parser(file_path, file_name, settings):
            """Return one in-memory section so this case controls only the embedding thread."""
            return [DocumentSection("alpha")]

        with patch.object(self.index, "upsert_batch", delayed_write):
            worker = await self.start_worker(parser=local_parser)
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            try:
                await worker.stop()
                await worker.stop()
                from types import SimpleNamespace

                self.assertIsNone(
                    worker._acquire_file_lock(
                        SimpleNamespace(attachment_id=UUID(send["documents"][0]["id"]))
                    )
                )
            finally:
                released.set()
            await self.wait_for_state(Attachment, send["documents"][0]["id"], "pending")
            await worker.stop()
        self.assertEqual(self.index.collection.count(), 0)
