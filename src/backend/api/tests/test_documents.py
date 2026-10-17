"""Exercise document HTTP contracts and cleanup using disposable storage/PostgreSQL."""

import asyncio
from datetime import datetime, timezone, timedelta
import fcntl
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

import sqlalchemy as sa

from src.backend.api.tests import test_conversations as conversation_tests

TEST_URL = conversation_tests.TEST_URL
from src.backend.database.models import Attachment, Message, Submission, Run
from src.backend.indexing.tests.fixtures import TinyEmbeddings
from src.backend.indexing.vectors import DocumentVectorIndex
from src.backend.uploads.config import UploadSettings
from src.backend.uploads.document_cleanup import DocumentCleanupWorker


@unittest.skipUnless(TEST_URL, "LOCALMIND_MIGRATION_TEST_URL is not configured")
class DocumentApiTests(unittest.TestCase):
    """Check scope, bounded lists, safe previews and restartable deletion."""

    setUpClass = classmethod(conversation_tests.ConversationApiTests.setUpClass.__func__)
    tearDownClass = classmethod(conversation_tests.ConversationApiTests.tearDownClass.__func__)
    seed_conversation = conversation_tests.ConversationApiTests.seed_conversation

    def setUp(self) -> None:
        """Create a disposable schema, HTTP client and isolated upload directory."""
        conversation_tests.ConversationApiTests.setUp(self)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.settings_patch = patch(
            "src.backend.api.v1.documents.UploadSettings.from_environment",
            return_value=UploadSettings(root=self.root),
        )
        self.settings_patch.start()
        self.addCleanup(self.settings_patch.stop)
        self.conversation_id = self.seed_conversation()
        self.prefix = f"/api/v1/conversations/{self.conversation_id}/documents"

    def seed_document(self, name: str = "report.txt", *, conversation_id=None) -> tuple:
        """Save one legacy-style attachment and local bytes; return UUID and path."""
        document_id = uuid4()
        path = self.root / str(document_id)
        path.write_bytes(b"example document content")
        with self.engine.begin() as connection:
            connection.execute(
                sa.insert(Attachment).values(
                    id=document_id,
                    conversation_id=conversation_id or self.conversation_id,
                    file_name=name,
                    file_path=str(path),
                    file_type="text/plain",
                    file_size=path.stat().st_size,
                    status="ready",
                    created_at=datetime(2026, 1, 1),
                )
            )
        # Legacy paths are confined to UPLOAD_DIR, independent of configured new-upload root.
        legacy_patch = patch("src.backend.constants.UPLOAD_DIR", self.root)
        legacy_patch.start()
        self.addCleanup(legacy_patch.stop)
        return document_id, path

    def test_list_pages_scope_and_metadata(self) -> None:
        """Fetch stable pages without private paths; reject foreign document/cursor IDs."""
        ids = [self.seed_document()[0] for _ in range(3)]
        first = self.client.get(self.prefix, params={"limit": 2})
        self.assertEqual(first.status_code, 200, first.text)
        page = first.json()
        self.assertTrue(page["has_more"])
        self.assertEqual(len(page["items"]), 2)
        self.assertNotIn("file_path", str(page))
        second = self.client.get(
            self.prefix, params={"limit": 2, "before": page["next_cursor"]}
        ).json()
        self.assertEqual(len(second["items"]), 1)
        self.assertEqual(
            {row["id"] for row in page["items"] + second["items"]}, {str(value) for value in ids}
        )
        other_chat = self.seed_conversation()
        foreign_id, _ = self.seed_document(conversation_id=other_chat)
        self.assertEqual(self.client.get(f"{self.prefix}/{foreign_id}").status_code, 404)
        self.assertEqual(self.client.delete(f"{self.prefix}/{foreign_id}").status_code, 404)
        wrong_page = self.client.get(
            f"/api/v1/conversations/{other_chat}/documents", params={"before": page["next_cursor"]}
        )
        self.assertEqual(wrong_page.status_code, 422)
        self.assertEqual(self.client.get(self.prefix, params={"limit": 101}).status_code, 422)

    def test_safe_preview_download_and_missing_content(self) -> None:
        """Serve HTML as text, encode Unicode names, download DOCX and handle missing bytes."""
        document_id, path = self.seed_document("résumé.html")
        response = self.client.get(f"{self.prefix}/{document_id}/content")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.content, path.read_bytes())
        self.assertTrue(response.headers["content-type"].startswith("text/plain"))
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertIn("%C3%A9", response.headers["content-disposition"])
        self.assertTrue(
            self.client.get(f"{self.prefix}/{document_id}/content?download=true")
            .headers["content-disposition"]
            .startswith("attachment")
        )
        docx_id, _ = self.seed_document("report.docx")
        self.assertTrue(
            self.client.get(f"{self.prefix}/{docx_id}/content")
            .headers["content-disposition"]
            .startswith("attachment")
        )
        pdf_id, _ = self.seed_document("report.pdf")
        self.assertEqual(
            self.client.get(f"{self.prefix}/{pdf_id}/content").headers["content-type"],
            "application/pdf",
        )
        path.unlink()
        self.assertEqual(self.client.get(f"{self.prefix}/{document_id}/content").status_code, 404)
        self.assertEqual(self.client.get(f"{self.prefix}/{document_id}").status_code, 200)

    def test_path_escape_is_neither_served_nor_deleted(self) -> None:
        """Reject a saved path outside managed uploads, leaving unrelated bytes intact."""
        document_id, _ = self.seed_document()
        with tempfile.TemporaryDirectory() as other:
            external = Path(other) / "private.txt"
            external.write_text("private")
            with self.engine.begin() as connection:
                connection.execute(
                    sa.update(Attachment)
                    .where(Attachment.id == document_id)
                    .values(file_path=str(external))
                )
            self.assertEqual(
                self.client.get(f"{self.prefix}/{document_id}/content").status_code, 404
            )
            self.client.delete(f"{self.prefix}/{document_id}")
            index = DocumentVectorIndex(self.root / "vectors", TinyEmbeddings())
            worker = DocumentCleanupWorker(self.sessions, index, self.root)
            self.assertFalse(asyncio.run(worker.cleanup_deleted_document(document_id)))
            self.assertTrue(external.exists())

    def test_deletion_preserves_citations_and_cleanup_retries(self) -> None:
        """Hide immediately, keep historical citation and retry a failed physical deletion."""
        document_id, path = self.seed_document()
        from src.backend.database.models import message_attachments

        message_id = uuid4()
        with self.engine.begin() as connection:
            connection.execute(
                sa.insert(Message).values(
                    id=message_id,
                    conversation_id=self.conversation_id,
                    role="assistant",
                    content="answer",
                )
            )
            connection.execute(
                sa.insert(message_attachments).values(
                    message_id=message_id, attachment_id=document_id
                )
            )
        self.assertEqual(self.client.delete(f"{self.prefix}/{document_id}").status_code, 204)
        self.assertEqual(self.client.delete(f"{self.prefix}/{document_id}").status_code, 204)
        self.assertEqual(self.client.get(f"{self.prefix}/{document_id}").status_code, 404)
        self.assertEqual(self.client.get(self.prefix).json()["items"], [])
        history = self.client.get(f"/api/v1/conversations/{self.conversation_id}/messages").json()
        self.assertTrue(history["items"][0]["citations"][0]["deleted"])
        index = DocumentVectorIndex(self.root / "vectors", TinyEmbeddings())
        foreign_id = uuid4()
        for identity, file_id in (
            ("old", document_id),
            ("partial", document_id),
            ("foreign", foreign_id),
        ):
            index.collection.upsert(
                ids=[identity],
                embeddings=[[1.0, 0.0, 0.0, 0.0]],
                documents=[identity],
                metadatas=[
                    {"conversation_id": str(self.conversation_id), "source_file_id": str(file_id)}
                ],
            )
        legacy = index.client.get_or_create_collection("langchain", embedding_function=None)
        legacy.upsert(
            ids=["legacy"],
            embeddings=[[1.0, 0.0, 0.0, 0.0]],
            documents=["legacy"],
            metadatas=[{"source_file_id": str(document_id)}],
        )
        worker = DocumentCleanupWorker(self.sessions, index, self.root)
        with patch.object(
            index, "delete_document_vectors", side_effect=RuntimeError("private exception")
        ):
            self.assertFalse(asyncio.run(worker.cleanup_deleted_document(document_id)))
        self.assertTrue(path.exists())
        with self.engine.connect() as connection:
            self.assertEqual(
                connection.scalar(
                    sa.select(Attachment.cleanup_error).where(Attachment.id == document_id)
                ),
                "DOCUMENT_CLEANUP_FAILED",
            )
        # Reconstruct worker to prove pending cleanup belongs to PostgreSQL, not memory.
        worker = DocumentCleanupWorker(self.sessions, index, self.root)
        with self.engine.begin() as connection:
            connection.execute(
                sa.update(Attachment)
                .where(Attachment.id == document_id)
                .values(cleanup_attempted_at=datetime.now(timezone.utc) - timedelta(seconds=31))
            )
        self.assertEqual(asyncio.run(worker.retry_pending_document_cleanup()), 1)
        self.assertFalse(path.exists())
        self.assertEqual(index.collection.get()["ids"], ["foreign"])
        self.assertEqual(legacy.count(), 0)
        self.assertFalse(asyncio.run(worker.cleanup_deleted_document(document_id)))
        with self.engine.connect() as connection:
            self.assertIsNotNone(
                connection.scalar(
                    sa.select(Attachment.cleanup_completed_at).where(Attachment.id == document_id)
                )
            )

    def test_cleanup_waits_for_indexing_lock_and_handles_missing_bytes(self) -> None:
        """Retain bytes during native indexing; retry deletion safely after lock release."""
        document_id, path = self.seed_document()
        self.client.delete(f"{self.prefix}/{document_id}")
        index = DocumentVectorIndex(self.root / "vectors", TinyEmbeddings())
        worker = DocumentCleanupWorker(self.sessions, index, self.root)
        locks = self.root / "indexing_locks"
        locks.mkdir()
        with (locks / f"{document_id}.lock").open("ab") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertFalse(asyncio.run(worker.cleanup_deleted_document(document_id)))
            self.assertTrue(path.exists())
        path.unlink()
        self.assertTrue(asyncio.run(worker.cleanup_deleted_document(document_id)))

    def test_deleted_chat_is_hidden_and_its_files_are_cleaned(self) -> None:
        """Conversation archival queues its documents marked as deleted for the same cleanup."""
        document_id, path = self.seed_document()
        response = self.client.delete(f"/api/v1/conversations/{self.conversation_id}")
        self.assertEqual(response.status_code, 204, response.text)
        self.assertEqual(self.client.get(self.prefix).status_code, 404)
        index = DocumentVectorIndex(self.root / "vectors", TinyEmbeddings())
        self.assertTrue(
            asyncio.run(
                DocumentCleanupWorker(self.sessions, index, self.root).cleanup_deleted_document(
                    document_id
                )
            )
        )
        self.assertFalse(path.exists())

    def test_delete_pending_send_fails_run_and_releases_filename(self) -> None:
        """Delete an accepted file before indexing; fail its run and allow a new same-name send."""
        import json
        from src.backend.api.v1.submissions import get_upload_settings

        api = self.client.app.routes[-1].app
        api.dependency_overrides[get_upload_settings] = lambda: UploadSettings(root=self.root)
        url = f"/api/v1/conversations/{self.conversation_id}/submissions"

        def send():
            """Accept one fresh send containing a five-byte text file."""
            payload = {
                "client_submission_id": str(uuid4()),
                "content": "Read it",
                "settings": {"model_name": "test"},
                "files_manifest": [
                    {"client_file_id": "local", "name": "same.txt", "size_bytes": 5}
                ],
            }
            return self.client.post(
                url,
                files=[
                    ("payload", (None, json.dumps(payload))),
                    ("files", ("same.txt", b"hello", "text/plain")),
                ],
            )

        accepted = send()
        self.assertEqual(accepted.status_code, 202, accepted.text)
        document_id = accepted.json()["documents"][0]["id"]
        self.assertEqual(self.client.get(f"{self.prefix}/{document_id}/content").content, b"hello")
        self.assertEqual(self.client.delete(f"{self.prefix}/{document_id}").status_code, 204)
        with self.engine.connect() as connection:
            self.assertEqual(connection.scalar(sa.select(Run.status)), "failed")
            self.assertEqual(connection.scalar(sa.select(Submission.status)), "failed")
        self.assertEqual(send().status_code, 202)
        worker = DocumentCleanupWorker(
            self.sessions, DocumentVectorIndex(self.root / "vectors", TinyEmbeddings()), self.root
        )
        self.assertTrue(asyncio.run(worker.cleanup_deleted_document(UUID(document_id))))

    def test_live_files_are_never_cleaned(self) -> None:
        """Reject physical cleanup when the document has not been marked as deleted."""
        document_id, path = self.seed_document()
        worker = DocumentCleanupWorker(
            self.sessions, DocumentVectorIndex(self.root / "vectors", TinyEmbeddings()), self.root
        )
        self.assertFalse(asyncio.run(worker.cleanup_deleted_document(document_id)))
        self.assertTrue(path.exists())
