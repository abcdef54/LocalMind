"""Check Send acceptance/retry and file cleanup using temporary storage and a disposable DB."""

import asyncio
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

import httpx

from fastapi import FastAPI
from fastapi.testclient import TestClient
import sqlalchemy as sa
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.backend.api.application import install_api_foundation
from src.backend.api.v1.dependencies import get_session
from src.backend.api.v1.submissions import get_upload_settings
from src.backend.database.migration import upgrade_database
from src.backend.database.models import Attachment, Conversation, Message, Run, Submission
from src.backend.uploads.config import UploadSettings
from src.backend.uploads.storage import SubmissionFileBatch, remove_orphaned_upload_batches

TEST_URL = os.getenv("LOCALMIND_MIGRATION_TEST_URL")


@unittest.skipUnless(TEST_URL, "LOCALMIND_MIGRATION_TEST_URL is not configured")
class SubmissionApiTests(unittest.TestCase):
    """Exercise multipart API calls without document indexing, inference or production storage."""

    @classmethod
    def setUpClass(cls):
        """Open the disposable test database, rejecting any other database name.

        Require localmind_migration_tests because individual cases drop its public schema.
        Create a synchronous engine for fixtures; never use the application's database.
        """
        url = make_url(TEST_URL)
        if url.database != "localmind_migration_tests":
            raise RuntimeError("Tests require disposable database localmind_migration_tests")
        cls.engine = sa.create_engine(url.set(drivername="postgresql+psycopg2"))

    @classmethod
    def tearDownClass(cls):
        """Dispose the synchronous fixture engine after all cases finish."""
        cls.engine.dispose()

    def setUp(self):
        """Recreate the disposable schema and use a temporary upload root with small limits for
        HTTP calls.
        """
        with self.engine.begin() as connection:
            connection.execute(sa.text("DROP SCHEMA public CASCADE"))
            connection.execute(sa.text("CREATE SCHEMA public"))
            upgrade_database(connection)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.settings = UploadSettings(
            root=Path(self.temporary.name), max_file_bytes=20, max_batch_bytes=30
        )
        self.async_engine = create_async_engine(TEST_URL, poolclass=NullPool)
        self.sessions = async_sessionmaker(self.async_engine, expire_on_commit=False)
        self.addCleanup(lambda: asyncio.run(self.async_engine.dispose()))

        async def session_dependency():
            """Yield a session connected only to the disposable database for this test request."""
            async with self.sessions() as session:
                yield session

        app = FastAPI()
        self.root_app = app
        self.api = install_api_foundation(app)
        self.api.dependency_overrides[get_session] = session_dependency
        self.api.dependency_overrides[get_upload_settings] = lambda: self.settings
        self.client = TestClient(app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)
        self.conversation_id = self.seed_conversation()

    def seed_conversation(self):
        """Insert a fresh live conversation in the disposable DB and return its UUID."""
        identity = uuid4()
        with self.engine.begin() as connection:
            connection.execute(sa.insert(Conversation).values(id=identity, title="Upload test"))
        return identity

    def url(self, conversation_id=None):
        """Return the submission URL for conversation_id or this case's default conversation."""
        return f"/api/v1/conversations/{conversation_id or self.conversation_id}/submissions"

    def payload(self, files=(), **overrides):
        """Return JSON-ready Send fields whose manifest matches (filename, bytes) files; apply
        explicit overrides.
        """
        return {
            "client_submission_id": str(uuid4()),
            "content": "Compare these files",
            "settings": {"model_name": "local"},
            "files_manifest": [
                {"client_file_id": str(index), "name": name, "size_bytes": len(content)}
                for index, (name, content) in enumerate(files)
            ],
            **overrides,
        }

    def send(self, files=(), payload=None, conversation_id=None, **kwargs):
        """POST a multipart payload and file tuples, optionally selecting a conversation or extra
        HTTP arguments.
        """
        payload = payload if payload is not None else self.payload(files)
        parts = [("payload", (None, json.dumps(payload)))]
        parts.extend(
            ("files", (name, content, "application/octet-stream")) for name, content in files
        )
        return self.client.post(self.url(conversation_id), files=parts, **kwargs)

    def count(self, model):
        """Return the current row count of the supplied ORM model in the disposable database."""
        with self.engine.connect() as connection:
            return connection.scalar(sa.select(sa.func.count()).select_from(model))

    def finish_run(self):
        """Mark fixture runs failed to allow a later send; this simulates completion without
        inference.
        """
        with self.engine.begin() as connection:
            connection.execute(sa.update(Run).values(status="failed"))

    def assert_empty(self):
        """Assert the rejected send left no messages, files, submissions, runs or
        staging/permanent batch folders.
        """
        for model in (Message, Attachment, Submission, Run):
            self.assertEqual(self.count(model), 0)
        self.assertEqual(list(self.settings.root.glob("staging/*")), [])
        self.assertEqual(list(self.settings.root.glob("blobs/*")), [])

    def test_two_files_accept_and_history_distinguishes_attachments_from_citations(self):
        """Save both files and expose them as user attachments, with no citations or private disk
        paths.
        """
        files = [("notes.py", b"print(1)"), ("notes.cpp", b"int main() {}")]
        response = self.send(files)
        self.assertEqual(response.status_code, 202, response.text)
        accepted = response.json()
        self.assertEqual(accepted["status"], "accepted")
        self.assertEqual(len(accepted["documents"]), 2)
        with self.engine.connect() as connection:
            self.assertEqual(connection.scalar(sa.select(Run.status)), "pending")
            documents = connection.execute(sa.select(Attachment)).mappings().all()
            for document in documents:
                self.assertEqual(document["status"], "pending")
                self.assertEqual(
                    Path(document["file_path"]).read_bytes(), dict(files)[document["file_name"]]
                )
                self.assertNotIn(document["file_name"], document["storage_key"])
        history = self.client.get(f"/api/v1/conversations/{self.conversation_id}/messages").json()
        self.assertEqual(len(history["items"][0]["attachments"]), 2)
        self.assertEqual(history["items"][0]["citations"], [])
        self.assertNotIn(self.temporary.name, json.dumps(history))
        self.assertEqual(list(self.settings.root.glob("staging/*")), [])

    def test_text_only_queues_and_empty_send_rejected(self):
        """Reject an empty Send, but queue valid text-only input even with the longest allowed
        model name.
        """
        response = self.send(payload=self.payload(content="  "))
        self.assertEqual(response.status_code, 422)
        self.assert_empty()
        response = self.send(payload=self.payload(settings={"model_name": "x" * 255}))
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()["status"], "ready")
        with self.engine.connect() as connection:
            self.assertEqual(connection.scalar(sa.select(Run.status)), "queued")

    def test_file_only_send_is_accepted(self):
        """Accept an attachment-only send when its text is empty."""
        response = self.send(
            [("report.pdf", b"%PDF-1.7")],
            payload=self.payload([("report.pdf", b"%PDF-1.7")], content=""),
        )
        self.assertEqual(response.status_code, 202, response.text)

    def test_same_name_in_batch_rejected_but_different_extension_allowed(self):
        """Reject two identical complete filenames while allowing the same stem with different
        extensions.
        """
        response = self.send([("same.py", b"a"), ("same.py", b"b")])
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["error"]["code"], "DUPLICATE_FILE_NAME")
        self.assert_empty()
        self.assertEqual(self.send([("same.py", b"a"), ("same.cpp", b"a")]).status_code, 202)

    def test_existing_name_rejected_other_conversation_and_deleted_name_allowed(self):
        """Block live same-chat filenames, allow another chat, and allow reuse after deletion."""
        files = [("same.txt", b"a")]
        self.assertEqual(self.send(files).status_code, 202)
        self.finish_run()
        response = self.send([("same.txt", b"changed")])
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["error"]["code"], "DUPLICATE_FILE_NAME")
        self.assertEqual(
            self.send(files, conversation_id=self.seed_conversation()).status_code, 202
        )
        with self.engine.begin() as connection:
            connection.execute(
                sa.update(Attachment)
                .where(Attachment.conversation_id == self.conversation_id)
                .values(deleted_at=datetime(2026, 1, 1), status="deleted")
            )
        self.assertEqual(self.send(files).status_code, 202)

    def test_legacy_attachment_name_is_also_checked(self):
        """Reject a name already used by an undeleted old attachment without reading its old file
        path.
        """
        with self.engine.begin() as connection:
            connection.execute(
                sa.insert(Attachment).values(
                    conversation_id=self.conversation_id,
                    file_name="legacy.pdf",
                    file_path="/legacy/no-read",
                    file_type="application/pdf",
                    file_size=1,
                )
            )
        response = self.send([("legacy.pdf", b"pdf")])
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.count(Submission), 0)

    def test_retry_and_lookup_return_same_ids_even_while_run_active(self):
        """Replay a matching Send once, return identical IDs on lookup and reject changed retry
        metadata.
        """
        files = [("a.txt", b"ab")]
        payload = self.payload(files)
        first = self.send(files, payload).json()
        second = self.send(files, payload)
        self.assertEqual(second.status_code, 202, second.text)
        self.assertEqual(first, second.json())
        lookup = self.client.get(self.url() + "/by-client-id/" + payload["client_submission_id"])
        self.assertEqual(lookup.json(), first)
        self.assertEqual(self.count(Submission), 1)
        self.assertEqual(self.count(Attachment), 1)
        self.assertEqual(len(list(self.settings.root.glob("blobs/*"))), 1)
        payload["content"] = "changed"
        response = self.send(files, payload)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "IDEMPOTENCY_CONFLICT")

    def test_metadata_only_retry_does_not_hash_file_content(self):
        """Return the original send for the same key/name/size even when replacement file bytes
        differ.
        """
        files = [("a.txt", b"ab")]
        payload = self.payload(files)
        first = self.send(files, payload).json()
        # Explicit v1 policy: equal names/sizes with the same client key replay
        # the first acceptance, even if replacement bytes differ.
        second = self.send([("a.txt", b"cd")], payload)
        self.assertEqual(second.json(), first)

    def test_concurrent_same_key_accepts_once(self):
        """Submit the same client key simultaneously and require one saved user message/file
        batch.
        """
        files = [("a.txt", b"ab")]
        payload = self.payload(files)

        async def concurrent_sends():
            """Send two requests on one ASGI event loop and return both responses after they
            finish.
            """
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.root_app), base_url="http://test"
            ) as client:

                async def send_once():
                    """Post one copy of the shared payload/file so the conversation lock resolves
                    the race.
                    """
                    parts = [("payload", (None, json.dumps(payload))), ("files", ("a.txt", b"ab"))]
                    return await client.post(self.url(), files=parts)

                return await asyncio.gather(send_once(), send_once())

        responses = asyncio.run(concurrent_sends())
        self.assertTrue(
            all(response.status_code == 202 for response in responses), [r.text for r in responses]
        )
        self.assertEqual(responses[0].json(), responses[1].json())
        self.assertEqual(self.count(Message), 1)
        self.assertEqual(len(list(self.settings.root.glob("blobs/*"))), 1)

    def test_active_run_rejects_new_submission_and_cleans_files(self):
        """Keep the original send and remove new staging files when another send hits RUN_ACTIVE."""
        self.assertEqual(self.send().status_code, 202)
        response = self.send([("a.txt", b"a")])
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.count(Submission), 1)
        self.assertEqual(list(self.settings.root.glob("staging/*")), [])

    def test_file_and_batch_limits_cleanup_all_files(self):
        """Reject per-file/combined oversize batches completely, but accept a file exactly at the
        limit.
        """
        for files in (
            [("small.txt", b"a"), ("large.txt", b"x" * 21)],
            [("a.txt", b"x" * 20), ("b.txt", b"y" * 11)],
        ):
            with self.subTest(files=[name for name, _ in files]):
                response = self.send(files)
                self.assertEqual(response.status_code, 413, response.text)
                self.assert_empty()
        self.assertEqual(self.send([("exact.txt", b"x" * 20)]).status_code, 202)

    def test_count_type_and_manifest_errors_leave_no_rows_or_blobs(self):
        """Reject file count/type/path and manifest mismatches without leaving partial accepted
        data.
        """
        cases = [
            ([("bad.exe", b"a")], None, 415),
            ([("../bad.py", b"a")], None, 422),
            ([("a.txt", b"a")], self.payload(), 422),
            ([("a.txt", b"a")], self.payload([("a.txt", b"aa")]), 422),
            (
                [("a.txt", b"a"), ("b.txt", b"b")],
                self.payload([("b.txt", b"b"), ("a.txt", b"a")]),
                422,
            ),
        ]
        for files, payload, expected in cases:
            with self.subTest(files=files):
                response = self.send(files, payload)
                self.assertEqual(response.status_code, expected, response.text)
                self.assert_empty()
        self.settings = replace(self.settings, max_files=1)
        self.assertEqual(self.send([("a.txt", b"a"), ("b.txt", b"b")]).status_code, 413)
        self.assert_empty()

    def test_invalid_payload_key_and_scope(self):
        """Reject invalid settings/text/header keys and unknown/deleted conversations without
        saved sends.
        """
        for payload in (
            self.payload(settings={"model_name": "local", "ranking_group_size": 1}),
            self.payload(content=1),
        ):
            self.assertEqual(self.send(payload=payload).status_code, 422)
            self.assert_empty()
        self.assertEqual(self.send(headers={"Idempotency-Key": str(uuid4())}).status_code, 422)
        self.assertEqual(self.send(conversation_id=uuid4()).status_code, 404)
        with self.engine.begin() as connection:
            connection.execute(sa.update(Conversation).values(deleted_at=datetime(2026, 1, 1)))
        self.assertEqual(self.send().status_code, 404)
        self.assert_empty()

    def test_storage_failure_rolls_back_database_and_cleans_batch(self):
        """Roll back metadata and remove staging when moving files fails; hide the private disk
        error.
        """
        with patch.object(
            SubmissionFileBatch,
            "move_to_permanent_storage",
            side_effect=OSError("private disk error"),
        ):
            response = self.send([("a.txt", b"a")])
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private disk error", response.text)
        self.assert_empty()

    def test_commit_failure_retains_blobs_until_safe_reconciliation(self):
        """Keep moved files on uncertain commit, then delete them only after a later check finds
        no submission.
        """
        from sqlalchemy.orm.session import SessionTransaction

        original_commit = SessionTransaction.commit

        def uncertain_commit(transaction, *args, **kwargs):
            """Fail the submission transaction commit after file movement, without affecting the
            earlier read transaction.
            """
            if (
                transaction.parent is None
                and transaction.session.scalar(sa.select(Submission.id)) is not None
            ):
                raise RuntimeError("uncertain commit")
            return original_commit(transaction, *args, **kwargs)

        with patch.object(SessionTransaction, "commit", uncertain_commit):
            response = self.send([("a.txt", b"a")])
        self.assertEqual(response.status_code, 500)
        self.assertEqual(self.count(Submission), 0)
        self.assertEqual(len(list(self.settings.root.glob("blobs/*"))), 1)
        self.age_batches()
        self.assertEqual(asyncio.run(self.reconcile()), 1)
        self.assert_empty()

    def age_batches(self):
        """Set batch directory timestamps 100 seconds into the past to test cleanup without
        sleeping.
        """
        for directory in list(self.settings.root.glob("blobs/*")) + list(
            self.settings.root.glob("staging/*")
        ):
            os.utime(directory, (time.time() - 100, time.time() - 100))

    async def reconcile(self):
        """Run abandoned-folder cleanup with a one-second grace using a disposable-DB session;
        return count.
        """
        async with self.sessions() as session:
            return await remove_orphaned_upload_batches(session, self.settings.root, 1)

    def test_reconciliation_preserves_committed_and_live_batches(self):
        """Delete an old orphan while retaining a saved submission folder and a still-locked
        request folder.
        """
        self.assertEqual(self.send([("a.txt", b"a")]).status_code, 202)
        orphan = SubmissionFileBatch(self.settings.root)
        orphan.create_staging_directory()
        orphan.keep_files_after_request = True
        orphan.cleanup_and_release_lock()
        live = SubmissionFileBatch(self.settings.root)
        live.create_staging_directory()
        live.keep_files_after_request = True
        self.addCleanup(live.cleanup_and_release_lock)
        self.age_batches()
        self.assertEqual(asyncio.run(self.reconcile()), 1)
        self.assertTrue(live.current_directory.exists())
        self.assertEqual(len(list(self.settings.root.glob("blobs/*"))), 1)

    def test_openapi_documents_multipart_and_lookup(self):
        """Document multipart/SubmissionInput and return 404 when a client key has no saved send."""
        schema = self.client.get("/api/v1/openapi.json").json()
        operation = schema["paths"]["/conversations/{conversation_id}/submissions"]["post"]
        self.assertIn("multipart/form-data", operation["requestBody"]["content"])
        self.assertIn("SubmissionInput", schema["components"]["schemas"])
        self.assertEqual(
            self.client.get(self.url() + "/by-client-id/" + str(uuid4())).status_code, 404
        )

    def test_upload_concurrency_limit_rejects_without_receiving_second_body(self):
        """Return 429/Retry-After while the only receiver slot is held, before parsing the second
        body.
        """
        self.settings = replace(self.settings, max_parallel_uploads=1)

        async def competing_uploads():
            """Hold the first streaming request open, issue a second request, then release the
            first for cleanup.
            """
            started, release = asyncio.Event(), asyncio.Event()

            async def blocked_body():
                """Signal that receiving started and wait for the case to release the fake body."""
                started.set()
                await release.wait()
                yield b"--unused--\r\n"

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.root_app), base_url="http://test"
            ) as client:
                first = asyncio.create_task(
                    client.post(
                        self.url(),
                        content=blocked_body(),
                        headers={"Content-Type": "multipart/form-data; boundary=unused"},
                    )
                )
                await started.wait()
                try:
                    second = await client.post(self.url(), content=b"not-parsed")
                    self.assertEqual(second.status_code, 429, second.text)
                    self.assertEqual(second.headers["Retry-After"], "1")
                finally:
                    release.set()
                    await first

        asyncio.run(competing_uploads())
        self.assert_empty()

    def test_failure_after_blob_promotion_rolls_back_and_removes_final_directory(self):
        """Remove already moved files if response construction fails before a DB commit is
        attempted.
        """
        with patch(
            "src.backend.services.submissions.build_submission_response",
            side_effect=RuntimeError("response failed"),
        ):
            response = self.send([("a.txt", b"a")])
        self.assertEqual(response.status_code, 500)
        self.assert_empty()

    def test_lost_commit_ack_preserves_accepted_files_and_retry_identity(self):
        """Keep files after an actual commit with lost acknowledgement, and recover the same IDs
        through lookup/retry.
        """
        from sqlalchemy.orm.session import SessionTransaction

        original_commit = SessionTransaction.commit

        def commit_then_lose_ack(transaction, *args, **kwargs):
            """Commit the submission successfully, then raise a fake lost acknowledgement after
            the database changed.
            """
            has_submission = (
                transaction.parent is None
                and transaction.session.scalar(sa.select(Submission.id)) is not None
            )
            original_commit(transaction, *args, **kwargs)
            if has_submission:
                raise RuntimeError("commit acknowledgement lost")

        files = [("a.txt", b"a")]
        payload = self.payload(files)
        with patch.object(SessionTransaction, "commit", commit_then_lose_ack):
            response = self.send(files, payload)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(self.count(Submission), 1)
        self.age_batches()
        self.assertEqual(asyncio.run(self.reconcile()), 0)
        lookup = self.client.get(self.url() + "/by-client-id/" + payload["client_submission_id"])
        self.assertEqual(lookup.status_code, 200, lookup.text)
        retried = self.send(files, payload)
        self.assertEqual(retried.status_code, 202, retried.text)
        self.assertEqual(retried.json(), lookup.json())
        self.assertEqual(len(list(self.settings.root.glob("blobs/*"))), 1)
