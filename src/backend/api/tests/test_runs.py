"""Check run snapshots, event replay/SSE and safe cancel/retry on disposable PostgreSQL."""

import asyncio
import os
import unittest
from unittest.mock import patch
from uuid import uuid4

import sqlalchemy as sa

from src.backend.api.tests import test_conversations as conversation_tests
from src.backend.execution.tests import test_executor as executor_tests
from src.backend.database.models import Attachment, Conversation, Run, RunEvent
from src.backend.execution.events import update_state_and_add_run_event
from src.backend.execution.store import RunResult, RunStore

TEST_URL = os.getenv("LOCALMIND_MIGRATION_TEST_URL")


@unittest.skipUnless(TEST_URL, "LOCALMIND_MIGRATION_TEST_URL is not configured")
class RunApiTests(unittest.TestCase):
    """Exercise actual HTTP responses without model loading or production DB connections."""

    setUpClass = classmethod(conversation_tests.ConversationApiTests.setUpClass.__func__)
    tearDownClass = classmethod(conversation_tests.ConversationApiTests.tearDownClass.__func__)
    seed_run = executor_tests.ExecutorIntegrationTests.seed_run
    seed_conversation = conversation_tests.ConversationApiTests.seed_conversation

    def setUp(self):
        """Reset the disposable schema and seed one queued task."""
        conversation_tests.ConversationApiTests.setUp(self)
        self.store = RunStore(self.sessions)
        self.run_id = self.seed_run()
        self.url = f"/api/v1/runs/{self.run_id}"

    def claim(self):
        """Reserve the task without starting external compute."""
        return asyncio.run(
            self.store.reserve_next_queued_run_for_worker(
                worker_id="api-test", lease_seconds=60, max_concurrency=1
            )
        )

    def test_snapshot_is_scoped_and_exposes_no_owner_or_paths(self):
        """A wrong expected conversation cannot read a run or its event history."""
        snapshot = self.client.get(self.url)
        self.assertEqual(snapshot.status_code, 200, snapshot.text)
        self.assertEqual(snapshot.json()["last_event_sequence"], 0)
        self.assertEqual(self.client.get(self.url + "/result").status_code, 409)
        self.assertTrue(snapshot.json()["compute_stopped"])
        self.assertNotIn("lease_owner", snapshot.json())
        other = self.seed_conversation()
        for suffix in ("", "/event-log", "/events"):
            self.assertEqual(
                self.client.get(
                    self.url + suffix, params={"conversation_id": str(other)}
                ).status_code,
                404,
            )
        self.assertEqual(
            self.client.post(
                self.url + "/cancel", params={"conversation_id": str(other)}
            ).status_code,
            404,
        )
        self.assertEqual(self.client.get(f"/api/v1/runs/{uuid4()}").status_code, 404)

    def test_cancel_before_claim_is_immediate_and_retry_is_explicit(self):
        """Queued work cancels without inference; retry keeps IDs and rejects duplicate requests."""
        cancelled = self.client.post(self.url + "/cancel")
        self.assertEqual(cancelled.status_code, 202, cancelled.text)
        snapshot = cancelled.json()
        self.assertEqual(snapshot["status"], "cancelled")
        self.assertTrue(snapshot["compute_stopped"])
        sequence = snapshot["last_event_sequence"]
        self.assertEqual(
            self.client.post(self.url + "/cancel").json()["last_event_sequence"], sequence
        )
        retried = self.client.post(self.url + "/retry", json={"expected_attempt": 0})
        self.assertEqual(retried.status_code, 202, retried.text)
        self.assertEqual(retried.json()["id"], str(self.run_id))
        self.assertEqual(retried.json()["status"], "queued")
        self.assertEqual(retried.json()["visualization"], {"phase": "queued"})
        self.assertGreater(retried.json()["last_event_sequence"], sequence)
        self.assertEqual(
            self.client.post(self.url + "/retry", json={"expected_attempt": 0}).status_code, 409
        )
        self.assertEqual(
            self.client.post(self.url + "/retry", json={"expected_attempt": True}).status_code, 422
        )
        claim = self.claim()
        self.assertEqual(claim.attempt, 1)

    def test_running_cancel_does_not_confirm_stoppage_or_publish_late_answer(self):
        """Keep the slot reserved until a safe handler boundary explicitly confirms cancellation."""
        claim = self.claim()
        response = self.client.post(self.url + "/cancel")
        self.assertEqual(response.json()["status"], "cancel_requested")
        self.assertFalse(response.json()["compute_stopped"])
        self.assertFalse(
            asyncio.run(
                self.store.save_assistant_reply_and_mark_run_completed(claim, RunResult("too late"))
            )
        )
        self.assertEqual(
            self.client.post(self.url + "/retry", json={"expected_attempt": 1}).status_code, 409
        )
        # The fixture never invoked a model, so compute stoppage is known here.
        self.assertTrue(asyncio.run(self.store.confirm_cancellation(claim)))
        final = self.client.get(self.url).json()
        self.assertEqual(final["status"], "cancelled")
        self.assertTrue(final["compute_stopped"])
        self.assertIsNone(final["result_message_id"])
        self.assertEqual(
            self.client.post(self.url + "/retry", json={"expected_attempt": 0}).status_code, 409
        )

    def test_interrupted_retry_requires_confirmed_compute_stoppage(self):
        """HTTP retry never clears an unknown owner; operator acknowledgement is separate."""
        claim = self.claim()
        asyncio.run(self.store.interrupt(claim, code="COMPUTE_STATE_UNKNOWN"))
        response = self.client.post(self.url + "/retry", json={"expected_attempt": 1})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "COMPUTE_UNCONFIRMED")
        self.assertTrue(
            asyncio.run(
                self.store.release_interrupted_run_slot_after_compute_stop_confirmed(
                    self.run_id, expected_owner=claim.lease_owner
                )
            )
        )
        self.assertTrue(self.client.get(self.url).json()["compute_stopped"])
        self.assertEqual(
            self.client.post(self.url + "/retry", json={"expected_attempt": 1}).status_code, 202
        )

    def test_completion_wins_when_it_commits_before_cancel(self):
        """Cancelling a completed run preserves its one saved answer and terminal event."""
        claim = self.claim()
        self.assertTrue(
            asyncio.run(
                self.store.save_assistant_reply_and_mark_run_completed(claim, RunResult("answer"))
            )
        )
        response = self.client.post(self.url + "/cancel")
        self.assertEqual(response.json()["status"], "completed")
        self.assertIsNotNone(response.json()["result_message_id"])
        result = self.client.get(response.json()["result_url"])
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["content"], "answer")
        self.assertEqual(
            self.client.post(self.url + "/retry", json={"expected_attempt": 1}).status_code, 409
        )

    def test_event_pages_and_terminal_sse_resume_without_duplicates(self):
        """Replay committed sequence IDs even if inference finished before subscription."""
        claim = self.claim()
        asyncio.run(
            self.store.save_assistant_reply_and_mark_run_completed(claim, RunResult("answer"))
        )
        page = self.client.get(self.url + "/event-log", params={"limit": 1}).json()
        self.assertEqual(page["items"][0]["type"], "run.started")
        self.assertTrue(page["has_more"])
        response = self.client.get(self.url + "/events")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["x-accel-buffering"], "no")
        self.assertIn(f"id: {self.run_id}:1", response.text)
        self.assertIn(f"id: {self.run_id}:2", response.text)
        resumed = self.client.get(
            self.url + "/events", headers={"Last-Event-ID": f"{self.run_id}:1"}
        )
        self.assertNotIn(f"id: {self.run_id}:1\n", resumed.text)
        self.assertIn(f"id: {self.run_id}:2", resumed.text)
        self.assertEqual(
            self.client.get(
                self.url + "/events", headers={"Last-Event-ID": f"{uuid4()}:1"}
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.get(self.url + "/events", params={"after": 999}).status_code, 409
        )
        automatic_reconnect = self.client.get(
            self.url + "/events", params={"after": 1}, headers={"Last-Event-ID": f"{self.run_id}:2"}
        )
        self.assertEqual(automatic_reconnect.status_code, 200)
        self.assertNotIn("id:", automatic_reconnect.text)
        self.assertEqual(self.client.app.routes[-1].app.state.run_stream_count, 0)

    def test_retention_requires_snapshot_resync_but_sequence_never_resets(self):
        """Expired event cursors return 410 and a usable full snapshot remains available."""

        async def write_events():
            """Use a tiny retention cap to test expiry without generating thousands of rows."""
            async with self.sessions() as session, session.begin():
                run = await session.scalar(
                    sa.select(Run).where(Run.id == self.run_id).with_for_update()
                )
                for _ in range(5):
                    await update_state_and_add_run_event(session, run, "run.queued", {})

        with patch("src.backend.execution.events.EVENT_RETENTION", 3):
            asyncio.run(write_events())
        self.assertEqual(self.client.get(self.url + "/event-log").status_code, 410)
        self.assertEqual(self.client.get(self.url + "/events").status_code, 410)
        snapshot = self.client.get(self.url).json()
        self.assertEqual(snapshot["last_event_sequence"], 5)
        self.assertEqual(snapshot["first_event_sequence"], 3)
        page = self.client.get(self.url + "/event-log", params={"after": 2}).json()
        self.assertEqual([row["sequence"] for row in page["items"]], [3, 4, 5])

    def test_subscriber_capacity_and_archived_conversation(self):
        """Bound connections per process and reject resources in archived conversations."""
        api = self.client.app.routes[-1].app
        api.state.run_stream_count = 32
        self.assertEqual(self.client.get(self.url + "/events").status_code, 429)
        with self.engine.begin() as connection:
            conversation_id = connection.scalar(
                sa.select(Run.conversation_id).where(Run.id == self.run_id)
            )
            connection.execute(
                sa.update(Conversation)
                .where(Conversation.id == conversation_id)
                .values(deleted_at=sa.func.now())
            )
        self.assertEqual(self.client.get(self.url).status_code, 404)

    def test_retry_rejects_other_active_run_in_same_conversation(self):
        """Explicit retry cannot bypass the one-active-run constraint for a chat."""
        with self.engine.begin() as connection:
            connection.execute(sa.update(Run).where(Run.id == self.run_id).values(status="failed"))
            original = connection.execute(sa.select(Run).where(Run.id == self.run_id)).one()
            from src.backend.database.models import Message, Submission

            submission_id, message_id = uuid4(), uuid4()
            connection.execute(
                sa.insert(Submission).values(
                    id=submission_id,
                    conversation_id=original.conversation_id,
                    client_submission_id=uuid4(),
                    fingerprint="a" * 64,
                    settings_snapshot={"model_name": "fake"},
                    status="ready",
                )
            )
            connection.execute(
                sa.insert(Message).values(
                    id=message_id,
                    conversation_id=original.conversation_id,
                    submission_id=submission_id,
                    role="user",
                    content="another send",
                )
            )
            connection.execute(
                sa.insert(Run).values(
                    conversation_id=original.conversation_id,
                    submission_id=submission_id,
                    message_id=message_id,
                    settings_snapshot={"model_name": "fake"},
                    status="queued",
                )
            )
        response = self.client.post(self.url + "/retry", json={"expected_attempt": 0})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "RUN_ACTIVE")

    def test_retry_failed_indexing_waits_for_files_without_reupload(self):
        """A retry retains input/file IDs, resets failed files and keeps the run pending."""
        document_id = uuid4()
        with self.engine.begin() as connection:
            run = connection.execute(sa.select(Run).where(Run.id == self.run_id)).one()
            connection.execute(sa.update(Run).where(Run.id == run.id).values(status="failed"))
            connection.execute(
                sa.insert(Attachment).values(
                    id=document_id,
                    conversation_id=run.conversation_id,
                    submission_id=run.submission_id,
                    message_id=run.message_id,
                    client_file_id="local-1",
                    storage_key=f"blobs/{run.submission_id}/{document_id}",
                    file_path="/unused/file",
                    file_name="report.txt",
                    file_type="text/plain",
                    file_size=10,
                    status="failed",
                )
            )
        response = self.client.post(self.url + "/retry", json={"expected_attempt": 0})
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()["status"], "pending")
        self.assertEqual(response.json()["documents"][0]["status"], "pending")
        self.assertEqual(response.json()["documents"][0]["id"], str(document_id))
        self.client.post(self.url + "/cancel")
        with self.engine.begin() as connection:
            connection.execute(
                sa.update(Attachment)
                .where(Attachment.id == document_id)
                .values(status="deleted", deleted_at=sa.func.now())
            )
        self.assertEqual(
            self.client.post(self.url + "/retry", json={"expected_attempt": 0}).status_code, 409
        )

    def test_old_input_retry_is_rejected_after_a_newer_question(self):
        """Retry must not apply a newer conversation summary to an older failed question."""
        from datetime import timedelta
        from src.backend.database.models import Message

        with self.engine.begin() as connection:
            run = connection.execute(sa.select(Run).where(Run.id == self.run_id)).one()
            connection.execute(sa.update(Run).where(Run.id == run.id).values(status="failed"))
            original_time = connection.scalar(
                sa.select(Message.created_at).where(Message.id == run.message_id)
            )
            connection.execute(
                sa.insert(Message).values(
                    conversation_id=run.conversation_id,
                    role="user",
                    content="newer question",
                    created_at=original_time + timedelta(seconds=1),
                )
            )
        response = self.client.post(self.url + "/retry", json={"expected_attempt": 0})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "RUN_INPUT_SUPERSEDED")
