"""Check PostgreSQL rejects duplicate sends, invalid run states and mismatched message links."""

from concurrent.futures import ThreadPoolExecutor
import os
from threading import Barrier
import unittest
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError

from src.backend.api.v1.schemas import ChatSettings
from src.backend.database.migration import upgrade_database
from src.backend.database.models import Conversation, Message, Run, Submission

TEST_URL = os.getenv("LOCALMIND_MIGRATION_TEST_URL")
SETTINGS = ChatSettings(model_name="local-model", reasoning_mode="high").model_dump()


@unittest.skipUnless(TEST_URL, "LOCALMIND_MIGRATION_TEST_URL is not configured")
class SubmissionRunSchemaTests(unittest.TestCase):
    """Verify send/run database constraints independently of API and model execution."""

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

    def setUp(self) -> None:
        """Recreate the disposable schema, apply migrations and insert a conversation for this
        case.
        """
        with self.engine.begin() as connection:
            connection.execute(sa.text("DROP SCHEMA public CASCADE"))
            connection.execute(sa.text("CREATE SCHEMA public"))
            upgrade_database(connection)
            self.conversation_id = connection.execute(
                sa.insert(Conversation).values(title="Research")
            ).inserted_primary_key[0]

    def submission_values(self, *, conversation_id=None, client_key=None) -> dict:
        """Return valid insert fields with fresh UUIDs, optionally replacing conversation ID or
        client key.
        """
        return {
            "id": uuid4(),
            "conversation_id": conversation_id or self.conversation_id,
            "client_submission_id": client_key or uuid4(),
            "fingerprint": "a" * 64,
            "settings_snapshot": SETTINGS,
        }

    def create_input(self, connection, *, conversation_id=None):
        """Insert a submission and matching user message on connection; return both UUIDs without
        committing.
        """
        values = self.submission_values(conversation_id=conversation_id)
        connection.execute(sa.insert(Submission).values(**values))
        message_id = connection.execute(
            sa.insert(Message).values(
                conversation_id=values["conversation_id"],
                submission_id=values["id"],
                content="Question",
                role="user",
            )
        ).inserted_primary_key[0]
        return values["id"], message_id

    def create_run(self, connection, *, status="pending", conversation_id=None):
        """Insert a send, user message and run with the requested status; return
        run/submission/message UUIDs.
        """
        owner_id = conversation_id or self.conversation_id
        submission_id, message_id = self.create_input(connection, conversation_id=owner_id)
        run_id = connection.execute(
            sa.insert(Run).values(
                conversation_id=owner_id,
                submission_id=submission_id,
                message_id=message_id,
                settings_snapshot=SETTINGS,
                status=status,
            )
        ).inserted_primary_key[0]
        return run_id, submission_id, message_id

    def test_defaults_and_settings_snapshot_round_trip(self) -> None:
        """Check inserted defaults, UTC timestamps and saved settings survive a database read."""
        with self.engine.begin() as connection:
            run_id, submission_id, message_id = self.create_run(connection)
        with self.engine.connect() as connection:
            run = (
                connection.execute(sa.select(Run.__table__).where(Run.id == run_id))
                .mappings()
                .one()
            )
            submission = (
                connection.execute(
                    sa.select(Submission.__table__).where(Submission.id == submission_id)
                )
                .mappings()
                .one()
            )
        self.assertEqual(run["status"], "pending")
        self.assertEqual(submission["status"], "accepted")
        self.assertEqual(run["message_id"], message_id)
        self.assertEqual(run["settings_snapshot"], SETTINGS)
        self.assertEqual(run["attempt"], 0)
        self.assertFalse(run["cancel_requested"])
        self.assertIsNotNone(run["created_at"].tzinfo)
        self.assertIsNotNone(submission["created_at"].tzinfo)

    def test_idempotency_key_is_unique_per_conversation_only(self) -> None:
        """Reject the same client key twice in one chat but allow it in a different chat."""
        client_key = uuid4()
        with self.engine.begin() as connection:
            connection.execute(
                sa.insert(Submission).values(**self.submission_values(client_key=client_key))
            )
            with self.assertRaises(IntegrityError):
                with connection.begin_nested():
                    connection.execute(
                        sa.insert(Submission).values(
                            **self.submission_values(client_key=client_key)
                        )
                    )
            other_id = connection.execute(
                sa.insert(Conversation).values(title="Other")
            ).inserted_primary_key[0]
            connection.execute(
                sa.insert(Submission).values(
                    **self.submission_values(conversation_id=other_id, client_key=client_key)
                )
            )

    def test_concurrent_submissions_with_same_key_commit_only_once(self) -> None:
        """Race two inserts with one client key and require exactly one committed submission."""
        client_key = uuid4()
        start = Barrier(2)

        def insert_submission() -> bool:
            """Wait at the thread barrier, attempt the insert and return False only for
            uniqueness rejection.
            """
            start.wait(timeout=5)
            try:
                with self.engine.begin() as connection:
                    connection.execute(
                        sa.insert(Submission).values(
                            **self.submission_values(client_key=client_key)
                        )
                    )
                return True
            except IntegrityError:
                return False

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(lambda _: insert_submission(), range(2)))
        self.assertEqual(sorted(outcomes), [False, True])
        with self.engine.connect() as connection:
            self.assertEqual(
                connection.scalar(sa.select(sa.func.count()).select_from(Submission)), 1
            )

    def test_submission_has_one_user_message_and_one_run(self) -> None:
        """Reject a second input message or second run linked to the same submission."""
        with self.engine.begin() as connection:
            run_id, submission_id, message_id = self.create_run(connection)
            with self.assertRaises(IntegrityError):
                with connection.begin_nested():
                    connection.execute(
                        sa.insert(Message).values(
                            conversation_id=self.conversation_id,
                            submission_id=submission_id,
                            content="Duplicate question",
                            role="user",
                        )
                    )
            with self.assertRaises(IntegrityError):
                with connection.begin_nested():
                    connection.execute(
                        sa.insert(Run).values(
                            conversation_id=self.conversation_id,
                            submission_id=submission_id,
                            message_id=message_id,
                            settings_snapshot=SETTINGS,
                            status="failed",
                        )
                    )

    def test_only_one_active_run_per_conversation(self) -> None:
        """Reject a second active run until the previous run leaves an active status."""
        with self.engine.begin() as connection:
            run_id, _, _ = self.create_run(connection)
            with self.assertRaises(IntegrityError):
                with connection.begin_nested():
                    self.create_run(connection)
            connection.execute(sa.update(Run).where(Run.id == run_id).values(status="failed"))
            second_run, _, _ = self.create_run(connection)
            self.assertNotEqual(run_id, second_run)

    def test_result_is_unique_and_completed_run_requires_its_own_answer(self) -> None:
        """Require one matching assistant message before completion and reject duplicate run
        answers.
        """
        with self.engine.begin() as connection:
            run_id, _, _ = self.create_run(connection)
            with self.assertRaises(IntegrityError):
                with connection.begin_nested():
                    connection.execute(
                        sa.update(Run).where(Run.id == run_id).values(status="completed")
                    )
            answer_id = connection.execute(
                sa.insert(Message).values(
                    conversation_id=self.conversation_id,
                    run_id=run_id,
                    content="Answer",
                    role="assistant",
                )
            ).inserted_primary_key[0]
            with self.assertRaises(IntegrityError):
                with connection.begin_nested():
                    connection.execute(
                        sa.insert(Message).values(
                            conversation_id=self.conversation_id,
                            run_id=run_id,
                            content="Duplicate",
                            role="assistant",
                        )
                    )
            connection.execute(
                sa.update(Run)
                .where(Run.id == run_id)
                .values(
                    status="completed",
                    result_message_id=answer_id,
                )
            )

    def test_cross_conversation_and_wrong_submission_references_are_rejected(self) -> None:
        """Reject messages/runs borrowing a conversation or input from a different submission."""
        with self.engine.begin() as connection:
            _, submission_id, message_id = self.create_run(connection, status="failed")
            other_id = connection.execute(
                sa.insert(Conversation).values(title="Other")
            ).inserted_primary_key[0]
            with self.assertRaises(IntegrityError):
                with connection.begin_nested():
                    connection.execute(
                        sa.insert(Message).values(
                            conversation_id=other_id,
                            submission_id=submission_id,
                            content="Wrong owner",
                            role="user",
                        )
                    )
            second_submission, _ = self.create_input(connection)
            with self.assertRaises(IntegrityError):
                with connection.begin_nested():
                    connection.execute(
                        sa.insert(Run).values(
                            conversation_id=self.conversation_id,
                            submission_id=second_submission,
                            message_id=message_id,
                            settings_snapshot=SETTINGS,
                            status="failed",
                        )
                    )

    def test_run_cannot_use_another_runs_result(self) -> None:
        """Reject marking one run completed with an assistant message produced for another run."""
        with self.engine.begin() as connection:
            first_run, _, _ = self.create_run(connection, status="failed")
            second_run, _, _ = self.create_run(connection, status="failed")
            answer_id = connection.execute(
                sa.insert(Message).values(
                    conversation_id=self.conversation_id,
                    run_id=second_run,
                    content="Other answer",
                    role="assistant",
                )
            ).inserted_primary_key[0]
            with self.assertRaises(IntegrityError):
                with connection.begin_nested():
                    connection.execute(
                        sa.update(Run)
                        .where(Run.id == first_run)
                        .values(
                            result_message_id=answer_id,
                            status="completed",
                        )
                    )

    def test_invalid_states_fingerprints_snapshots_and_attempts_are_rejected(self) -> None:
        """Reject unsupported statuses, malformed fingerprints, non-object settings and negative
        attempts.
        """
        with self.engine.begin() as connection:
            for invalid in (
                {"status": "unknown"},
                {"fingerprint": "not-sha256"},
                {"settings_snapshot": []},
            ):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(IntegrityError):
                        with connection.begin_nested():
                            connection.execute(
                                sa.insert(Submission).values(
                                    **{**self.submission_values(), **invalid}
                                )
                            )
            run_id, _, _ = self.create_run(connection)
            for invalid in ({"status": "unknown"}, {"attempt": -1}, {"settings_snapshot": []}):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(IntegrityError):
                        with connection.begin_nested():
                            connection.execute(
                                sa.update(Run).where(Run.id == run_id).values(**invalid)
                            )

    def test_legacy_messages_remain_valid_without_submission_or_run(self) -> None:
        """Keep accepting original user/assistant rows whose new submission/run links are empty."""
        with self.engine.begin() as connection:
            for role in ("user", "assistant"):
                connection.execute(
                    sa.insert(Message).values(
                        conversation_id=self.conversation_id,
                        role=role,
                        content="Legacy message",
                    )
                )


if __name__ == "__main__":
    unittest.main()
