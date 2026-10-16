"""Check attachment ownership, live filenames and old-record preservation in PostgreSQL."""

from datetime import datetime
import os
import unittest
from uuid import uuid4

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, insert, select, text, update
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError

from src.backend.database.config import PROJECT_ROOT
from src.backend.database.migration import upgrade_database
from src.backend.database.models import Attachment, Conversation, Message, Submission

TEST_URL = os.getenv("LOCALMIND_MIGRATION_TEST_URL")


@unittest.skipUnless(TEST_URL, "LOCALMIND_MIGRATION_TEST_URL is not configured")
class AttachmentSchemaTests(unittest.TestCase):
    """Verify database constraints for new attachments without reading or writing document
    contents.
    """

    @classmethod
    def setUpClass(cls):
        """Open the disposable test database, rejecting any other database name.

        Require localmind_migration_tests because individual cases drop its public schema.
        Create a synchronous engine for fixtures; never use the application's database.
        """
        url = make_url(TEST_URL)
        if url.database != "localmind_migration_tests":
            raise RuntimeError("Tests require disposable database localmind_migration_tests")
        cls.engine = create_engine(url.set(drivername="postgresql+psycopg2"))

    @classmethod
    def tearDownClass(cls):
        """Dispose the synchronous fixture engine after all cases finish."""
        cls.engine.dispose()

    def setUp(self):
        """Recreate the disposable schema and insert a matching conversation, send and user
        message.
        """
        self.conversation_id, self.submission_id, self.message_id = uuid4(), uuid4(), uuid4()
        with self.engine.begin() as connection:
            connection.execute(text("DROP SCHEMA public CASCADE"))
            connection.execute(text("CREATE SCHEMA public"))
            upgrade_database(connection)
            connection.execute(insert(Conversation).values(id=self.conversation_id, title="Files"))
            connection.execute(
                insert(Submission).values(
                    id=self.submission_id,
                    conversation_id=self.conversation_id,
                    client_submission_id=uuid4(),
                    fingerprint="a" * 64,
                    settings_snapshot={},
                )
            )
            connection.execute(
                insert(Message).values(
                    id=self.message_id,
                    conversation_id=self.conversation_id,
                    submission_id=self.submission_id,
                    content="Attached",
                    role="user",
                )
            )

    def document(self):
        """Return valid attachment insert fields linked to this case's send/message, without
        inserting them.
        """
        return dict(
            id=uuid4(),
            conversation_id=self.conversation_id,
            submission_id=self.submission_id,
            message_id=self.message_id,
            client_file_id=str(uuid4()),
            storage_key=str(uuid4()),
            file_name="report.pdf",
            file_path="/server-generated",
            file_type="application/pdf",
            file_size=1,
            status="pending",
        )

    def test_cross_conversation_and_partial_submission_links_are_rejected(self):
        """Reject attachments linked to another chat or missing required parts of their new send
        linkage.
        """
        other_id = uuid4()
        with self.engine.begin() as connection:
            connection.execute(insert(Conversation).values(id=other_id, title="Other"))
        for changes in (
            {"conversation_id": other_id},
            {"message_id": None},
            {"submission_id": None},
            {"storage_key": None},
        ):
            document = self.document()
            document.update(changes)
            with self.subTest(changes=changes), self.assertRaises(IntegrityError):
                with self.engine.begin() as connection:
                    connection.execute(insert(Attachment).values(**document))

    def test_live_filename_unique_but_tombstone_releases_name(self):
        """Reject a repeated live filename, then allow it after the original document is marked
        deleted.
        """
        original = self.document()
        with self.engine.begin() as connection:
            connection.execute(insert(Attachment).values(**original))
        with self.assertRaises(IntegrityError):
            with self.engine.begin() as connection:
                connection.execute(insert(Attachment).values(**self.document()))
        with self.engine.begin() as connection:
            connection.execute(
                update(Attachment)
                .where(Attachment.id == original["id"])
                .values(deleted_at=datetime(2026, 1, 1), status="deleted")
            )
            connection.execute(insert(Attachment).values(**self.document()))

    def test_status_and_error_constraints(self):
        """Reject unsupported file states, negative byte sizes and non-object error records."""
        for changes in ({"status": "bogus"}, {"file_size": -1}, {"error": []}):
            document = self.document()
            document.update(changes)
            with self.subTest(changes=changes), self.assertRaises(IntegrityError):
                with self.engine.begin() as connection:
                    connection.execute(insert(Attachment).values(**document))

    def test_downgrade_and_upgrade_preserve_legacy_records(self):
        """Keep an old attachment's disk path through downgrade and restore its ready default on
        upgrade.
        """
        legacy_id = uuid4()
        with self.engine.begin() as connection:
            connection.execute(
                insert(Attachment).values(
                    id=legacy_id,
                    conversation_id=self.conversation_id,
                    file_name="legacy.pdf",
                    file_path="/legacy/file.pdf",
                    file_type="application/pdf",
                    file_size=1,
                )
            )
            config = Config(str(PROJECT_ROOT / "alembic.ini"))
            config.attributes["connection"] = connection
            config.attributes["configure_logger"] = False
            command.downgrade(config, "0003_submissions_runs")
            self.assertEqual(
                connection.scalar(
                    text("SELECT file_path FROM attachments WHERE id = :id"), {"id": legacy_id}
                ),
                "/legacy/file.pdf",
            )
            upgrade_database(connection)
            self.assertEqual(
                connection.scalar(select(Attachment.status).where(Attachment.id == legacy_id)),
                "pending",
            )
