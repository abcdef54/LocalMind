"""Check indexing migration readiness gates and attempt constraints on disposable PostgreSQL."""

import os
import unittest
from uuid import uuid4

from alembic import command
from alembic.config import Config
import sqlalchemy as sa
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError

from src.backend.database.config import PROJECT_ROOT
from src.backend.database.migration import upgrade_database
from src.backend.database.models import Attachment, Conversation, Message, Run, Submission

TEST_URL = os.getenv("LOCALMIND_MIGRATION_TEST_URL")


@unittest.skipUnless(TEST_URL, "LOCALMIND_MIGRATION_TEST_URL is not configured")
class IndexingSchemaTests(unittest.TestCase):
    """Never connect to application data; each case drops the guarded test DB schema."""

    @classmethod
    def setUpClass(cls):
        """Open only localmind_migration_tests, rejecting any other database name."""
        url = make_url(TEST_URL)
        if url.database != "localmind_migration_tests":
            raise RuntimeError("Tests require disposable database localmind_migration_tests")
        cls.engine = sa.create_engine(url.set(drivername="postgresql+psycopg2"))

    @classmethod
    def tearDownClass(cls):
        """Dispose the fixture connection pool after schema checks finish."""
        cls.engine.dispose()

    def setUp(self):
        """Drop the disposable schema and prepare Alembic's explicitly supplied connection."""
        with self.engine.begin() as connection:
            connection.execute(sa.text("DROP SCHEMA public CASCADE"))
            connection.execute(sa.text("CREATE SCHEMA public"))
        self.config = Config(str(PROJECT_ROOT / "alembic.ini"))
        self.config.attributes["configure_logger"] = False

    def test_upgrade_requeues_old_documents_and_gates_existing_queued_reply(self):
        """Preserve source paths/messages but make old ready files wait for scoped reindexing."""
        conversation_id, submission_id, message_id, attachment_id, run_id = [
            uuid4() for _ in range(5)
        ]
        with self.engine.begin() as connection:
            self.config.attributes["connection"] = connection
            command.upgrade(self.config, "0004_submission_attachments")
            connection.execute(
                sa.insert(Conversation).values(id=conversation_id, title="Old ready send")
            )
            connection.execute(
                sa.insert(Submission).values(
                    id=submission_id,
                    conversation_id=conversation_id,
                    client_submission_id=uuid4(),
                    fingerprint="a" * 64,
                    settings_snapshot={},
                    status="ready",
                )
            )
            connection.execute(
                sa.insert(Message).values(
                    id=message_id,
                    conversation_id=conversation_id,
                    submission_id=submission_id,
                    role="user",
                    content="Keep this message",
                )
            )
            connection.execute(
                sa.insert(Run).values(
                    id=run_id,
                    conversation_id=conversation_id,
                    submission_id=submission_id,
                    message_id=message_id,
                    status="queued",
                    settings_snapshot={},
                )
            )
            # Use frozen old columns: new ORM defaults must not insert fields absent at revision 0004.
            connection.execute(
                sa.text("""INSERT INTO attachments
                (id, conversation_id, submission_id, message_id, client_file_id, storage_key,
                 file_name, file_path, file_type, file_size, created_at, status)
                VALUES (:id, :conversation, :submission, :message, 'local-1', 'blobs/old',
                        'old.txt', '/unchanged/source', 'text/plain', 1, now(), 'ready')"""),
                {
                    "id": attachment_id,
                    "conversation": conversation_id,
                    "submission": submission_id,
                    "message": message_id,
                },
            )
            upgrade_database(connection)
            self.assertEqual(connection.scalar(sa.select(Attachment.status)), "pending")
            self.assertEqual(connection.scalar(sa.select(Submission.status)), "accepted")
            self.assertEqual(connection.scalar(sa.select(Run.status)), "pending")
            self.assertEqual(
                connection.scalar(sa.select(Attachment.file_path)), "/unchanged/source"
            )
            self.assertEqual(connection.scalar(sa.select(Message.content)), "Keep this message")

    def test_indexing_requires_owner_generation_and_valid_counts(self):
        """Reject incomplete reservations and invalid attempt/chunk counts at the DB layer."""
        with self.engine.begin() as connection:
            upgrade_database(connection)
            conversation_id = connection.execute(
                sa.insert(Conversation).values(title="Index constraints")
            ).inserted_primary_key[0]
            values = dict(
                conversation_id=conversation_id,
                file_name="one.txt",
                file_path="/not-read",
                file_type="text/plain",
                file_size=1,
            )
            for changes in (
                {"index_attempt": -1},
                {"chunk_count": -1},
                {"status": "indexing"},
                {"index_owner": "missing-expiry"},
            ):
                with self.subTest(changes=changes), self.assertRaises(IntegrityError):
                    with connection.begin_nested():
                        connection.execute(sa.insert(Attachment).values(**values, **changes))
