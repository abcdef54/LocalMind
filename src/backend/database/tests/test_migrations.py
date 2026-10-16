"""Check schema creation, adoption and downgrade on a disposable PostgreSQL database.

Require LOCALMIND_MIGRATION_TEST_URL with database name localmind_migration_tests.
Cases drop public schema; never configure this suite with valuable application data.
"""

import asyncio
from datetime import datetime
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from alembic.util import CommandError
import sqlalchemy as sa
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from src.backend.database.config import PROJECT_ROOT
from src.backend.database.migration import upgrade_database
from src.backend.database.models import Base

TEST_DATABASE_URL = os.getenv("LOCALMIND_MIGRATION_TEST_URL")
BASELINE_REVISION = "0001_existing_schema"
MIGRATION_SCRIPTS = ScriptDirectory.from_config(Config(str(PROJECT_ROOT / "alembic.ini")))
LEGACY_METADATA = MIGRATION_SCRIPTS.get_revision(BASELINE_REVISION).module.baseline_metadata()


@unittest.skipUnless(TEST_DATABASE_URL, "LOCALMIND_MIGRATION_TEST_URL is not configured")
class MigrationIntegrationTests(unittest.TestCase):
    """Verify Alembic changes preserve old rows and reject mismatched existing schemas."""

    @classmethod
    def setUpClass(cls) -> None:
        """Open the disposable test database, rejecting any other database name.

        Require localmind_migration_tests because individual cases drop its public schema.
        Create a synchronous engine for fixtures; never use the application's database.
        """
        database_url = make_url(TEST_DATABASE_URL)
        if database_url.database != "localmind_migration_tests":
            raise RuntimeError("Migration tests require database localmind_migration_tests.")
        cls.engine = sa.create_engine(database_url.set(drivername="postgresql+psycopg2"))

    @classmethod
    def tearDownClass(cls) -> None:
        """Dispose the synchronous fixture engine after all cases finish."""
        cls.engine.dispose()

    def setUp(self) -> None:
        """Reset the disposable database schema before each migration scenario."""
        self.reset_database()

    def reset_database(self) -> None:
        """Drop and recreate public schema in the guarded test database; delete all its test
        rows.
        """
        with self.engine.begin() as connection:
            connection.execute(sa.text("DROP SCHEMA public CASCADE"))
            connection.execute(sa.text("CREATE SCHEMA public"))

    def migration_config(self, connection: sa.Connection) -> Config:
        """Return Alembic configuration using the supplied fixture connection without resetting
        loggers.
        """
        config = Config(str(PROJECT_ROOT / "alembic.ini"))
        config.attributes["connection"] = connection
        config.attributes["configure_logger"] = False
        return config

    def assert_at_head(self) -> None:
        """Assert the current revision is latest and inspected SQL schema matches ORM
        definitions.
        """
        with self.engine.connect() as connection:
            migration_context = MigrationContext.configure(connection)
            self.assertEqual(
                migration_context.get_current_revision(), MIGRATION_SCRIPTS.get_current_head()
            )
            self.assertEqual(
                compare_metadata(migration_context, Base.metadata),
                [],
            )

    def assert_not_baselined(self) -> None:
        """Assert Alembic has not recorded any revision after an invalid adoption attempt."""
        with self.engine.connect() as connection:
            migration_context = MigrationContext.configure(connection)
            self.assertIsNone(migration_context.get_current_revision())

    def seed_legacy_database(self) -> None:
        """Create original tables and sample chat/message/file/citation/model rows without
        Alembic.
        """
        with self.engine.begin() as connection:
            LEGACY_METADATA.create_all(connection)
            conversation_id = connection.execute(
                LEGACY_METADATA.tables["conversations"]
                .insert()
                .values(
                    id=uuid4(), title="Existing chat", message_count=1, unsummarized_message_count=1
                )
            ).inserted_primary_key[0]
            message_id = connection.execute(
                LEGACY_METADATA.tables["messages"]
                .insert()
                .values(
                    id=uuid4(),
                    created_at=datetime(2026, 1, 1),
                    conversation_id=conversation_id,
                    content="Keep this message",
                    role="assistant",
                )
            ).inserted_primary_key[0]
            attachment_id = connection.execute(
                LEGACY_METADATA.tables["attachments"]
                .insert()
                .values(
                    id=uuid4(),
                    created_at=datetime(2026, 1, 1),
                    conversation_id=conversation_id,
                    file_name="existing.pdf",
                    file_path="/existing/file.pdf",
                    file_type="application/pdf",
                    file_size=123,
                )
            ).inserted_primary_key[0]
            connection.execute(
                LEGACY_METADATA.tables["message_attachments"]
                .insert()
                .values(
                    message_id=message_id,
                    attachment_id=attachment_id,
                )
            )
            connection.execute(
                LEGACY_METADATA.tables["installed_models"]
                .insert()
                .values(
                    id=uuid4(),
                    display_name="Existing model",
                    model_name="existing-model",
                    hf_repo="local/existing",
                    gguf_file="existing.gguf",
                    local_path="/models/existing.gguf",
                    quantization="Q4_K_M",
                    size_bytes=456,
                    is_default=True,
                )
            )

    def legacy_rows(self) -> dict[str, list[tuple]]:
        """Return sorted original-column rows for every baseline table so migrations can be
        compared.
        """
        with self.engine.connect() as connection:
            return {
                table.name: list(
                    connection.execute(sa.select(table).order_by(*table.primary_key.columns))
                )
                for table in LEGACY_METADATA.sorted_tables
            }

    def test_empty_database_upgrades_and_matches_orm_schema(self) -> None:
        """Create the schema from empty and verify Alembic detects no remaining ORM differences."""
        with self.engine.begin() as connection:
            upgrade_database(connection)
            command.check(self.migration_config(connection))
        self.assert_at_head()

    def test_legacy_adoption_and_repeated_upgrade_preserve_all_rows(self) -> None:
        """Adopt old tables and repeat upgrade without rewriting any original rows or citation
        links.
        """
        self.seed_legacy_database()
        original_rows = self.legacy_rows()
        for _ in range(2):
            with self.engine.begin() as connection:
                upgrade_database(connection)
        self.assert_at_head()
        self.assertEqual(self.legacy_rows(), original_rows)

    def test_versioned_baseline_upgrades_without_rewriting_legacy_data(self) -> None:
        """Apply newer revisions to an already versioned baseline without changing original row
        values.
        """
        with self.engine.begin() as connection:
            command.upgrade(self.migration_config(connection), BASELINE_REVISION)
        self.seed_legacy_database()
        original_rows = self.legacy_rows()
        with self.engine.begin() as connection:
            upgrade_database(connection)
        self.assert_at_head()
        self.assertEqual(self.legacy_rows(), original_rows)
        with self.engine.connect() as connection:
            legacy_message = (
                connection.execute(sa.select(Base.metadata.tables["messages"])).mappings().one()
            )
            self.assertIsNone(legacy_message["submission_id"])
            self.assertIsNone(legacy_message["run_id"])

    def test_null_legacy_conversation_time_is_backfilled_before_cursor_indexing(self) -> None:
        """Fill an unknown old creation time before requiring timestamps for cursor pagination."""
        self.seed_legacy_database()
        with self.engine.begin() as connection:
            connection.execute(sa.text("UPDATE conversations SET created_at = NULL"))
            upgrade_database(connection)
        with self.engine.connect() as connection:
            self.assertIsNotNone(connection.scalar(sa.text("SELECT created_at FROM conversations")))
        self.assert_at_head()

    def test_downgrade_to_baseline_preserves_preexisting_rows(self) -> None:
        """Remove newer schema additions while keeping original records, then upgrade again."""
        self.seed_legacy_database()
        original_rows = self.legacy_rows()
        with self.engine.begin() as connection:
            upgrade_database(connection)
            command.downgrade(self.migration_config(connection), BASELINE_REVISION)
        self.assertEqual(self.legacy_rows(), original_rows)
        with self.engine.begin() as connection:
            upgrade_database(connection)
        self.assert_at_head()

    def test_partial_legacy_schema_is_rejected_without_creating_missing_tables(self) -> None:
        """Refuse adoption of an incomplete old schema instead of creating missing tables
        silently.
        """
        with self.engine.begin() as connection:
            LEGACY_METADATA.tables["conversations"].create(connection)
        with self.assertRaisesRegex(CommandError, "does not match baseline"):
            with self.engine.begin() as connection:
                upgrade_database(connection)
        self.assert_not_baselined()
        with self.engine.connect() as connection:
            self.assertEqual(sa.inspect(connection).get_table_names(), ["conversations"])

    def test_schema_drift_is_rejected_without_modifying_legacy_rows(self) -> None:
        """Reject a changed old schema without recording a revision or rewriting its rows."""
        drift_statements = [
            "ALTER TABLE conversations ALTER COLUMN title TYPE varchar(100)",
            "ALTER TABLE conversations ALTER COLUMN title DROP NOT NULL",
            "ALTER TABLE conversations ALTER COLUMN message_count SET DEFAULT 0",
            "DROP INDEX ix_messages_conversation_id",
            "ALTER TABLE messages DROP CONSTRAINT messages_conversation_id_fkey",
            "ALTER TABLE installed_models DROP CONSTRAINT installed_models_model_name_key",
            "ALTER TABLE message_attachments DROP CONSTRAINT message_attachments_pkey",
            "ALTER TABLE attachments ADD COLUMN unexpected_column text",
        ]
        for statement in drift_statements:
            with self.subTest(schema_change=statement):
                self.reset_database()
                self.seed_legacy_database()
                original_rows = self.legacy_rows()
                with self.engine.begin() as connection:
                    connection.execute(sa.text(statement))
                with self.assertRaisesRegex(CommandError, "does not match baseline"):
                    with self.engine.begin() as connection:
                        upgrade_database(connection)
                self.assert_not_baselined()
                self.assertEqual(self.legacy_rows(), original_rows)

    def test_unrelated_table_is_not_silently_adopted(self) -> None:
        """Reject an existing database containing an unexpected table instead of declaring it
        compatible.
        """
        with self.engine.begin() as connection:
            connection.execute(sa.text("CREATE TABLE other_application (id integer)"))
        with self.assertRaisesRegex(CommandError, "does not match baseline"):
            with self.engine.begin() as connection:
                upgrade_database(connection)
        self.assert_not_baselined()

    def test_downgrade_and_upgrade_on_disposable_database(self) -> None:
        """Drop all application tables through downgrade and recreate them through upgrade."""
        with self.engine.begin() as connection:
            upgrade_database(connection)
            command.downgrade(self.migration_config(connection), "base")
        with self.engine.connect() as connection:
            self.assertEqual(sa.inspect(connection).get_table_names(), ["alembic_version"])
        with self.engine.begin() as connection:
            upgrade_database(connection)
        self.assert_at_head()

    def test_async_cli_works_outside_project_directory(self) -> None:
        """Run Alembic upgrade/check from another working directory using the explicit config
        path.
        """
        environment = {**os.environ, "POSTGRESQL_URL": TEST_DATABASE_URL}
        with tempfile.TemporaryDirectory() as directory:
            for arguments in [("upgrade", "head"), ("check",)]:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "alembic",
                        "-c",
                        str(PROJECT_ROOT / "alembic.ini"),
                        *arguments,
                    ],
                    cwd=directory,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assert_at_head()

    def test_async_application_startup_uses_migrations(self) -> None:
        """Call startup migration twice against a test engine and verify it reaches the latest
        revision.
        """

        async def initialize_test_database() -> None:
            # Importing this module creates an engine but does not connect to the app DB.
            """Temporarily replace the startup engine with the disposable engine, upgrade twice,
            then dispose it.
            """
            from src.backend.database import session

            test_engine = create_async_engine(TEST_DATABASE_URL)
            try:
                with patch.object(session, "engine", test_engine):
                    await session.upgrade_database_schema()
                    await session.upgrade_database_schema()
            finally:
                await test_engine.dispose()

        with patch.dict(os.environ, {"POSTGRESQL_URL": TEST_DATABASE_URL}):
            asyncio.run(initialize_test_database())
        self.assert_at_head()

    def test_offline_sql_contains_frozen_schema(self) -> None:
        """Generate creation SQL for every table without connecting or recording a DB revision."""
        environment = {**os.environ, "POSTGRESQL_URL": TEST_DATABASE_URL}
        completed = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head", "--sql"],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for table_name in Base.metadata.tables:
            self.assertIn(f"CREATE TABLE {table_name}", completed.stdout)
        self.assert_not_baselined()


if __name__ == "__main__":
    unittest.main()
