"""Integration tests against a disposable PostgreSQL database, never the app database.

Set LOCALMIND_MIGRATION_TEST_URL to an asyncpg URL whose database name is exactly
localmind_migration_tests. Each test drops that database's public schema.
"""

import asyncio
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.util import CommandError
import sqlalchemy as sa
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from src.backend.database.config import PROJECT_ROOT
from src.backend.database.migration import upgrade_database
from src.backend.database.models import Base


TEST_DATABASE_URL = os.getenv("LOCALMIND_MIGRATION_TEST_URL")
BASELINE_REVISION = "0001_existing_schema"


@unittest.skipUnless(TEST_DATABASE_URL, "LOCALMIND_MIGRATION_TEST_URL is not configured")
class MigrationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        database_url = make_url(TEST_DATABASE_URL)
        if database_url.database != "localmind_migration_tests":
            raise RuntimeError("Migration tests require database localmind_migration_tests.")
        cls.engine = sa.create_engine(database_url.set(drivername="postgresql+psycopg2"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.engine.dispose()

    def setUp(self) -> None:
        self.reset_database()

    def reset_database(self) -> None:
        with self.engine.begin() as connection:
            connection.execute(sa.text("DROP SCHEMA public CASCADE"))
            connection.execute(sa.text("CREATE SCHEMA public"))

    def migration_config(self, connection: sa.Connection) -> Config:
        config = Config(str(PROJECT_ROOT / "alembic.ini"))
        config.attributes["connection"] = connection
        config.attributes["configure_logger"] = False
        return config

    def assert_at_head(self) -> None:
        with self.engine.connect() as connection:
            migration_context = MigrationContext.configure(connection)
            self.assertEqual(migration_context.get_current_revision(), BASELINE_REVISION)
            self.assertEqual(
                compare_metadata(migration_context, Base.metadata),
                [],
            )

    def assert_not_baselined(self) -> None:
        with self.engine.connect() as connection:
            migration_context = MigrationContext.configure(connection)
            self.assertIsNone(migration_context.get_current_revision())

    def seed_legacy_database(self) -> None:
        """Emulate the old create_all startup, including rows and citation links."""
        with self.engine.begin() as connection:
            Base.metadata.create_all(connection)
            conversation_id = connection.execute(
                Base.metadata.tables["conversations"].insert().values(title="Existing chat")
            ).inserted_primary_key[0]
            message_id = connection.execute(
                Base.metadata.tables["messages"].insert().values(
                    conversation_id=conversation_id,
                    content="Keep this message",
                    role="assistant",
                )
            ).inserted_primary_key[0]
            attachment_id = connection.execute(
                Base.metadata.tables["attachments"].insert().values(
                    conversation_id=conversation_id,
                    file_name="existing.pdf",
                    file_path="/existing/file.pdf",
                    file_type="application/pdf",
                    file_size=123,
                )
            ).inserted_primary_key[0]
            connection.execute(
                Base.metadata.tables["message_attachments"].insert().values(
                    message_id=message_id,
                    attachment_id=attachment_id,
                )
            )
            connection.execute(
                Base.metadata.tables["installed_models"].insert().values(
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
        with self.engine.connect() as connection:
            return {
                table.name: list(
                    connection.execute(sa.select(table).order_by(*table.primary_key.columns))
                )
                for table in Base.metadata.sorted_tables
            }

    def test_empty_database_upgrades_and_matches_orm_schema(self) -> None:
        with self.engine.begin() as connection:
            upgrade_database(connection)
            command.check(self.migration_config(connection))
        self.assert_at_head()

    def test_legacy_adoption_and_repeated_upgrade_preserve_all_rows(self) -> None:
        self.seed_legacy_database()
        original_rows = self.legacy_rows()
        for _ in range(2):
            with self.engine.begin() as connection:
                upgrade_database(connection)
        self.assert_at_head()
        self.assertEqual(self.legacy_rows(), original_rows)

    def test_partial_legacy_schema_is_rejected_without_creating_missing_tables(self) -> None:
        with self.engine.begin() as connection:
            Base.metadata.tables["conversations"].create(connection)
        with self.assertRaisesRegex(CommandError, "does not match baseline"):
            with self.engine.begin() as connection:
                upgrade_database(connection)
        self.assert_not_baselined()
        with self.engine.connect() as connection:
            self.assertEqual(sa.inspect(connection).get_table_names(), ["conversations"])

    def test_schema_drift_is_rejected_without_modifying_legacy_rows(self) -> None:
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
        with self.engine.begin() as connection:
            connection.execute(sa.text("CREATE TABLE other_application (id integer)"))
        with self.assertRaisesRegex(CommandError, "does not match baseline"):
            with self.engine.begin() as connection:
                upgrade_database(connection)
        self.assert_not_baselined()

    def test_downgrade_and_upgrade_on_disposable_database(self) -> None:
        with self.engine.begin() as connection:
            upgrade_database(connection)
            command.downgrade(self.migration_config(connection), "base")
        with self.engine.connect() as connection:
            self.assertEqual(sa.inspect(connection).get_table_names(), ["alembic_version"])
        with self.engine.begin() as connection:
            upgrade_database(connection)
        self.assert_at_head()

    def test_async_cli_works_outside_project_directory(self) -> None:
        environment = {**os.environ, "POSTGRESQL_URL": TEST_DATABASE_URL}
        with tempfile.TemporaryDirectory() as directory:
            for arguments in [("upgrade", "head"), ("check",)]:
                completed = subprocess.run(
                    [sys.executable, "-m", "alembic", "-c", str(PROJECT_ROOT / "alembic.ini"), *arguments],
                    cwd=directory,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assert_at_head()

    def test_async_application_startup_uses_migrations(self) -> None:
        async def initialize_test_database() -> None:
            # Importing this module creates an engine but does not connect to the app DB.
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
