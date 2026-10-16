"""Create the original five tables or register an existing database that matches them.

Revision ID: 0001_existing_schema
Revises: None

The table definitions below describe the original schema and must stay unchanged
when later features add columns. Add those changes in a new revision instead.
"""

from alembic import op
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.util import CommandError
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Connection

revision = "0001_existing_schema"
down_revision = None
branch_labels = None
depends_on = None


def baseline_metadata() -> sa.MetaData:
    """Return SQLAlchemy table definitions for the original five database tables.

    Build a separate MetaData object without creating tables or importing current ORM
    models. Include SQL defaults only; Python defaults such as UUID generation are
    not database defaults. Later model changes must not change this baseline.
    """
    metadata = sa.MetaData()
    sa.Table(
        "conversations",
        metadata,
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("deleted_at", sa.DateTime()),
        sa.Column("summary", sa.Text()),
        sa.Column("message_count", sa.Integer(), nullable=False),
        sa.Column("unsummarized_message_count", sa.Integer(), nullable=False),
    )
    sa.Table(
        "messages",
        metadata,
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("personality", sa.String(50)),
        sa.Column("model_name", sa.String(50)),
        sa.Column("reasoning_mode", sa.String(20)),
        sa.Column("temperature", sa.Float()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("deleted_at", sa.DateTime()),
    )
    sa.Table(
        "attachments",
        metadata,
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("file_name", sa.String(255), nullable=False),
        sa.Column("file_path", sa.String(512), nullable=False),
        sa.Column("file_type", sa.String(255), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("extracted_text", sa.Text()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("deleted_at", sa.DateTime()),
    )
    sa.Table(
        "message_attachments",
        metadata,
        sa.Column(
            "message_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("messages.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "attachment_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("attachments.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    sa.Table(
        "installed_models",
        metadata,
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("model_name", sa.String(), nullable=False, unique=True),
        sa.Column("hf_repo", sa.String(), nullable=False),
        sa.Column("gguf_file", sa.String(), nullable=False),
        sa.Column("local_path", sa.String(), nullable=False),
        sa.Column("quantization", sa.String(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("installed_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("is_default", sa.Boolean(), nullable=False),
    )
    return metadata


def validate_legacy_schema(connection: Connection, metadata: sa.MetaData) -> None:
    """Check that an existing database matches the supplied baseline table definitions.

    connection is the open database connection; metadata is the original schema.
    Compare tables, columns, types, defaults and primary keys. Return None on a match.
    Raise CommandError if anything differs, before recording a baseline revision.
    Do not create tables, rewrite rows or silently mark a mismatched schema valid.
    """
    migration_context = MigrationContext.configure(
        connection,
        opts={"compare_type": True, "compare_server_default": True},
    )
    differences = compare_metadata(migration_context, metadata)

    # Alembic autogenerate does not detect primary-key changes reliably.
    inspector = sa.inspect(connection)
    existing_tables = set(inspector.get_table_names())
    for table in metadata.sorted_tables:
        if table.name not in existing_tables:
            continue
        actual_primary_key = inspector.get_pk_constraint(table.name)["constrained_columns"]
        expected_primary_key = [column.name for column in table.primary_key.columns]
        if actual_primary_key != expected_primary_key:
            differences.append(
                ("primary_key_mismatch", table.name, actual_primary_key, expected_primary_key)
            )

    if differences:
        difference_details = "\n".join(f"  - {difference!r}" for difference in differences)
        raise CommandError(
            "Legacy database does not match baseline 0001_existing_schema. "
            "No baseline was recorded. Back up the database and reconcile its schema; "
            "do not use 'alembic stamp head' to bypass this check.\n"
            f"{difference_details}"
        )


def upgrade() -> None:
    """Create baseline tables if empty, or validate existing tables before adoption.

    When generating offline SQL, emit table creation only. Online, reject partial,
    changed or extra application tables through validate_legacy_schema. Existing rows
    are left intact; Alembic records the revision only after this function succeeds.
    """
    metadata = baseline_metadata()
    connection = op.get_bind()

    if op.get_context().as_sql:
        metadata.create_all(connection, checkfirst=False)
        return

    existing_tables = set(sa.inspect(connection).get_table_names()) - {"alembic_version"}
    if existing_tables:
        validate_legacy_schema(connection, metadata)
        return

    metadata.create_all(connection, checkfirst=False)


def downgrade() -> None:
    """Drop the five baseline tables and all rows they contain.

    This also deletes data from databases originally adopted from the old app.
    Use a backup to recover it; upgrading again recreates tables, not lost rows.
    """
    baseline_metadata().drop_all(op.get_bind(), checkfirst=False)
