"""Add processing status and send/message links to uploaded file records."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0004_submission_attachments"
down_revision = "0003_submissions_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add attachment status and links to the send and user message that received it.

    Keep old file paths and citation links. Old files become ready and deleted files
    become deleted. Require new links to match the same conversation and user send;
    prevent duplicate live filenames for new uploads without rewriting old duplicates.
    No files are moved or read by this migration.
    """
    for column in (
        sa.Column("submission_id", postgresql.UUID(as_uuid=True)),
        sa.Column("message_id", postgresql.UUID(as_uuid=True)),
        sa.Column("client_file_id", sa.String(128)),
        sa.Column("storage_key", sa.String(512)),
        sa.Column("status", sa.String(24), nullable=False, server_default="ready"),
        sa.Column("error", postgresql.JSONB(none_as_null=True)),
    ):
        op.add_column("attachments", column)
    op.execute("UPDATE attachments SET status = 'deleted' WHERE deleted_at IS NOT NULL")
    op.create_foreign_key(
        "fk_attachments_input_submission",
        "attachments",
        "messages",
        ["message_id", "submission_id", "conversation_id"],
        ["id", "submission_id", "conversation_id"],
    )
    op.create_check_constraint(
        "ck_attachments_status",
        "attachments",
        "status IN ('pending', 'indexing', 'ready', 'failed', 'deleted')",
    )
    op.create_check_constraint("ck_attachments_size", "attachments", "file_size >= 0")
    op.create_check_constraint(
        "ck_attachments_submission_link",
        "attachments",
        "(submission_id IS NULL AND message_id IS NULL AND storage_key IS NULL) OR "
        "(submission_id IS NOT NULL AND message_id IS NOT NULL AND storage_key IS NOT NULL AND client_file_id IS NOT NULL)",
    )
    op.create_check_constraint(
        "ck_attachments_error", "attachments", "error IS NULL OR jsonb_typeof(error) = 'object'"
    )
    op.create_unique_constraint("uq_attachments_storage_key", "attachments", ["storage_key"])
    op.create_unique_constraint(
        "uq_attachments_client_file", "attachments", ["submission_id", "client_file_id"]
    )
    op.create_index("ix_attachments_message_id", "attachments", ["message_id"])
    op.create_index("ix_attachments_status_created", "attachments", ["status", "created_at"])
    op.create_index(
        "uq_attachments_live_name",
        "attachments",
        ["conversation_id", "file_name"],
        unique=True,
        postgresql_where=sa.text("submission_id IS NOT NULL AND deleted_at IS NULL"),
    )


def downgrade() -> None:
    """Remove the new attachment links/status fields while keeping rows and disk files.

    New uploads lose their send/message/client/storage links. Upgrading again does
    not restore that history; recovering it requires the database backup.
    """
    for name in (
        "uq_attachments_live_name",
        "ix_attachments_status_created",
        "ix_attachments_message_id",
    ):
        op.drop_index(name, table_name="attachments")
    op.drop_constraint("fk_attachments_input_submission", "attachments", type_="foreignkey")
    for name in ("uq_attachments_storage_key", "uq_attachments_client_file"):
        op.drop_constraint(name, "attachments", type_="unique")
    for name in (
        "ck_attachments_status",
        "ck_attachments_size",
        "ck_attachments_submission_link",
        "ck_attachments_error",
    ):
        op.drop_constraint(name, "attachments", type_="check")
    for name in ("error", "status", "storage_key", "client_file_id", "message_id", "submission_id"):
        op.drop_column("attachments", name)
