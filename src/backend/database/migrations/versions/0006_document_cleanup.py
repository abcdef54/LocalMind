"""Persist document cleanup progress without removing history or file bytes."""

from alembic import op
import sqlalchemy as sa

revision = "0006_document_cleanup"
down_revision = "0005_attachment_indexing"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add retry timestamps and safe errors; documents already marked as deleted await cleanup."""
    op.add_column("attachments", sa.Column("cleanup_completed_at", sa.DateTime(timezone=True)))
    op.add_column("attachments", sa.Column("cleanup_attempted_at", sa.DateTime(timezone=True)))
    op.add_column("attachments", sa.Column("cleanup_error", sa.String(64)))
    op.create_index(
        "ix_attachments_cleanup_pending",
        "attachments",
        ["deleted_at", "cleanup_attempted_at"],
        postgresql_where=sa.text("deleted_at IS NOT NULL AND cleanup_completed_at IS NULL"),
    )


def downgrade() -> None:
    """Drop cleanup bookkeeping; preserve documents and deletion timestamps."""
    op.drop_index("ix_attachments_cleanup_pending", table_name="attachments")
    for name in ("cleanup_error", "cleanup_attempted_at", "cleanup_completed_at"):
        op.drop_column("attachments", name)
