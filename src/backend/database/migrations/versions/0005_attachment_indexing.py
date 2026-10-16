"""Track durable indexing attempts and reindex existing files into a scoped collection."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0005_attachment_indexing"
down_revision = "0004_submission_attachments"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add indexing reservations and hide old vectors until their files are reindexed.

    Existing undeleted attachments become pending. Their file paths, original text,
    messages and citations stay intact. No file/model/vector operations run here.
    """
    for column in (
        sa.Column("index_generation", postgresql.UUID(as_uuid=True)),
        sa.Column("index_owner", sa.String(128)),
        sa.Column("index_expires_at", sa.DateTime(timezone=True)),
        sa.Column("index_attempt", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
    ):
        op.add_column("attachments", column)
    op.execute("UPDATE attachments SET status = 'pending' WHERE deleted_at IS NULL")
    # Existing sends must wait for the new scoped collection before queued AI can run.
    # Completed/failed/cancelled AI runs retain their historical outcome.
    op.execute("""
        UPDATE submissions SET status = 'accepted'
        WHERE status IN ('accepted', 'indexing', 'ready')
          AND id IN (SELECT submission_id FROM attachments WHERE deleted_at IS NULL)
    """)
    op.execute("""
        UPDATE runs SET status = 'pending'
        WHERE status = 'queued'
          AND submission_id IN (SELECT submission_id FROM attachments WHERE deleted_at IS NULL)
    """)
    op.create_check_constraint("ck_attachments_index_attempt", "attachments", "index_attempt >= 0")
    op.create_check_constraint("ck_attachments_chunk_count", "attachments", "chunk_count >= 0")
    op.create_check_constraint(
        "ck_attachments_index_lease",
        "attachments",
        "(index_owner IS NULL) = (index_expires_at IS NULL)",
    )
    op.create_check_constraint(
        "ck_attachments_indexing_owner",
        "attachments",
        "status != 'indexing' OR (index_owner IS NOT NULL AND index_generation IS NOT NULL)",
    )


def downgrade() -> None:
    """Remove indexing attempt fields; leave document bytes and vector storage untouched.

    Pending/indexing work cannot be resumed by the older schema. Back up before
    downgrade; another upgrade requires reindexing into the new collection again.
    """
    for name in (
        "ck_attachments_index_attempt",
        "ck_attachments_chunk_count",
        "ck_attachments_index_lease",
        "ck_attachments_indexing_owner",
    ):
        op.drop_constraint(name, "attachments", type_="check")
    for name in (
        "chunk_count",
        "index_attempt",
        "index_expires_at",
        "index_owner",
        "index_generation",
    ):
        op.drop_column("attachments", name)
