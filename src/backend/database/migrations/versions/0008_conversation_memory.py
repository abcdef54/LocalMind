"""Track exactly which messages a saved conversation summary covers."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0008_conversation_memory"
down_revision = "0007_generation_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add summary boundaries and rebuild counters; keep legacy summary text for inspection.

    Legacy summary coverage is unknown. The memory service ignores it until a new
    summary has a trusted boundary; stored messages are used to rebuild safely.
    """
    op.add_column("conversations", sa.Column("summary_through_created_at", sa.DateTime()))
    op.add_column(
        "conversations", sa.Column("summary_through_message_id", postgresql.UUID(as_uuid=True))
    )
    op.add_column(
        "conversations",
        sa.Column("summary_version", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.create_check_constraint(
        "ck_conversations_summary_boundary",
        "conversations",
        "(summary_through_created_at IS NULL) = (summary_through_message_id IS NULL)",
    )
    op.create_check_constraint(
        "ck_conversations_summary_version", "conversations", "summary_version >= 0"
    )
    op.execute(
        """UPDATE conversations SET message_count = (
        SELECT count(*) FROM messages WHERE messages.conversation_id = conversations.id AND messages.deleted_at IS NULL),
        unsummarized_message_count = (SELECT count(*) FROM messages WHERE messages.conversation_id = conversations.id AND messages.deleted_at IS NULL)"""
    )


def downgrade() -> None:
    """Remove boundary/version fields without deleting messages or summary text."""
    op.drop_constraint("ck_conversations_summary_boundary", "conversations", type_="check")
    op.drop_constraint("ck_conversations_summary_version", "conversations", type_="check")
    for name in ("summary_version", "summary_through_message_id", "summary_through_created_at"):
        op.drop_column("conversations", name)
