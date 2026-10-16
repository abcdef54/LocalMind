"""Add indexes for loading conversations and messages one page at a time.

Revision ID: 0002_history_pagination
Revises: 0001_existing_schema
"""

from alembic import op
import sqlalchemy as sa

revision = "0002_history_pagination"
down_revision = "0001_existing_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Old create_all allowed an explicit NULL; its original creation time is unknown.
    """Fill missing conversation creation times and add timestamp/ID pagination indexes.

    Use migration time only where created_at is NULL, since the original time is
    unknown. Then require conversation created_at and index the column pairs used
    by page queries. Leave existing non-NULL timestamps and message rows unchanged.
    """
    op.execute("UPDATE conversations SET created_at = now() WHERE created_at IS NULL")
    op.alter_column(
        "conversations", "created_at", existing_type=sa.DateTime(timezone=True), nullable=False
    )
    op.create_index("ix_conversations_created_id", "conversations", ["created_at", "id"])
    op.create_index(
        "ix_messages_conversation_created_id", "messages", ["conversation_id", "created_at", "id"]
    )


def downgrade() -> None:
    """Remove the pagination indexes and allow NULL conversation creation times again.

    Do not delete conversations/messages or undo timestamps filled during upgrade.
    """
    op.drop_index("ix_messages_conversation_created_id", table_name="messages")
    op.drop_index("ix_conversations_created_id", table_name="conversations")
    op.alter_column(
        "conversations", "created_at", existing_type=sa.DateTime(timezone=True), nullable=True
    )
