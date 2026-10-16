"""Index bounded history queries and make conversation cursor timestamps non-null.

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
    op.execute("UPDATE conversations SET created_at = now() WHERE created_at IS NULL")
    op.alter_column(
        "conversations", "created_at", existing_type=sa.DateTime(timezone=True), nullable=False
    )
    op.create_index("ix_conversations_created_id", "conversations", ["created_at", "id"])
    op.create_index(
        "ix_messages_conversation_created_id", "messages", ["conversation_id", "created_at", "id"]
    )


def downgrade() -> None:
    op.drop_index("ix_messages_conversation_created_id", table_name="messages")
    op.drop_index("ix_conversations_created_id", table_name="conversations")
    op.alter_column(
        "conversations", "created_at", existing_type=sa.DateTime(timezone=True), nullable=True
    )
