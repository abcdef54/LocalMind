"""Store answer generation metadata and support the v1 model-name length."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0007_generation_metadata"
down_revision = "0006_document_cleanup"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add nullable answer metadata without changing historical answers or invoking AI."""
    op.add_column("messages", sa.Column("generation_metadata", postgresql.JSONB(none_as_null=True)))
    op.create_check_constraint(
        "ck_messages_generation_metadata",
        "messages",
        "generation_metadata IS NULL OR jsonb_typeof(generation_metadata) = 'object'",
    )
    op.alter_column("messages", "model_name", existing_type=sa.String(50), type_=sa.String(255))


def downgrade() -> None:
    """Remove generation metadata; reject shrinking names if existing values are too long."""
    op.alter_column("messages", "model_name", existing_type=sa.String(255), type_=sa.String(50))
    op.drop_constraint("ck_messages_generation_metadata", "messages", type_="check")
    op.drop_column("messages", "generation_metadata")
