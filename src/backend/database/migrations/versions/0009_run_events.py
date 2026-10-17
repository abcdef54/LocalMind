"""Add ordered run event history and an atomic visualization snapshot."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0009_run_events"
down_revision = "0008_conversation_memory"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Initialize old runs with empty event history; preserve status, answers and leases."""
    op.add_column(
        "runs",
        sa.Column("event_sequence", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        "runs",
        sa.Column(
            "visualization",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.create_table(
        "run_events",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("sequence", sa.BigInteger(), primary_key=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("type", sa.String(64), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.clock_timestamp(),
        ),
        sa.ForeignKeyConstraint(
            ["run_id", "conversation_id"],
            ["runs.id", "runs.conversation_id"],
            ondelete="CASCADE",
            name="fk_run_events_conversation",
        ),
        sa.CheckConstraint("sequence > 0", name="ck_run_events_sequence"),
        sa.CheckConstraint("attempt >= 0", name="ck_run_events_attempt"),
        sa.CheckConstraint("jsonb_typeof(payload) = 'object'", name="ck_run_events_payload"),
    )


def downgrade() -> None:
    """Remove events/projection without deleting runs, messages or uploaded documents."""
    op.drop_table("run_events")
    op.drop_column("runs", "visualization")
    op.drop_column("runs", "event_sequence")
