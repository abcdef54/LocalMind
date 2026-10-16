"""Add durable submission/run identities without changing legacy message ownership.

Revision ID: 0003_submissions_runs
Revises: 0002_history_pagination
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0003_submissions_runs"
down_revision = "0002_history_pagination"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "submissions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("client_submission_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("settings_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.String(24), nullable=False, server_default="accepted"),
        sa.Column("error", postgresql.JSONB(none_as_null=True)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "conversation_id", "client_submission_id", name="uq_submissions_client_key"
        ),
        sa.UniqueConstraint("id", "conversation_id", name="uq_submissions_identity"),
        sa.CheckConstraint("fingerprint ~ '^[0-9a-f]{64}$'", name="ck_submissions_fingerprint"),
        sa.CheckConstraint(
            "status IN ('accepted', 'indexing', 'ready', 'failed', 'cancelled')",
            name="ck_submissions_status",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(settings_snapshot) = 'object'", name="ck_submissions_settings"
        ),
        sa.CheckConstraint(
            "error IS NULL OR jsonb_typeof(error) = 'object'", name="ck_submissions_error"
        ),
    )
    op.create_index("ix_submissions_status_created", "submissions", ["status", "created_at"])
    op.add_column("messages", sa.Column("submission_id", postgresql.UUID(as_uuid=True)))
    op.add_column("messages", sa.Column("run_id", postgresql.UUID(as_uuid=True)))
    op.create_unique_constraint("uq_messages_submission_id", "messages", ["submission_id"])
    op.create_unique_constraint("uq_messages_run_id", "messages", ["run_id"])
    op.create_unique_constraint(
        "uq_messages_input_identity", "messages", ["id", "submission_id", "conversation_id"]
    )
    op.create_unique_constraint(
        "uq_messages_result_identity", "messages", ["id", "run_id", "conversation_id"]
    )
    op.create_foreign_key(
        "fk_messages_submission_conversation",
        "messages",
        "submissions",
        ["submission_id", "conversation_id"],
        ["id", "conversation_id"],
    )
    op.create_check_constraint(
        "ck_messages_submission_role", "messages", "submission_id IS NULL OR role = 'user'"
    )
    op.create_check_constraint(
        "ck_messages_run_role", "messages", "run_id IS NULL OR role = 'assistant'"
    )
    op.create_table(
        "runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("submission_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("message_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("result_message_id", postgresql.UUID(as_uuid=True)),
        sa.Column("settings_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.String(24), nullable=False, server_default="pending"),
        sa.Column("error", postgresql.JSONB(none_as_null=True)),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "cancel_requested", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column("lease_owner", sa.String(128)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("submission_id", name="uq_runs_submission_id"),
        sa.UniqueConstraint("message_id", name="uq_runs_message_id"),
        sa.UniqueConstraint("id", "conversation_id", name="uq_runs_identity"),
        sa.ForeignKeyConstraint(
            ["submission_id", "conversation_id"],
            ["submissions.id", "submissions.conversation_id"],
            name="fk_runs_submission_conversation",
        ),
        sa.ForeignKeyConstraint(
            ["message_id", "submission_id", "conversation_id"],
            ["messages.id", "messages.submission_id", "messages.conversation_id"],
            name="fk_runs_input_submission",
        ),
        sa.ForeignKeyConstraint(
            ["result_message_id", "id", "conversation_id"],
            ["messages.id", "messages.run_id", "messages.conversation_id"],
            name="fk_runs_result_identity",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'queued', 'running', 'cancel_requested', 'completed', 'failed', 'cancelled', 'interrupted')",
            name="ck_runs_status",
        ),
        sa.CheckConstraint("attempt >= 0", name="ck_runs_attempt"),
        sa.CheckConstraint("jsonb_typeof(settings_snapshot) = 'object'", name="ck_runs_settings"),
        sa.CheckConstraint("error IS NULL OR jsonb_typeof(error) = 'object'", name="ck_runs_error"),
        sa.CheckConstraint(
            "status != 'completed' OR result_message_id IS NOT NULL",
            name="ck_runs_completed_result",
        ),
    )
    op.create_foreign_key(
        "fk_messages_run_conversation",
        "messages",
        "runs",
        ["run_id", "conversation_id"],
        ["id", "conversation_id"],
    )
    op.create_index("ix_runs_status_created", "runs", ["status", "created_at"])
    op.create_index(
        "uq_runs_active_conversation",
        "runs",
        ["conversation_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('pending', 'queued', 'running', 'cancel_requested')"),
    )


def downgrade() -> None:
    op.drop_constraint("fk_messages_run_conversation", "messages", type_="foreignkey")
    op.drop_table("runs")
    op.drop_constraint("fk_messages_submission_conversation", "messages", type_="foreignkey")
    for name in ("ck_messages_submission_role", "ck_messages_run_role"):
        op.drop_constraint(name, "messages", type_="check")
    for name in (
        "uq_messages_submission_id",
        "uq_messages_run_id",
        "uq_messages_input_identity",
        "uq_messages_result_identity",
    ):
        op.drop_constraint(name, "messages", type_="unique")
    op.drop_column("messages", "run_id")
    op.drop_column("messages", "submission_id")
    op.drop_table("submissions")
