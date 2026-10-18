"""Persist model download tasks and installation readiness without changing old registry data."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0010_model_downloads"
down_revision = "0009_run_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add resumable task records and file manifests; existing models await startup reconciliation."""
    op.add_column(
        "installed_models",
        sa.Column("status", sa.String(20), nullable=False, server_default="configuring"),
    )
    op.add_column("installed_models", sa.Column("source_revision", sa.String(128), nullable=True))
    op.add_column(
        "installed_models",
        sa.Column(
            "files", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")
        ),
    )
    op.create_check_constraint(
        "ck_installed_models_status",
        "installed_models",
        "status IN ('ready', 'missing', 'deleting', 'configuring')",
    )
    op.create_table(
        "model_downloads",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("repo_id", sa.String(200), nullable=False),
        sa.Column("gguf_filename", sa.String(512), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="queued"),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("downloaded_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("total_bytes", sa.BigInteger(), nullable=True),
        sa.Column("source_revision", sa.String(128), nullable=True),
        sa.Column(
            "model_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("installed_models.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("error", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.clock_timestamp(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.clock_timestamp(),
        ),
        sa.UniqueConstraint("repo_id", "gguf_filename", name="uq_model_download_source"),
        sa.CheckConstraint(
            "status IN ('queued', 'downloading', 'registering', 'completed', 'failed', 'interrupted')",
            name="ck_model_download_status",
        ),
        sa.CheckConstraint(
            "attempt >= 0 AND downloaded_bytes >= 0 AND (total_bytes IS NULL OR total_bytes >= downloaded_bytes)",
            name="ck_model_download_progress",
        ),
    )
    op.create_index("ix_model_download_status_created", "model_downloads", ["status", "created_at"])


def downgrade() -> None:
    """Remove task/status metadata while preserving installed model registry rows and files."""
    op.drop_table("model_downloads")
    op.drop_constraint("ck_installed_models_status", "installed_models")
    op.drop_column("installed_models", "files")
    op.drop_column("installed_models", "source_revision")
    op.drop_column("installed_models", "status")
