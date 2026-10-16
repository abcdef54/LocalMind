"""Define PostgreSQL tables for conversations, messages, documents, sends and AI jobs."""

from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    BigInteger,
    Float,
    Boolean,
    Table,
    Index,
    UniqueConstraint,
    CheckConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, DeclarativeBase
from sqlalchemy.sql import func

import uuid
import datetime

ACTIVE_RUN_STATUSES = ("pending", "queued", "running", "cancel_requested")


class Base(DeclarativeBase):
    """Collect ORM table definitions used by sessions and Alembic schema comparisons."""

    pass


class Conversation(Base):
    """Store one chat's title, timestamps and saved conversation summary.

    messages and attachments link its history/documents. deleted_at hides the chat
    without immediately deleting rows or disk files. Counters support later summary
    updates; this model does not maintain them automatically on every new message.
    """

    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conversations_created_id", "created_at", "id"),)

    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    deleted_at = Column(DateTime, nullable=True)

    summary = Column(Text, nullable=True)
    message_count = Column(Integer, nullable=False, default=0)
    unsummarized_message_count = Column(Integer, nullable=False, default=0)

    messages = relationship(
        "Message", back_populates="conversation", passive_deletes=True, cascade="all, delete-orphan"
    )
    attachments = relationship(
        "Attachment",
        back_populates="conversation",
        passive_deletes=True,
        cascade="all, delete-orphan",
    )


message_attachments = Table(
    "message_attachments",
    Base.metadata,
    Column(
        "message_id",
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "attachment_id",
        UUID(as_uuid=True),
        ForeignKey("attachments.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class Message(Base):
    """Store one user or assistant message in a conversation.

    New user messages link to a submission; new assistant messages link to a run.
    Old messages can leave both links empty. attachments lists files sent with a
    user message, while citations lists documents referenced by an answer. Message
    creation times keep the original naive-UTC database format for compatibility.
    """

    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_conversation_created_id", "conversation_id", "created_at", "id"),
        UniqueConstraint("submission_id", name="uq_messages_submission_id"),
        UniqueConstraint("run_id", name="uq_messages_run_id"),
        UniqueConstraint(
            "id", "submission_id", "conversation_id", name="uq_messages_input_identity"
        ),
        UniqueConstraint("id", "run_id", "conversation_id", name="uq_messages_result_identity"),
        ForeignKeyConstraint(
            ["submission_id", "conversation_id"],
            ["submissions.id", "submissions.conversation_id"],
            name="fk_messages_submission_conversation",
        ),
        ForeignKeyConstraint(
            ["run_id", "conversation_id"],
            ["runs.id", "runs.conversation_id"],
            name="fk_messages_run_conversation",
            use_alter=True,
        ),
        CheckConstraint(
            "submission_id IS NULL OR role = 'user'", name="ck_messages_submission_role"
        ),
        CheckConstraint("run_id IS NULL OR role = 'assistant'", name="ck_messages_run_role"),
    )

    id = Column(UUID(as_uuid=True), nullable=False, primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content = Column(Text, nullable=False)
    role = Column(String(20), nullable=False)
    personality = Column(String(50), nullable=True)
    model_name = Column(String(50), nullable=True)
    reasoning_mode = Column(String(20), nullable=True)
    temperature = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)
    submission_id = Column(UUID(as_uuid=True), nullable=True)
    run_id = Column(UUID(as_uuid=True), nullable=True)

    citations = relationship("Attachment", secondary=message_attachments, lazy="selectin")
    attachments = relationship(
        "Attachment",
        foreign_keys="Attachment.message_id",
        lazy="selectin",
        viewonly=True,
        primaryjoin="Message.id == Attachment.message_id",
    )
    conversation = relationship("Conversation", back_populates="messages", passive_deletes=True)


class Attachment(Base):
    """Store a document's original filename, disk location and processing state.

    New uploads link to their user message and submission in the same conversation.
    Their storage_key is generated by the server; file_size counts actual bytes.
    Old documents keep their paths/citation links and have no new submission links.
    Deleted documents keep metadata for history; this row does not delete disk files.
    """

    __tablename__ = "attachments"
    __table_args__ = (
        ForeignKeyConstraint(
            ["message_id", "submission_id", "conversation_id"],
            ["messages.id", "messages.submission_id", "messages.conversation_id"],
            name="fk_attachments_input_submission",
        ),
        CheckConstraint(
            "status IN ('pending', 'indexing', 'ready', 'failed', 'deleted')",
            name="ck_attachments_status",
        ),
        CheckConstraint("file_size >= 0", name="ck_attachments_size"),
        CheckConstraint(
            "(submission_id IS NULL AND message_id IS NULL AND storage_key IS NULL) OR "
            "(submission_id IS NOT NULL AND message_id IS NOT NULL AND storage_key IS NOT NULL AND client_file_id IS NOT NULL)",
            name="ck_attachments_submission_link",
        ),
        CheckConstraint(
            "error IS NULL OR jsonb_typeof(error) = 'object'", name="ck_attachments_error"
        ),
        UniqueConstraint("storage_key", name="uq_attachments_storage_key"),
        UniqueConstraint("submission_id", "client_file_id", name="uq_attachments_client_file"),
        Index("ix_attachments_message_id", "message_id"),
        Index("ix_attachments_status_created", "status", "created_at"),
        Index(
            "uq_attachments_live_name",
            "conversation_id",
            "file_name",
            unique=True,
            postgresql_where=text("submission_id IS NOT NULL AND deleted_at IS NULL"),
        ),
    )

    id = Column(UUID(as_uuid=True), nullable=False, primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_type = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    submission_id = Column(UUID(as_uuid=True), nullable=True)
    message_id = Column(UUID(as_uuid=True), nullable=True)
    client_file_id = Column(String(128), nullable=True)
    storage_key = Column(String(512), nullable=True)
    status = Column(String(24), nullable=False, server_default="ready")
    error = Column(JSONB(none_as_null=True), nullable=True)
    extracted_text = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)

    conversation = relationship("Conversation", back_populates="attachments", passive_deletes=True)


class InstalledModel(Base):
    """Store an installed GGUF model's registry name, source and local file location.

    model_name is sent to inference; display_name is shown in the UI. This record
    describes a model installation but does not load or run that model itself.
    """

    __tablename__ = "installed_models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Display in the UI
    display_name = Column(String, nullable=False)
    # Name sent to llama-swap/OpenAI API
    model_name = Column(String, nullable=False, unique=True)
    hf_repo = Column(String, nullable=False)
    gguf_file = Column(String, nullable=False)
    local_path = Column(String, nullable=False)
    quantization = Column(String, nullable=False)
    size_bytes = Column(BigInteger, nullable=False)
    installed_at = Column(DateTime, server_default=func.now())
    is_default = Column(Boolean, nullable=False, default=False)


class Submission(Base):
    """Store one press of Send, including its saved settings and processing state.

    conversation_id and client_submission_id identify the send for retries. fingerprint
    compares text/settings/file metadata, not file bytes. File sends wait for indexing;
    text-only sends can become ready immediately. Constraints prevent duplicate keys
    within a conversation; the service, not this model, performs retry handling.
    """

    __tablename__ = "submissions"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id", "client_submission_id", name="uq_submissions_client_key"
        ),
        UniqueConstraint("id", "conversation_id", name="uq_submissions_identity"),
        CheckConstraint("fingerprint ~ '^[0-9a-f]{64}$'", name="ck_submissions_fingerprint"),
        CheckConstraint(
            "status IN ('accepted', 'indexing', 'ready', 'failed', 'cancelled')",
            name="ck_submissions_status",
        ),
        CheckConstraint(
            "jsonb_typeof(settings_snapshot) = 'object'", name="ck_submissions_settings"
        ),
        CheckConstraint(
            "error IS NULL OR jsonb_typeof(error) = 'object'", name="ck_submissions_error"
        ),
        Index("ix_submissions_status_created", "status", "created_at"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    client_submission_id = Column(UUID(as_uuid=True), nullable=False)
    fingerprint = Column(String(64), nullable=False)
    settings_snapshot = Column(JSONB, nullable=False)
    status = Column(String(24), nullable=False, server_default="accepted")
    error = Column(JSONB(none_as_null=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=lambda: datetime.datetime.now(datetime.timezone.utc),
    )


class Run(Base):
    """Store the AI job for one submission and link its input and final answer.

    status tracks waiting/running/finished work. lease_owner and lease_expires_at
    record which worker may save its result and until when; attempt distinguishes
    executions. At most one active run is allowed per conversation. An interrupted
    job may keep its owner/slot until model stoppage is confirmed separately.
    """

    __tablename__ = "runs"
    __table_args__ = (
        UniqueConstraint("submission_id", name="uq_runs_submission_id"),
        UniqueConstraint("message_id", name="uq_runs_message_id"),
        UniqueConstraint("id", "conversation_id", name="uq_runs_identity"),
        ForeignKeyConstraint(
            ["submission_id", "conversation_id"],
            ["submissions.id", "submissions.conversation_id"],
            name="fk_runs_submission_conversation",
        ),
        ForeignKeyConstraint(
            ["message_id", "submission_id", "conversation_id"],
            ["messages.id", "messages.submission_id", "messages.conversation_id"],
            name="fk_runs_input_submission",
        ),
        ForeignKeyConstraint(
            ["result_message_id", "id", "conversation_id"],
            ["messages.id", "messages.run_id", "messages.conversation_id"],
            name="fk_runs_result_identity",
        ),
        CheckConstraint(
            "status IN ('pending', 'queued', 'running', 'cancel_requested', 'completed', 'failed', 'cancelled', 'interrupted')",
            name="ck_runs_status",
        ),
        CheckConstraint("attempt >= 0", name="ck_runs_attempt"),
        CheckConstraint("jsonb_typeof(settings_snapshot) = 'object'", name="ck_runs_settings"),
        CheckConstraint("error IS NULL OR jsonb_typeof(error) = 'object'", name="ck_runs_error"),
        CheckConstraint(
            "status != 'completed' OR result_message_id IS NOT NULL",
            name="ck_runs_completed_result",
        ),
        Index("ix_runs_status_created", "status", "created_at"),
        Index(
            "uq_runs_active_conversation",
            "conversation_id",
            unique=True,
            postgresql_where=text("status IN ('pending', 'queued', 'running', 'cancel_requested')"),
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    submission_id = Column(UUID(as_uuid=True), nullable=False)
    message_id = Column(UUID(as_uuid=True), nullable=False)
    result_message_id = Column(UUID(as_uuid=True), nullable=True)
    settings_snapshot = Column(JSONB, nullable=False)
    status = Column(String(24), nullable=False, server_default="pending")
    error = Column(JSONB(none_as_null=True), nullable=True)
    attempt = Column(Integer, nullable=False, server_default=text("0"))
    cancel_requested = Column(Boolean, nullable=False, server_default=text("false"))
    lease_owner = Column(String(128), nullable=True)
    lease_expires_at = Column(DateTime(timezone=True), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=lambda: datetime.datetime.now(datetime.timezone.utc),
    )
