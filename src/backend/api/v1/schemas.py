"""Validate v1 request fields and describe the JSON returned to the frontend."""

from datetime import datetime, timezone
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints, field_validator

DEFAULT_RANKING_GROUP_SIZE = 3
MAX_RANKING_GROUP_SIZE = 7

ModelName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
ReasoningMode = Literal["low", "medium", "high", "extra"]
Personality = Literal["general", "coder", "researcher", "genz"]


class ChatSettings(BaseModel):
    """Validate the model settings saved for one Send request.

    Require a model name; check supported personality/reasoning values, temperature
    0..2, boolean tools_enabled and ranking_group_size 2..7 (default 3). Group size
    limits candidates compared together, not how many candidates reasoning generates.
    This schema does not verify model installation or change the legacy AI endpoint.
    """

    model_config = ConfigDict(extra="forbid")

    model_name: ModelName
    personality: Personality = "general"
    reasoning_mode: ReasoningMode = "low"
    temperature: Annotated[float, Field(strict=True, ge=0, le=2, allow_inf_nan=False)] = 0.0
    tools_enabled: StrictBool = True
    ranking_group_size: Annotated[
        int,
        Field(
            strict=True,
            ge=2,
            le=MAX_RANKING_GROUP_SIZE,
            description="Maximum candidates per ranking group, independent of reasoning width.",
        ),
    ] = DEFAULT_RANKING_GROUP_SIZE


class ConversationInput(BaseModel):
    """Accept a trimmed, nonempty conversation title of at most 255 characters.

    Used for both creation and rename. Reject unknown request fields.
    """

    model_config = ConfigDict(extra="forbid")

    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]


class ConversationResponse(BaseModel):
    """Return a conversation's UUID, title and creation timestamp to the frontend."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    created_at: datetime


class CitationResponse(BaseModel):
    """Return metadata for a document referenced by an assistant answer.

    file_size is in bytes. deleted=True means the document was removed but its
    historical reference remains. Never include a filesystem path or document content.
    """

    id: UUID
    file_name: str
    file_type: str
    file_size: int
    deleted: bool


class DocumentResponse(BaseModel):
    """Return metadata for a file sent by the user, without its path or contents.

    client_file_id matches the file shown in the composer and may be None for old
    records. file_size is in bytes. status reports processing; deleted reports
    whether the file was removed while keeping its reference in message history.
    error contains a safe processing error when indexing fails. chunk_count is
    the number of text chunks stored by the successful indexing attempt.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    client_file_id: str | None
    file_name: str
    file_type: str
    file_size: int
    status: str
    deleted: bool = False
    error: dict | None = None
    chunk_count: int = 0


class MessageResponse(BaseModel):
    """Return one chat message with settings, timestamp and file references.

    attachments are files sent by the user; citations are documents referenced in
    an assistant answer. submission_id/run_id link new messages to their send/job;
    old messages may have neither. generation_metadata contains compact tournament
    counts/timing for new answers, or None for old messages. Created times use UTC.
    """

    id: UUID
    conversation_id: UUID
    content: str
    role: str
    personality: str | None
    model_name: str | None
    reasoning_mode: str | None
    temperature: float | None
    created_at: datetime
    submission_id: UUID | None = None
    run_id: UUID | None = None
    generation_metadata: dict | None = None
    attachments: list[DocumentResponse] = Field(default_factory=list)
    citations: list[CitationResponse] = Field(default_factory=list)

    @field_validator("created_at")
    @classmethod
    def serialize_legacy_utc(cls, timestamp: datetime) -> datetime:
        # Legacy message timestamps are naive UTC; preserve storage compatibility.
        """Return a message creation timestamp as timezone-aware UTC.

        Treat old timestamps without a timezone as UTC, matching their original storage.
        Convert timestamps that already have a timezone without changing the instant.
        """
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)


class ConversationPage(BaseModel):
    """Return one newest-first conversation page and the cursor for older conversations.

    has_more is True only when another page exists; next_cursor is None otherwise.
    """

    items: list[ConversationResponse]
    next_cursor: str | None
    has_more: bool


class MessagePage(BaseModel):
    """Return one message page, oldest to newest, and the cursor for an older page.

    The first page contains the newest messages. has_more and next_cursor tell the
    frontend whether it can request more history without fetching the whole chat.
    """

    items: list[MessageResponse]
    next_cursor: str | None
    has_more: bool


class FileManifestEntry(BaseModel):
    """Describe one file the browser says it is sending.

    name includes the extension, size_bytes declares its size, and client_file_id
    identifies its composer attachment. The server checks these against the actual
    multipart files in order; the declarations alone are not trusted.
    """

    model_config = ConfigDict(extra="forbid")

    client_file_id: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    name: Annotated[str, StringConstraints(min_length=1, max_length=255)]
    size_bytes: Annotated[int, Field(strict=True, ge=0)]


class SubmissionInput(BaseModel):
    """Accept the text, settings and file list for one press of Send.

    The browser generates client_submission_id once and reuses it on retries.
    Text may be empty if files are present. The service checks that at least text
    or a file exists and that files_manifest matches the actual received files.
    """

    model_config = ConfigDict(extra="forbid")

    client_submission_id: UUID
    content: Annotated[str, Field(max_length=32000)] = ""
    settings: ChatSettings
    files_manifest: list[FileManifestEntry] = Field(default_factory=list)


class SubmissionResponse(BaseModel):
    """Return the saved send's IDs and current processing state.

    message_id is the user message, run_id is its AI job, and documents lists user
    attachments. The same IDs are returned on retry/lookup. Acceptance means the
    send was saved, not that files finished indexing or the AI finished answering.
    """

    submission_id: UUID
    client_submission_id: UUID
    conversation_id: UUID
    message_id: UUID
    run_id: UUID
    status: str
    documents: list[DocumentResponse]
    error: dict | None = None


class DocumentPage(BaseModel):
    """Return a bounded newest-first document page without storage paths."""

    items: list[DocumentResponse]
    next_cursor: str | None
    has_more: bool
