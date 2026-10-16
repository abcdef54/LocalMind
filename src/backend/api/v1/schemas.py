"""Chat settings contract for upcoming v1 submissions; no AI initialization."""

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
    """Validate request settings; registry availability is checked by later services.

    Ranking group size is a per-run limit, independent of candidate count. The
    field is defined here but is not wired into the legacy LangGraph endpoint.
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
    model_config = ConfigDict(extra="forbid")

    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    created_at: datetime


class CitationResponse(BaseModel):
    id: UUID
    file_name: str
    file_type: str
    file_size: int
    deleted: bool


class MessageResponse(BaseModel):
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
    citations: list[CitationResponse] = Field(default_factory=list)

    @field_validator("created_at")
    @classmethod
    def serialize_legacy_utc(cls, timestamp: datetime) -> datetime:
        # Legacy message timestamps are naive UTC; preserve storage compatibility.
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)


class ConversationPage(BaseModel):
    items: list[ConversationResponse]
    next_cursor: str | None
    has_more: bool


class MessagePage(BaseModel):
    items: list[MessageResponse]
    next_cursor: str | None
    has_more: bool
