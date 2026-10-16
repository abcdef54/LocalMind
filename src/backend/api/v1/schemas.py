"""Chat settings contract for upcoming v1 submissions; no AI initialization."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints


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
