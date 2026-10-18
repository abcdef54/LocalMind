"""Publish supported settings and actual server limits for the new frontend."""

from typing import get_args

from fastapi import APIRouter
from pydantic import BaseModel

from src.backend.ai.langgraph.graph_configs import TOURNAMENT_MODE_CONFIG
from src.backend.api.v1.schemas import ChatSettings, Personality
from src.backend.constants import (
    DEFAULT_MODEL_GGUF_FILE,
    DEFAULT_MODEL_REPO_ID,
    DEFAULT_RANKING_GROUP_SIZE,
    MAX_RANKING_GROUP_SIZE,
)
from src.backend.execution.config import ExecutorSettings
from src.backend.execution.events import EVENT_RETENTION
from src.backend.execution.replay import EVENT_PAGE_SIZE
from src.backend.memory.config import MemorySettings
from src.backend.model_management.config import ModelManagementSettings
from src.backend.uploads.config import SUPPORTED_EXTENSIONS, UploadSettings

router = APIRouter(tags=["capabilities"])


class CapabilitiesResponse(BaseModel):
    """Describe settings/limits shared by clients; no credentials or filesystem paths are exposed."""

    chat: dict
    uploads: dict
    pagination: dict
    events: dict
    model_downloads: dict


@router.get("/capabilities", response_model=CapabilitiesResponse)
async def get_server_capabilities() -> CapabilitiesResponse:
    """Return current environment limits and options from the same definitions used by backend code.

    Values describe configured support, not hardware detection or per-model abilities.
    In particular tools_enabled is an app option, not a claim every downloaded GGUF
    supports tool calling. This performs no DB, inference or network operation.
    """
    uploads = UploadSettings.from_environment()
    memory = MemorySettings.from_environment()
    models = ModelManagementSettings.from_environment()
    executor = ExecutorSettings.from_environment()
    defaults = {
        name: ChatSettings.model_fields[name].default
        for name in ("personality", "reasoning_mode", "temperature", "tools_enabled")
    }
    temperature_bounds = next(
        metadata
        for metadata in ChatSettings.model_fields["temperature"].metadata
        if hasattr(metadata, "ge")
    )
    temperature_maximum = next(
        metadata.le
        for metadata in ChatSettings.model_fields["temperature"].metadata
        if hasattr(metadata, "le")
    )
    return CapabilitiesResponse(
        chat={
            "defaults": defaults,
            "personalities": list(get_args(Personality)),
            "reasoning_modes": [
                {"name": name, "candidate_count": count}
                for name, count in TOURNAMENT_MODE_CONFIG.items()
            ],
            "ranking_group_size": {
                "default": DEFAULT_RANKING_GROUP_SIZE,
                "min": 2,
                "max": MAX_RANKING_GROUP_SIZE,
            },
            "temperature": {"min": temperature_bounds.ge, "max": temperature_maximum},
            "context_tokens": memory.context_tokens,
            "reply_tokens": memory.reply_tokens,
            "max_concurrent_runs": executor.max_concurrency,
        },
        uploads={
            "extensions": sorted(SUPPORTED_EXTENSIONS),
            "max_file_bytes": uploads.max_file_bytes,
            "max_batch_bytes": uploads.max_batch_bytes,
            "max_files": uploads.max_files,
            "duplicate_policy": "same_exact_filename_in_same_conversation",
        },
        pagination={"default_limit": 50, "max_limit": 100},
        events={
            "sse": True,
            "replay": True,
            "schema_version": 1,
            "retained_events_per_run": EVENT_RETENTION,
            "max_page_size": EVENT_PAGE_SIZE,
        },
        model_downloads={
            "background": True,
            "max_concurrent_downloads": 1,
            "max_queued_downloads": models.max_queued_downloads,
            "timeout_seconds": models.download_timeout_seconds,
            "retry": "submit_same_source",
            "suggested_default": {
                "repo_id": DEFAULT_MODEL_REPO_ID,
                "gguf_filename": DEFAULT_MODEL_GGUF_FILE,
            },
        },
    )
