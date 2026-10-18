"""Expose model discovery, persisted downloads and safe installation metadata under v1."""

import asyncio
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query

from src.backend.ai.llms import huggingface
from src.backend.api.errors import ApiError
from src.backend.model_management.config import ModelManagementSettings
from src.backend.model_management.schemas import (
    CatalogModelResponse,
    InstalledModelResponse,
    ModelDownloadInput,
    ModelDownloadResponse,
    ModelQuantResponse,
    validate_model_repository,
)
from src.backend.model_management.store import ModelRegistry

router = APIRouter(tags=["models"])


def get_model_registry() -> ModelRegistry:
    """Build the lightweight registry using application sessions/settings; perform no IO."""
    from src.backend.database.session import async_session_maker

    return ModelRegistry(async_session_maker, ModelManagementSettings.from_environment())


RegistryDependency = Annotated[ModelRegistry, Depends(get_model_registry)]


@router.get("/models", response_model=list[InstalledModelResponse])
async def list_installed_models(registry: RegistryDependency):
    """Return installed models and readiness without scanning disk or modifying configuration."""
    return await registry.list_installed_models()


@router.get("/models/catalog", response_model=list[CatalogModelResponse])
async def search_model_catalog(
    registry: RegistryDependency,
    query: Annotated[str | None, Query(max_length=200)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    """Search Hugging Face GGUF metadata with bounded query length, result count and wait.

    Args:
        registry: Supplies the server's metadata-request timeout.
        query: Optional model name fragment; omitted means popular GGUF repositories.
        limit: Maximum results, default 20 and capped at 100.

    Raises:
        ApiError: 504 on timeout or 502 when the external catalog is unavailable.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(huggingface.browse_models, query, limit),
            registry.settings.catalog_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise ApiError(504, "MODEL_CATALOG_TIMEOUT", "Model catalog request timed out") from None
    except Exception:
        raise ApiError(502, "MODEL_CATALOG_UNAVAILABLE", "Model catalog is unavailable") from None


@router.get("/models/quants", response_model=list[ModelQuantResponse])
async def list_model_quantizations(
    registry: RegistryDependency, repo_id: Annotated[str, Query(min_length=1, max_length=200)]
):
    """Return quantizations for one validated repository, grouping split parts by model.

    Args:
        registry: Supplies the metadata timeout.
        repo_id: Hugging Face owner/repository identifier, passed as a query parameter.

    Raises:
        ApiError: 422 for a malformed source, 504 for timeout or 502 for catalog failure.
    """
    try:
        validate_model_repository(repo_id)
    except ValueError:
        raise ApiError(
            422, "INVALID_MODEL_REPOSITORY", "Use an owner/repository identifier"
        ) from None
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(huggingface.list_quants, repo_id),
            registry.settings.catalog_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise ApiError(504, "MODEL_CATALOG_TIMEOUT", "Model catalog request timed out") from None
    except Exception:
        raise ApiError(502, "MODEL_CATALOG_UNAVAILABLE", "Model catalog is unavailable") from None


@router.get("/models/{model_id}", response_model=InstalledModelResponse)
async def get_installed_model(model_id: UUID, registry: RegistryDependency):
    """Return public registry metadata for model_id; a missing UUID returns 404."""
    return await registry.get_installed_model(model_id)


@router.delete("/models/{model_id}", response_model=InstalledModelResponse, status_code=202)
async def delete_installed_model(model_id: UUID, registry: RegistryDependency):
    """Record model_id's deletion intent; the worker removes config/files before deleting its row."""
    return await registry.request_model_deletion(model_id)


@router.post("/model-downloads", response_model=ModelDownloadResponse, status_code=202)
async def create_model_download(source: ModelDownloadInput, registry: RegistryDependency):
    """Save source's task and return its ID immediately; repeated sources reuse the same task.

    Args:
        source: Validated repository and first GGUF filename.
        registry: Persists/deduplicates tasks and applies queue limits.

    Failed/interrupted downloads are retried by submitting the same source again.
    No Hugging Face request or file download occurs in this handler.
    """
    return await registry.create_or_requeue_model_download(source)


@router.get("/model-downloads", response_model=list[ModelDownloadResponse])
async def list_model_downloads(
    registry: RegistryDependency, limit: Annotated[int, Query(ge=1, le=100)] = 50
):
    """Return at most limit recent tasks so a refreshed browser can restore download status."""
    return await registry.list_model_downloads(limit)


@router.get("/model-downloads/{task_id}", response_model=ModelDownloadResponse)
async def get_model_download(task_id: UUID, registry: RegistryDependency):
    """Read persisted task_id progress/result/error; this request never starts or retries work."""
    return await registry.get_model_download(task_id)
