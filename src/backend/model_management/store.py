"""Persist download requests and coordinate registry changes with accepted chat runs."""

from pathlib import Path
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.backend.api.errors import ApiError
from src.backend.database.models import ACTIVE_RUN_STATUSES, InstalledModel, ModelDownload, Run
from src.backend.model_management.config import ModelManagementSettings
from src.backend.model_management.files import try_reserving_model_storage
from src.backend.model_management.schemas import (
    ModelDownloadInput,
    ModelDownloadResponse,
    InstalledModelResponse,
)

# Sends/retries and model deletion take the same transaction lock, so a deletion
# cannot overlook a run that is concurrently being accepted into the database.
MODEL_REGISTRY_LOCK = 731602101


async def lock_model_registry_and_check_selected_model(
    session: AsyncSession, model_name: str
) -> None:
    """Serialize run admission with model deletion and reject unavailable known models.

    Args:
        session: Existing submission/retry transaction; the lock lasts until it ends.
        model_name: Saved inference name selected for this run.

    Models managed externally by a separately configured inference server may not
    have a registry row. Preserve that existing behavior; known local models must
    be ready. This does not perform a remote health check or load a model.

    Raises:
        ApiError: A registered model is missing, being configured or being deleted.
    """
    await session.execute(select(func.pg_advisory_xact_lock(MODEL_REGISTRY_LOCK)))
    model = await session.scalar(
        select(InstalledModel).where(InstalledModel.model_name == model_name)
    )
    if model is not None and model.status != "ready":
        raise ApiError(409, "MODEL_NOT_READY", "The selected local model is not ready")


class ModelRegistry:
    """Save model/task requests without running downloads inside HTTP handlers.

    Args:
        sessions: Factory for short database transactions, also used by the worker.
        settings: Shared storage locations and queue admission limits.
    """

    def __init__(self, sessions: async_sessionmaker, settings: ModelManagementSettings):
        self.sessions = sessions
        self.settings = settings

    async def list_installed_models(self) -> list[InstalledModelResponse]:
        """Return registry metadata only; never scan files, download or rewrite configuration."""
        async with self.sessions() as session:
            models = await session.scalars(
                select(InstalledModel).order_by(
                    InstalledModel.installed_at.desc(), InstalledModel.id
                )
            )
            return [InstalledModelResponse.model_validate(model) for model in models]

    async def get_installed_model(self, model_id: UUID) -> InstalledModelResponse:
        """Return public metadata for model_id, or raise ApiError 404 when absent."""
        async with self.sessions() as session:
            model = await session.get(InstalledModel, model_id)
            if model is None:
                raise ApiError(404, "MODEL_NOT_FOUND", "Model was not found")
            return InstalledModelResponse.model_validate(model)

    async def create_or_requeue_model_download(
        self, source: ModelDownloadInput
    ) -> ModelDownloadResponse:
        """Persist one source's task and return immediately without contacting Hugging Face.

        Args:
            source: Validated repository and GGUF filename, including a possible first split part.

        Returns:
            Existing active/completed task, or the same failed/interrupted task reset
            to queued. Repeated requests never create duplicate active source tasks.
            Completed tasks whose model was removed may be explicitly queued again.

        Raises:
            ApiError: Queue capacity is reached or the model is being deleted.
        """
        async with self.sessions() as session, session.begin():
            await session.execute(select(func.pg_advisory_xact_lock(MODEL_REGISTRY_LOCK)))
            task = await session.scalar(
                select(ModelDownload)
                .where(
                    ModelDownload.repo_id == source.repo_id,
                    ModelDownload.gguf_filename == source.gguf_filename,
                )
                .with_for_update()
            )
            model = await session.scalar(
                select(InstalledModel).where(
                    InstalledModel.hf_repo == source.repo_id,
                    InstalledModel.gguf_file == source.gguf_filename,
                )
            )
            if task is not None and task.model_id is not None:
                model = await session.get(InstalledModel, task.model_id)
            if model is not None and model.status == "deleting":
                raise ApiError(409, "MODEL_DELETING", "Model deletion is still in progress")
            if task is not None and (
                task.status in {"queued", "downloading", "registering"}
                or task.status == "completed"
                and model is not None
                and model.status == "ready"
            ):
                return ModelDownloadResponse.model_validate(task)
            queued = await session.scalar(
                select(func.count())
                .select_from(ModelDownload)
                .where(ModelDownload.status == "queued")
            )
            needs_download = model is None or model.status not in {"ready", "configuring"}
            if needs_download and queued >= self.settings.max_queued_downloads:
                raise ApiError(429, "MODEL_DOWNLOAD_QUEUE_FULL", "Model download queue is full")
            if task is None:
                task = ModelDownload(repo_id=source.repo_id, gguf_filename=source.gguf_filename)
                session.add(task)
            if model is not None and model.status == "ready":
                task.status = "completed"
                task.model_id = model.id
                task.downloaded_bytes = model.size_bytes
                task.total_bytes = model.size_bytes
                task.source_revision = model.source_revision
            elif model is not None and model.status == "configuring":
                # Files are already registered; the worker only needs to publish config.
                task.status = "registering"
                task.model_id = model.id
            else:
                task.status = "queued"
                task.downloaded_bytes = 0
                task.total_bytes = None
                task.source_revision = None
                task.model_id = None
            task.error = None
            task.updated_at = func.clock_timestamp()
            await session.flush()
            await session.refresh(task)
            return ModelDownloadResponse.model_validate(task)

    async def get_model_download(self, task_id: UUID) -> ModelDownloadResponse:
        """Read persisted progress for task_id, including terminal failures; raise 404 if absent."""
        async with self.sessions() as session:
            task = await session.get(ModelDownload, task_id)
            if task is None:
                raise ApiError(404, "MODEL_DOWNLOAD_NOT_FOUND", "Model download was not found")
            return ModelDownloadResponse.model_validate(task)

    async def list_model_downloads(self, limit: int) -> list[ModelDownloadResponse]:
        """Return at most limit recent tasks so a refreshed frontend can rediscover downloads.

        Args:
            limit: Validated API page size from 1 to 100.
        """
        async with self.sessions() as session:
            tasks = await session.scalars(
                select(ModelDownload)
                .order_by(ModelDownload.updated_at.desc(), ModelDownload.id)
                .limit(limit)
            )
            return [ModelDownloadResponse.model_validate(task) for task in tasks]

    async def request_model_deletion(self, model_id: UUID) -> InstalledModelResponse:
        """Mark an unused nondefault model for worker cleanup and return its deleting state.

        Args:
            model_id: Installed model to remove from config, disk and the registry.

        Repeated requests are harmless. A filesystem lock serializes this decision
        with installation; the registry transaction lock serializes it with sends
        and retries. The worker persists deletion intent until cleanup succeeds.

        Raises:
            ApiError: Model is missing, default, used by an active/unconfirmed run,
                or another process is currently modifying shared model storage.
        """
        with try_reserving_model_storage(self.settings) as descriptor:
            if descriptor is None:
                raise ApiError(
                    409, "MODEL_STORAGE_BUSY", "Model storage is being updated; try again shortly"
                )
            async with self.sessions() as session, session.begin():
                await session.execute(select(func.pg_advisory_xact_lock(MODEL_REGISTRY_LOCK)))
                model = await session.get(InstalledModel, model_id, with_for_update=True)
                if model is None:
                    raise ApiError(404, "MODEL_NOT_FOUND", "Model was not found")
                if model.is_default:
                    raise ApiError(
                        409, "DEFAULT_MODEL_PROTECTED", "The default model cannot be deleted"
                    )
                active_run = await session.scalar(
                    select(Run.id)
                    .where(
                        Run.settings_snapshot["model_name"].astext == model.model_name,
                        or_(Run.status.in_(ACTIVE_RUN_STATUSES), Run.lease_owner.is_not(None)),
                    )
                    .limit(1)
                )
                if active_run is not None:
                    raise ApiError(
                        409, "MODEL_IN_USE", "An active or unconfirmed run uses this model"
                    )
                model.status = "deleting"
                await session.flush()
                return InstalledModelResponse.model_validate(model)
