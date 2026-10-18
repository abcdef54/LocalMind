"""Poll durable model tasks, own download subprocesses and publish complete installations."""

import asyncio
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from uuid import UUID, uuid4

from sqlalchemy import func, select

from src.backend.ai.llms.huggingface import extract_quant, is_secondary_split_part
from src.backend.constants import DEFAULT_MODEL_GGUF_FILE, DEFAULT_MODEL_REPO_ID
from src.backend.database.models import InstalledModel, ModelDownload
from src.backend.model_management.files import (
    resolve_managed_model_path,
    select_all_parts_of_requested_model,
    try_reserving_model_storage,
    validate_complete_model_files,
    write_llama_swap_config_for_installed_models,
)
from src.backend.model_management.store import ModelRegistry
from src.backend.uploads.storage import run_file_operation

logger = logging.getLogger("uvicorn.error")


class ModelDownloadWorker:
    """Process downloads independently of chat inference and browser connections.

    Args:
        registry: Session factory/settings shared with model API handlers.

    A shared filesystem lock permits one download/config mutation at a time, even
    with multiple backend processes sharing the same model volume. The child inherits
    that lock: after a parent crash, another worker waits until the child exits before
    marking its task interrupted. Retries are explicit POST requests, never automatic.
    """

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.settings = registry.settings
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        """Start background polling immediately; do not await model scans or downloads."""
        if self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(
                self._poll_for_downloads_and_registry_changes(), name="model-download-worker"
            )

    async def stop(self) -> None:
        """Stop polling, terminate/reap an active child and preserve its interrupted task."""
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _poll_for_downloads_and_registry_changes(self) -> None:
        """Reserve shared storage, reconcile changes and process one queued task per iteration."""
        while not self._stop.is_set():
            try:
                with try_reserving_model_storage(self.settings) as descriptor:
                    if descriptor is not None:
                        await self._reconcile_local_models_and_configuration()
                        download = (
                            await self._recover_abandoned_downloads_and_reserve_next_queued_task()
                        )
                        if download is not None:
                            await self._download_and_register_model(download, descriptor)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error("Model worker failed exception_type=%s", type(error).__name__)
            try:
                await asyncio.wait_for(self._stop.wait(), self.settings.poll_seconds)
            except asyncio.TimeoutError:
                pass

    async def _recover_abandoned_downloads_and_reserve_next_queued_task(self) -> dict | None:
        """Recover ownerless tasks and reserve the oldest queued source under the storage lock.

        Returns:
            Task ID, source and incremented attempt, or None when the queue is empty.
            Obtaining the inherited filesystem lock proves old download children have
            exited, so interrupted tasks cannot overlap new writers to the same files.
        """
        async with self.registry.sessions() as session, session.begin():
            abandoned = await session.scalars(
                select(ModelDownload)
                .where(ModelDownload.status.in_(["downloading", "registering"]))
                .with_for_update()
            )
            for task in abandoned:
                model = await session.get(InstalledModel, task.model_id) if task.model_id else None
                if model is not None and model.status == "ready":
                    task.status = "completed"
                    task.total_bytes = model.size_bytes
                    task.downloaded_bytes = model.size_bytes
                    task.source_revision = model.source_revision
                    task.error = None
                else:
                    task.status = "interrupted"
                    task.error = {
                        "code": "DOWNLOAD_INTERRUPTED",
                        "message": "The previous worker stopped before recording completion",
                    }
                task.updated_at = func.clock_timestamp()
            task = await session.scalar(
                select(ModelDownload)
                .where(ModelDownload.status == "queued")
                .order_by(ModelDownload.created_at, ModelDownload.id)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if task is None:
                return None
            model = await session.get(InstalledModel, task.model_id) if task.model_id else None
            if model is None:
                model = await session.scalar(
                    select(InstalledModel).where(
                        InstalledModel.hf_repo == task.repo_id,
                        InstalledModel.gguf_file == task.gguf_filename,
                    )
                )
            if model is not None and model.status == "ready":
                task.status = "completed"
                task.model_id = model.id
                task.total_bytes = model.size_bytes
                task.downloaded_bytes = model.size_bytes
                task.source_revision = model.source_revision
                task.error = None
                task.updated_at = func.clock_timestamp()
                return None
            task.status = "downloading"
            task.attempt += 1
            task.updated_at = func.clock_timestamp()
            return {
                "id": task.id,
                "repo_id": task.repo_id,
                "gguf_filename": task.gguf_filename,
                "attempt": task.attempt,
            }

    def _scan_legacy_model_files(self) -> list[dict]:
        """Describe complete GGUF models manually placed directly in the model root.

        Returns:
            First-file paths and complete root-relative manifests. Ignore partial,
            malformed and secondary split files. Managed task directories are never
            imported by this scan, preventing partial downloads from becoming models.
        """
        available = [path.name for path in self.settings.root.iterdir() if path.is_file()]
        models = []
        for filename in sorted(available):
            if is_secondary_split_part(filename):
                continue
            try:
                names = select_all_parts_of_requested_model(filename, available)
                files = [
                    {
                        "filename": name,
                        "size_bytes": resolve_managed_model_path(self.settings.root, name)
                        .stat()
                        .st_size,
                    }
                    for name in names
                ]
                size = validate_complete_model_files(self.settings.root, files)
            except (ValueError, OSError):
                continue
            models.append({"gguf_file": filename, "files": files, "size_bytes": size})
        return models

    async def _reconcile_local_models_and_configuration(self) -> None:
        """Import complete legacy files, retry deletions and rebuild config under the storage lock.

        Keep missing rows visible instead of silently deleting them. Configuring rows
        become ready only after all files and configuration have been checked/written.
        A deletion removes config first, then bytes, then its row; a crash leaves
        deleting metadata so the next iteration safely repeats cleanup.
        """
        scanned = await run_file_operation(self._scan_legacy_model_files)
        async with self.registry.sessions() as session, session.begin():
            models = list(
                (
                    await session.scalars(
                        select(InstalledModel).order_by(
                            InstalledModel.installed_at, InstalledModel.id
                        )
                    )
                ).all()
            )
            known_files = {model.gguf_file for model in models}
            known_names = {model.model_name for model in models}
            for file in scanned:
                name = Path(file["gguf_file"]).stem
                if file["gguf_file"] in known_files or name in known_names:
                    continue
                is_builtin = file["gguf_file"] == DEFAULT_MODEL_GGUF_FILE
                model = InstalledModel(
                    model_name=name,
                    display_name=name,
                    hf_repo=DEFAULT_MODEL_REPO_ID if is_builtin else f"local/{name}",
                    local_path=str((self.settings.root / file["gguf_file"]).resolve()),
                    quantization=extract_quant(file["gguf_file"]),
                    is_default=is_builtin,
                    status="configuring",
                    **file,
                )
                session.add(model)
                models.append(model)
                known_files.add(file["gguf_file"])
                known_names.add(name)
            managed_root = self.settings.root / "managed"
            referenced_directories = {
                Path(model.gguf_file).parts[1]
                for model in models
                if len(Path(model.gguf_file).parts) > 2
                and Path(model.gguf_file).parts[0] == "managed"
            }
            if managed_root.is_dir():
                for directory in managed_root.iterdir():
                    try:
                        UUID(directory.name)
                    except ValueError:
                        continue
                    if (
                        directory.name not in referenced_directories
                        and directory.is_dir()
                        and not directory.is_symlink()
                    ):
                        await run_file_operation(shutil.rmtree, directory)
            usable = []
            for model in models:
                if model.status == "deleting":
                    continue
                try:
                    # Do not trust historical absolute paths for deleting/serving files.
                    expected = resolve_managed_model_path(self.settings.root, model.gguf_file)
                    if expected != Path(model.local_path).resolve():
                        raise ValueError("Registry path disagrees with model storage")
                    if not model.files:
                        model.files = next(
                            (
                                file["files"]
                                for file in scanned
                                if file["gguf_file"] == model.gguf_file
                            ),
                            [],
                        )
                    await run_file_operation(
                        validate_complete_model_files, self.settings.root, model.files
                    )
                except (ValueError, OSError):
                    model.status = "missing"
                    continue
                usable.append(model)
            await run_file_operation(
                write_llama_swap_config_for_installed_models,
                self.settings,
                [
                    {"model_name": model.model_name, "gguf_file": model.gguf_file}
                    for model in usable
                ],
            )
            for model in usable:
                model.status = "ready"
            if usable and not any(model.is_default for model in models):
                usable[0].is_default = True
            for model in models:
                if model.status == "deleting":
                    await run_file_operation(self._delete_model_files, model.gguf_file, model.files)
                    await session.delete(model)

    def _delete_model_files(self, first_filename: str, files: list[dict]) -> None:
        """Delete only validated model parts, keeping unrelated models and private paths untouched.

        Args:
            first_filename: First installed part, used to verify the manifest includes this model.
            files: Root-relative manifest saved at installation or legacy reconciliation.
        """
        if not files:
            files = [{"filename": first_filename}]
        if first_filename not in {file["filename"] for file in files}:
            raise ValueError("Missing deletion manifest")
        paths = [resolve_managed_model_path(self.settings.root, file["filename"]) for file in files]
        for path in paths:
            path.unlink(missing_ok=True)
        # Only downloaded installations have their own directory; remove its HF cache too.
        parts = Path(first_filename).parts
        if len(parts) > 2 and parts[0] == "managed":
            UUID(parts[1])
            directory = self.settings.root / "managed" / parts[1]
            if directory.exists() and directory.resolve().is_relative_to(
                self.settings.root.resolve()
            ):
                shutil.rmtree(directory)

    async def _save_download_progress(
        self, download: dict, progress: dict, *, registering: bool = False
    ) -> None:
        """Save completed-part byte counts for this exact attempt using a short transaction.

        Args:
            download: Worker-reserved task ID and attempt.
            progress: Child's validated JSON manifest and byte counters.
            registering: True after successful download, before filesystem publication.
        """
        async with self.registry.sessions() as session, session.begin():
            task = await session.get(ModelDownload, download["id"], with_for_update=True)
            if task.attempt != download["attempt"] or task.status not in {
                "downloading",
                "registering",
            }:
                raise ValueError("Download attempt was replaced")
            task.total_bytes = progress["total_bytes"]
            task.downloaded_bytes = progress["downloaded_bytes"]
            task.source_revision = progress["source_revision"]
            task.status = "registering" if registering else "downloading"
            task.updated_at = func.clock_timestamp()

    async def _record_download_failure(
        self, download: dict, code: str, *, interrupted: bool = False
    ) -> None:
        """Persist a safe terminal failure after the child has exited.

        Args:
            download: Reserved task identity and attempt; replaced attempts are untouched.
            code: Public failure identifier, without external exception messages.
            interrupted: True for shutdown; another POST is required to try again.
        """
        async with self.registry.sessions() as session, session.begin():
            task = await session.get(ModelDownload, download["id"], with_for_update=True)
            if task.attempt == download["attempt"] and task.status in {
                "downloading",
                "registering",
            }:
                task.status = "interrupted" if interrupted else "failed"
                task.error = {
                    "code": code,
                    "message": "Model download did not complete; submit the source again to retry",
                }
                task.updated_at = func.clock_timestamp()

    async def _download_and_register_model(self, download: dict, descriptor: int) -> None:
        """Run a bounded child, save progress and publish files only after successful completion.

        Args:
            download: Reserved task source, ID and attempt.
            descriptor: Storage lock inherited by the child, retained until it is reaped.
        """
        job_directory = self.settings.root / ".downloads" / str(download["id"])
        process = None
        try:
            job_directory.mkdir(parents=True, exist_ok=True)
            (job_directory / "progress.json").unlink(missing_ok=True)
            creation = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "src.backend.model_management.download_process",
                    str(job_directory.resolve()),
                    download["repo_id"],
                    download["gguf_filename"],
                    str(os.getpid()),
                    pass_fds=(descriptor,),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            )
            try:
                process = await asyncio.shield(creation)
            except asyncio.CancelledError:
                # Reap a child even when shutdown arrives while the OS is spawning it.
                process = await creation
                raise
            started = time.monotonic()
            last_progress = None
            while process.returncode is None:
                if time.monotonic() - started > self.settings.download_timeout_seconds:
                    raise asyncio.TimeoutError()
                try:
                    await asyncio.wait_for(process.wait(), self.settings.poll_seconds)
                except asyncio.TimeoutError:
                    pass
                progress_file = job_directory / "progress.json"
                if progress_file.exists():
                    progress = json.loads(progress_file.read_text())
                    if progress != last_progress:
                        await self._save_download_progress(download, progress)
                        last_progress = progress
            # A very fast child may exit before the polling loop starts.
            progress_file = job_directory / "progress.json"
            if progress_file.exists():
                progress = json.loads(progress_file.read_text())
                if progress != last_progress:
                    await self._save_download_progress(download, progress)
                    last_progress = progress
            if process.returncode != 0 or last_progress is None:
                raise ValueError("Download process failed")
            await self._save_download_progress(download, last_progress, registering=True)
            await self._publish_downloaded_files_and_register_model(
                download, job_directory, last_progress
            )
        except asyncio.CancelledError:
            if process is not None and process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
            await self._record_download_failure(download, "DOWNLOAD_INTERRUPTED", interrupted=True)
            raise
        except Exception as error:
            if process is not None and process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
            code = (
                "DOWNLOAD_TIMEOUT"
                if isinstance(error, asyncio.TimeoutError)
                else "MODEL_DOWNLOAD_FAILED"
            )
            await self._record_download_failure(download, code)
            logger.error(
                "Model download failed task_id=%s exception_type=%s",
                download["id"],
                type(error).__name__,
            )

    async def _publish_downloaded_files_and_register_model(
        self, download: dict, job_directory: Path, progress: dict
    ) -> None:
        """Move complete files, register configuring metadata, write config and finish the task.

        Args:
            download: Current task identity and pinned source selection.
            job_directory: Directory containing the child's completed files subdirectory.
            progress: Resolved revision and complete file manifest validated again here.

        A move/commit failure leaves an isolated installation for an explicit retry.
        Existing ready models are never overwritten, even for identical basenames.
        """
        await run_file_operation(
            validate_complete_model_files, job_directory / "files", progress["files"]
        )
        model_name = Path(download["gguf_filename"]).stem
        installation_id = uuid4()
        relative_directory = Path("managed") / str(installation_id)
        installed_directory = self.settings.root / relative_directory
        async with self.registry.sessions() as session, session.begin():
            existing = await session.scalar(
                select(InstalledModel).where(InstalledModel.model_name == model_name)
            )
            if existing is not None:
                if existing.hf_repo != download["repo_id"] or existing.status == "ready":
                    raise ValueError("Model inference name already exists")
                if existing.status == "deleting":
                    raise ValueError("Model deletion in progress")
                model = existing
            else:
                model = InstalledModel(
                    model_name=model_name,
                    is_default=not bool(
                        await session.scalar(
                            select(InstalledModel.id)
                            .where(InstalledModel.is_default.is_(True))
                            .limit(1)
                        )
                    ),
                )
                session.add(model)
            installed_directory.parent.mkdir(parents=True, exist_ok=True)
            (job_directory / "files").rename(installed_directory)
            model.status = "configuring"
            model.hf_repo = download["repo_id"]
            model.gguf_file = (relative_directory / download["gguf_filename"]).as_posix()
            model.local_path = str(resolve_managed_model_path(self.settings.root, model.gguf_file))
            model.display_name = model_name
            model.quantization = extract_quant(download["gguf_filename"])
            model.size_bytes = progress["total_bytes"]
            model.source_revision = progress["source_revision"]
            model.files = [
                {**file, "filename": (relative_directory / file["filename"]).as_posix()}
                for file in progress["files"]
            ]
            await session.flush()
            model_id = model.id
            task = await session.get(ModelDownload, download["id"], with_for_update=True)
            if task.attempt != download["attempt"] or task.status != "registering":
                raise ValueError("Download attempt was replaced")
            task.model_id = model_id
        # Config failures leave a configuring row, not a falsely ready installation.
        await self._reconcile_local_models_and_configuration()
        async with self.registry.sessions() as session, session.begin():
            task = await session.get(ModelDownload, download["id"], with_for_update=True)
            model = await session.get(InstalledModel, model_id)
            if (
                task.attempt != download["attempt"]
                or task.status != "registering"
                or model.status != "ready"
            ):
                raise ValueError("Installation is no longer eligible for completion")
            task.status = "completed"
            task.model_id = model_id
            task.error = None
            task.updated_at = func.clock_timestamp()
        # Registration is now durable. Remove staging metadata; installed files stay.
        await run_file_operation(shutil.rmtree, job_directory, True)
