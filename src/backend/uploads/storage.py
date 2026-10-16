"""Save files from one Send request temporarily, then keep or delete them.

Receive bytes into staging/<submission_id>. Before the database commit, move the
folder to blobs/<submission_id>. A request holds a file lock until it finishes,
so abandoned-file cleanup skips folders still being used.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import fcntl
from pathlib import Path
import shutil
import time
from typing import Any, BinaryIO, TypeVar
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from src.backend.database.models import Submission

FileOperationResult = TypeVar("FileOperationResult")


async def run_file_operation(
    operation: Callable[..., FileOperationResult], *arguments: Any
) -> FileOperationResult:
    """Run a blocking file operation in a thread and return its result.

    operation is the synchronous function; arguments are passed to it unchanged.
    If the caller is cancelled, wait for the file operation to finish before raising
    CancelledError, so cleanup cannot delete a file while a thread is still writing.
    Errors from the operation propagate to the caller.
    """
    pending_operation = asyncio.create_task(run_in_threadpool(operation, *arguments))
    try:
        return await asyncio.shield(pending_operation)
    except asyncio.CancelledError:
        await pending_operation
        raise


@dataclass
class ReceivedUploadFile:
    """Track a received file's IDs, name, MIME label and actual byte count.

    attachment_id is the server UUID used on disk. file_name is the original full
    name shown to the user. mime_type is inferred from the extension, not proven
    from contents. size_bytes is updated as bytes arrive; client_file_id is assigned
    when the received file matches the browser's manifest. File contents are not kept here.
    """

    attachment_id: UUID
    file_name: str
    mime_type: str
    size_bytes: int = 0
    client_file_id: str = ""


class SubmissionFileBatch:
    """Manage files uploaded in one Send request.

    Create temporary storage, move files to permanent storage before the
    database commit, and delete unaccepted files when the request finishes.
    Keep files if the commit succeeded or its outcome is uncertain.

    storage_root contains staging and blobs directories. submission_id is generated
    by the server and names the batch folder. received_files follows upload order;
    current_directory changes after the move. Set keep_files_after_request only
    when files must survive request cleanup. Always call cleanup_and_release_lock.
    """

    def __init__(self, storage_root: Path) -> None:
        """Prepare IDs and paths for a new batch under storage_root without creating files.

        Start with an empty received_files list and keep_files_after_request=False.
        Call create_staging_directory before receiving bytes.
        """
        self.storage_root = storage_root.resolve()
        self.submission_id = uuid4()
        self.current_directory = self.storage_root / "staging" / str(self.submission_id)
        self.received_files: list[ReceivedUploadFile] = []
        self.moved_to_permanent_storage = False
        self.keep_files_after_request = False
        self._directory_lock: BinaryIO | None = None

    def create_staging_directory(self) -> None:
        """Create this batch's temporary directory and lock it until request cleanup.

        Return None. Store the lock handle on the batch. Raise OSError on failure;
        the caller must still call cleanup_and_release_lock to remove partial storage.
        """
        self.current_directory.mkdir(parents=True)
        self._directory_lock = (self.current_directory / ".lock").open("wb")
        fcntl.flock(self._directory_lock, fcntl.LOCK_EX)

    def move_to_permanent_storage(self) -> None:
        """Move the entire temporary folder to blobs/<submission_id> before committing the DB.

        All received files must be closed. Both folders must share a filesystem so one
        rename moves the batch. Update current_directory and moved_to_permanent_storage;
        leave keep_files_after_request for the service to set after the commit attempt,
        including when that attempt has an uncertain outcome.
        Return None, or raise OSError so the database transaction can roll back.
        """
        permanent_directory = self.storage_root / "blobs" / str(self.submission_id)
        permanent_directory.parent.mkdir(parents=True, exist_ok=True)
        self.current_directory.rename(permanent_directory)
        self.current_directory = permanent_directory
        self.moved_to_permanent_storage = True

    def get_storage_key(self, attachment_id: UUID) -> str:
        """Return blobs/<submission_id>/<attachment_id> for the given server file UUID.

        The result is relative to storage_root and is saved in attachment metadata.
        It contains no user filename. This method creates or moves no files.
        """
        return f"blobs/{self.submission_id}/{attachment_id}"

    def cleanup_and_release_lock(self) -> None:
        """Delete unaccepted batch files and release the directory lock.

        Call when the request finishes, including on errors. Keep accepted files and
        files whose DB commit outcome is uncertain. Return None. If deletion raises
        OSError, still close the lock so later cleanup can retry.
        """
        try:
            if not self.keep_files_after_request and self.current_directory.exists():
                shutil.rmtree(self.current_directory)
        finally:
            if self._directory_lock is not None:
                self._directory_lock.close()


async def remove_orphaned_upload_batches(
    session: AsyncSession, storage_root: Path, minimum_age_seconds: float
) -> int:
    """Delete old upload folders that no live request or saved submission still uses.

    Scan staging and blobs under storage_root. Skip locked folders and folders younger
    than minimum_age_seconds. Use session to check each submission UUID in PostgreSQL;
    keep every folder with a committed submission. Return the number of folders deleted.

    Do not change DB rows or scan old upload storage. File/DB errors propagate;
    a folder whose DB check fails is not deleted. Earlier confirmed orphans may
    already have been removed. Use a grace period longer than normal request completion.
    """
    removed_batch_count = 0
    for directory_name in ("staging", "blobs"):
        parent_directory = storage_root / directory_name
        if not parent_directory.exists():
            continue
        for batch_directory in parent_directory.iterdir():
            if not batch_directory.is_dir() or batch_directory.is_symlink():
                continue
            try:
                submission_id = UUID(batch_directory.name)
                directory_age_seconds = time.time() - batch_directory.stat().st_mtime
                directory_lock = (batch_directory / ".lock").open("ab")
            except (ValueError, FileNotFoundError):
                # Ignore unrelated folders and batches removed by another request.
                continue
            with directory_lock:
                try:
                    fcntl.flock(directory_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if directory_age_seconds < minimum_age_seconds:
                    continue
                committed_submission_id = await session.scalar(
                    select(Submission.id).where(Submission.id == submission_id)
                )
                if committed_submission_id is None:
                    await run_file_operation(shutil.rmtree, batch_directory)
                    removed_batch_count += 1
    return removed_batch_count
