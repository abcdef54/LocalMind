"""Retry physical deletion of files marked as deleted without removing citation metadata."""

import asyncio
from contextlib import suppress
from datetime import datetime, timedelta, timezone
import fcntl
from functools import partial
import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.backend.database.models import Attachment
from src.backend.indexing.vectors import DocumentVectorIndex
from src.backend.services.documents import resolve_document_path
from src.backend.uploads.storage import run_file_operation

logger = logging.getLogger("uvicorn.error")


class DocumentCleanupWorker:
    """Remove deleted files and vectors, retrying failures after restart.

    Args:
        sessions: Factory for short independent database sessions.
        vector_index: Shared index supporting deletion of all document generations.
        storage_root: Upload root containing blobs and shared indexing_locks.

    Deleted document records in PostgreSQL identify pending cleanup across restarts.
    One pass handles at most 20 rows.
    The same OS lock used by indexing prevents deletion during a native write.
    Cleanup errors store a generic code and retry after 30 seconds. No model calls
    or document contents are needed. Multiple processes may safely repeat deletion.
    """

    def __init__(
        self, sessions: async_sessionmaker, vector_index: DocumentVectorIndex, storage_root: Path
    ) -> None:
        self.sessions = sessions
        self.vector_index = vector_index
        self.storage_root = storage_root
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        """Start checking saved deleted-document records; repeated starts do not duplicate polling."""
        if self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(self._poll())

    async def stop(self) -> None:
        """Stop polling and wait for current native deletion before releasing its lock."""
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None

    async def _poll(self) -> None:
        """Run bounded cleanup passes; retry database failures without logging private text."""
        while not self._stop.is_set():
            try:
                await self.retry_pending_document_cleanup()
            except Exception as error:
                logger.error("Document cleanup poll failed exception_type=%s", type(error).__name__)
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=1.0)

    async def retry_pending_document_cleanup(self) -> int:
        """Clean a bounded batch of deleted files due for retry.

        Returns:
            Number of documents whose physical cleanup completed in this pass.
            Locked documents remain pending; failures wait 30 seconds before retry.
        """
        retry_before = datetime.now(timezone.utc) - timedelta(seconds=30)
        async with self.sessions() as session:
            ids = list(
                (
                    await session.scalars(
                        select(Attachment.id)
                        .where(
                            Attachment.deleted_at.is_not(None),
                            Attachment.cleanup_completed_at.is_(None),
                            or_(
                                Attachment.cleanup_attempted_at.is_(None),
                                Attachment.cleanup_attempted_at <= retry_before,
                            ),
                        )
                        .order_by(Attachment.cleanup_attempted_at.asc().nullsfirst(), Attachment.id)
                        .limit(20)
                    )
                ).all()
            )
        completed = 0
        for document_id in ids:
            if self._stop.is_set():
                break
            completed += await self.cleanup_deleted_document(document_id)
        return completed

    async def cleanup_deleted_document(self, document_id: UUID) -> bool:
        """Remove the bytes and all vector generations of a document marked as deleted.

        Args:
            document_id: Saved attachment UUID; live rows are never cleaned.

        Returns:
            True after bytes/vectors are gone and completion is committed; False
            for live, missing, already-cleaned, locked or failed work. Missing bytes
            are successful deletion. Failed/uncertain commits can safely be retried.

        Keep the file lock until any synchronous vector deletion has finished.
        No DB transaction stays open during filesystem or Chroma operations.
        """
        lock_directory = self.storage_root / "indexing_locks"
        lock_directory.mkdir(parents=True, exist_ok=True)
        with (lock_directory / f"{document_id}.lock").open("ab") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                # Move a busy row behind other pending work so a large batch of
                # locked documents cannot starve unrelated cleanup indefinitely.
                async with self.sessions() as session, session.begin():
                    await session.execute(
                        update(Attachment)
                        .where(
                            Attachment.id == document_id,
                            Attachment.deleted_at.is_not(None),
                            Attachment.cleanup_completed_at.is_(None),
                        )
                        .values(cleanup_attempted_at=datetime.now(timezone.utc))
                    )
                return False
            async with self.sessions() as session:
                document = await session.get(Attachment, document_id)
                if (
                    document is None
                    or document.deleted_at is None
                    or document.cleanup_completed_at is not None
                ):
                    return False
                conversation_id = document.conversation_id
                # Detach metadata before closing the session; never keep a DB lock over IO.
                session.expunge(document)
            attempted_at = datetime.now(timezone.utc)
            error_code = None
            try:
                path = resolve_document_path(document, self.storage_root)
                await run_file_operation(
                    self.vector_index.delete_document_vectors, conversation_id, document_id
                )
                await run_file_operation(partial(path.unlink, missing_ok=True))
            except Exception as error:
                error_code = "DOCUMENT_CLEANUP_FAILED"
                logger.error(
                    "Document cleanup failed document_id=%s exception_type=%s",
                    document_id,
                    type(error).__name__,
                )
            async with self.sessions() as session, session.begin():
                await session.execute(
                    update(Attachment)
                    .where(
                        Attachment.id == document_id,
                        Attachment.deleted_at.is_not(None),
                        Attachment.cleanup_completed_at.is_(None),
                    )
                    .values(
                        cleanup_attempted_at=attempted_at,
                        cleanup_error=error_code,
                        extracted_text=None,
                        cleanup_completed_at=None if error_code else datetime.now(timezone.utc),
                    )
                )
            return error_code is None
