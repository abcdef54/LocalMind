"""Index pending attachments with renewable DB reservations and per-file process locks."""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
import fcntl
import logging
from pathlib import Path
from typing import BinaryIO, TypeVar
from uuid import uuid4

from src.backend.indexing.config import IndexingSettings
from src.backend.indexing.errors import DocumentIndexingError
from src.backend.indexing.parsing import parse_document
from src.backend.indexing.parser_process import DocumentSection
from src.backend.indexing.store import IndexingClaim, IndexingStore
from src.backend.indexing.vectors import DocumentVectorIndex, split_sections
from src.backend.uploads.storage import run_file_operation

logger = logging.getLogger("uvicorn.error")
OperationResult = TypeVar("OperationResult")
DocumentParser = Callable[[Path, str, IndexingSettings], Awaitable[list[DocumentSection]]]


class IndexingWorker:
    """Find pending files, parse/embed them and commit readiness without calling an LLM.

    Args:
        store: DB reservation and state-transition operations.
        vector_index: Scoped Chroma index receiving attempt-specific vectors.
        lock_directory: Shared local directory for per-document OS locks. Locks
            prevent an expired worker and replacement from embedding the same file together.
        settings: Polling, concurrency, extraction and embedding-batch limits.
        parser: Async text extractor; production uses a memory/time-limited subprocess.
            Tests may inject a deterministic local parser without loading models.

    Constructing does not start work. Call start during backend startup and stop
    during shutdown. Only committed ready generations become available to retrieval.
    """

    def __init__(
        self,
        store: IndexingStore,
        vector_index: DocumentVectorIndex,
        lock_directory: Path,
        *,
        settings: IndexingSettings | None = None,
        parser: DocumentParser = parse_document,
    ) -> None:
        self.store = store
        self.vector_index = vector_index
        self.lock_directory = lock_directory
        self.settings = settings or IndexingSettings()
        self.parser = parser
        self.worker_id = uuid4().hex
        self._stop_requested = asyncio.Event()
        self._dispatcher: asyncio.Task | None = None
        self._active: set[asyncio.Task] = set()

    async def _database_call(self, operation: Awaitable[OperationResult]) -> OperationResult:
        """Await one store operation within the configured timeout.

        Args:
            operation: Awaitable DB operation; no model/parse IO belongs here.

        Returns:
            Its result, propagating database errors and asyncio.TimeoutError.
        """
        return await asyncio.wait_for(operation, self.settings.database_timeout_seconds)

    async def start(self) -> None:
        """Recover expired reservations and schedule polling, without awaiting file completion.

        Raises:
            RuntimeError: If this worker still has a dispatcher or unfinished jobs.
            Exception: If initial DB recovery fails, preventing unsafe partial startup.
        """
        if self._dispatcher is not None or self._active:
            raise RuntimeError("Indexing worker is already started or still stopping")
        await self._database_call(self.store.recover_expired())
        self._stop_requested.clear()
        self._dispatcher = asyncio.create_task(
            self._dispatch(), name="document-indexing-dispatcher"
        )

    async def _dispatch(self) -> None:
        """Poll pending files until shutdown, logging safe error types on DB failures.

        Start no more local jobs than max_concurrency; the store also checks
        currently leased jobs across workers. Task completion removes local tracking.
        """
        while not self._stop_requested.is_set():
            try:
                await self._database_call(self.store.recover_expired())
                while (
                    len(self._active) < self.settings.max_concurrency
                    and not self._stop_requested.is_set()
                ):
                    claim = await self._database_call(
                        self.store.claim_next(
                            worker_id=self.worker_id,
                            lease_seconds=self.settings.lease_seconds,
                            max_concurrency=self.settings.max_concurrency,
                        )
                    )
                    if claim is None:
                        break
                    task = asyncio.create_task(
                        self._index_one(claim), name=f"index-{claim.attachment_id}"
                    )
                    self._active.add(task)
                    task.add_done_callback(self._job_finished)
            except Exception as error:
                logger.error("Indexing dispatcher failed exception_type=%s", type(error).__name__)
            try:
                await asyncio.wait_for(self._stop_requested.wait(), self.settings.poll_seconds)
            except asyncio.TimeoutError:
                pass

    def _job_finished(self, task: asyncio.Task) -> None:
        """Remove a finished job and retrieve its exception without logging file contents.

        Args:
            task: Completed indexing task tracked in _active.
        """
        self._active.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("Indexing task failed exception_type=%s", type(task.exception()).__name__)

    def _acquire_file_lock(self, claim: IndexingClaim) -> BinaryIO | None:
        """Try to lock this document without waiting for an expired worker's native IO.

        Args:
            claim: Attachment UUID used as a lock filename, never the user filename.

        Returns:
            Open locked handle, or None when another process still uses the file.
            Callers must close the returned handle after all local work has stopped.
        """
        self.lock_directory.mkdir(parents=True, exist_ok=True)
        lock = (self.lock_directory / f"{claim.attachment_id}.lock").open("ab")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            return None
        except Exception:
            lock.close()
            raise
        return lock

    async def _renew_reservation(
        self, claim: IndexingClaim, job: asyncio.Task, stop_heartbeat: asyncio.Event
    ) -> None:
        """Renew a file reservation; cancel its job if ownership or DB access is lost.

        Args:
            claim: Current indexing identity.
            job: Coroutine handling parsing and bounded embedding batches.
            stop_heartbeat: Signal to finish without cancelling an in-flight DB call.

        Cancellation kills the parser, or waits for the current synchronous vector
        batch before cleanup. It cannot publish a replaced generation as ready.
        """
        while not stop_heartbeat.is_set():
            try:
                await asyncio.wait_for(stop_heartbeat.wait(), self.settings.heartbeat_seconds)
                return
            except asyncio.TimeoutError:
                pass
            try:
                renewed = await self._database_call(
                    self.store.heartbeat(
                        claim,
                        lease_seconds=self.settings.lease_seconds,
                    )
                )
            except Exception as error:
                logger.error(
                    "Indexing heartbeat failed attachment_id=%s exception_type=%s",
                    claim.attachment_id,
                    type(error).__name__,
                )
                renewed = False
            if not renewed:
                if not stop_heartbeat.is_set():
                    job.cancel()
                return

    async def _parse_and_embed(self, claim: IndexingClaim) -> int:
        """Parse a file, split its text and embed/write one bounded batch at a time.

        Args:
            claim: Server-selected path, original filename and vector namespace.

        Returns:
            Positive chunk count after all vectors were upserted.

        Raises:
            DocumentIndexingError: If extraction or chunk limits reject the file.
            Exception: For vector/embedding IO failures; the worker saves a generic code.
        """
        sections = await self.parser(claim.file_path, claim.file_name, self.settings)
        chunks = await run_file_operation(split_sections, sections, claim.file_name, self.settings)
        for offset in range(0, len(chunks), self.settings.embedding_batch_size):
            batch = chunks[offset : offset + self.settings.embedding_batch_size]
            await run_file_operation(self.vector_index.upsert_batch, claim, batch, offset)
        return len(chunks)

    async def _index_one(self, claim: IndexingClaim) -> None:
        """Index one claim and retain vectors only after confirmed or uncertain ready commit.

        Args:
            claim: Reserved file attempt. All vector cleanup is limited to its generation.

        Keep vectors on an uncertain DB commit: the commit may have succeeded.
        Retrieval still requires the exact ready generation in PostgreSQL. Expired
        attempts rebuild under a new generation and remove obsolete vectors later.
        """
        lock = None
        heartbeat = None
        stop_heartbeat = asyncio.Event()
        keep_vectors = False
        commit_outcome_unknown = False
        try:
            lock = self._acquire_file_lock(claim)
            if lock is None:
                return
            job = asyncio.current_task()
            heartbeat = asyncio.create_task(self._renew_reservation(claim, job, stop_heartbeat))
            try:
                chunk_count = await self._parse_and_embed(claim)
            except DocumentIndexingError as error:
                await self._database_call(
                    self.store.finish(claim, chunk_count=0, error_code=error.code)
                )
                return
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error(
                    "Document indexing failed attachment_id=%s exception_type=%s",
                    claim.attachment_id,
                    type(error).__name__,
                )
                await self._database_call(
                    self.store.finish(claim, chunk_count=0, error_code="DOCUMENT_INDEXING_FAILED")
                )
                return
            # The DB may commit ready and lose its acknowledgement. Do not erase
            # vectors merely because the following await raises or is cancelled.
            commit_outcome_unknown = True
            keep_vectors = await self._database_call(
                self.store.finish(claim, chunk_count=chunk_count)
            )
            commit_outcome_unknown = False
            if keep_vectors:
                try:
                    await run_file_operation(
                        self.vector_index.delete_obsolete_attempts,
                        claim.attachment_id,
                        claim.generation,
                    )
                except Exception as error:
                    logger.error(
                        "Obsolete vector cleanup failed attachment_id=%s exception_type=%s",
                        claim.attachment_id,
                        type(error).__name__,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.error(
                "Indexing state update failed attachment_id=%s exception_type=%s",
                claim.attachment_id,
                type(error).__name__,
            )
        finally:
            stop_heartbeat.set()
            if heartbeat is not None:
                # Let a short DB renewal finish rather than cancelling its connection
                # handshake immediately after committing the ready/failed state.
                with suppress(asyncio.CancelledError):
                    await asyncio.shield(heartbeat)
            try:
                if lock is not None and not keep_vectors and not commit_outcome_unknown:
                    await run_file_operation(
                        self.vector_index.delete_attempt, claim.attachment_id, claim.generation
                    )
                await self._database_call(self.store.release(claim))
            except Exception as error:
                logger.error(
                    "Indexing cleanup deferred attachment_id=%s exception_type=%s",
                    claim.attachment_id,
                    type(error).__name__,
                )
            finally:
                if lock is not None:
                    lock.close()

    async def stop(self) -> None:
        """Stop claiming work, drain active jobs briefly, then request cancellation.

        Wait for parser termination/current vector batches before releasing file
        locks. Jobs that take longer remain tracked; stale generations are never
        exposed as ready. Return without retrying failed files or invoking the LLM.
        """
        self._stop_requested.set()
        if self._dispatcher is not None:
            done, _ = await asyncio.wait(
                {self._dispatcher}, timeout=self.settings.database_timeout_seconds + 1
            )
            if not done:
                self._dispatcher.cancel()
                done, _ = await asyncio.wait({self._dispatcher}, timeout=1.0)
            if done:
                with suppress(asyncio.CancelledError):
                    await self._dispatcher
                self._dispatcher = None
            else:
                logger.error("Indexing dispatcher is still finishing its database operation")
        if not self._active:
            return
        _, pending = await asyncio.wait(
            set(self._active), timeout=self.settings.shutdown_grace_seconds
        )
        for task in pending:
            task.cancel()
        if pending:
            _, remaining = await asyncio.wait(pending, timeout=1.0)
            if remaining:
                logger.error("%s indexing jobs still finishing native IO", len(remaining))
