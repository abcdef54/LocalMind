"""Reserve attachment indexing jobs and queue a reply only when every input file is ready."""

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.backend.api.errors import ApiError
from src.backend.execution.events import update_state_and_add_run_event
from src.backend.database.models import (
    ACTIVE_RUN_STATUSES,
    Attachment,
    Conversation,
    Run,
    Submission,
)
from src.backend.services.conversation_history import get_conversation

INDEXING_ADMISSION_LOCK = 1280134735


@dataclass(frozen=True)
class IndexingClaim:
    """Identify one file and the indexing attempt allowed to make its vectors visible.

    Args:
        attachment_id: Server document UUID.
        conversation_id: Chat that owns the document; added to every vector.
        submission_id: Send that owns the document, or None for a legacy document.
        file_path: Local file location to parse, including UUID-only storage names.
        file_name: Original filename used to choose PDF/DOCX/text parsing.
        generation: Unique vector namespace for this attempt, never reused after recovery.
        owner: Worker token that must still match before committing a result.
        attempt: Increasing attempt number used to reject stale worker writes.
    """

    attachment_id: UUID
    conversation_id: UUID
    submission_id: UUID | None
    file_path: Path
    file_name: str
    generation: UUID
    owner: str
    attempt: int


def owned_indexing_claim(claim: IndexingClaim):
    """Return SQL conditions for an unexpired attempt still allowed to change its file.

    Args:
        claim: IDs, owner, generation and attempt saved when this file was reserved.

    Returns:
        SQL expression checking all identities, indexing status and PostgreSQL time.
        No query runs until the caller includes it in a database operation.
    """
    return and_(
        Attachment.id == claim.attachment_id,
        Attachment.index_owner == claim.owner,
        Attachment.index_generation == claim.generation,
        Attachment.index_attempt == claim.attempt,
        Attachment.status == "indexing",
        Attachment.deleted_at.is_(None),
        Attachment.index_expires_at > func.clock_timestamp(),
    )


class IndexingStore:
    """Save indexing reservations and file/send/run transitions in short transactions.

    Args:
        sessions: Factory for independent async database sessions. Model calls and
            file parsing never keep one of these transactions or row locks open.
    """

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def claim_next(
        self, *, worker_id: str, lease_seconds: float, max_concurrency: int
    ) -> IndexingClaim | None:
        """Reserve an eligible pending attachment and give its vectors a new generation.

        Args:
            worker_id: Unique worker label, at most 64 characters.
            lease_seconds: Reservation lifetime measured from PostgreSQL's clock.
            max_concurrency: Maximum currently leased indexing jobs in this database.

        Returns:
            A claim after committing indexing status/owner/attempt, or None when
            admission is busy/full, no file is eligible, or a selected chat is locked.

        Raises:
            ValueError: If the worker label or limits are invalid.
        """
        if not worker_id or len(worker_id) > 64 or lease_seconds <= 0 or max_concurrency < 1:
            raise ValueError("Invalid indexing worker or reservation limits")
        async with self.sessions.begin() as session:
            if not await session.scalar(
                select(func.pg_try_advisory_xact_lock(INDEXING_ADMISSION_LOCK))
            ):
                return None
            leased_count = await session.scalar(
                select(func.count())
                .select_from(Attachment)
                .where(
                    Attachment.status == "indexing",
                    Attachment.index_expires_at > func.clock_timestamp(),
                )
            )
            if leased_count >= max_concurrency:
                return None
            candidate = (
                await session.execute(
                    select(Attachment.id, Attachment.conversation_id)
                    .join(Conversation, Conversation.id == Attachment.conversation_id)
                    .outerjoin(Submission, Submission.id == Attachment.submission_id)
                    .outerjoin(Run, Run.submission_id == Attachment.submission_id)
                    .where(
                        Attachment.status == "pending",
                        Attachment.deleted_at.is_(None),
                        Conversation.deleted_at.is_(None),
                        or_(
                            Attachment.submission_id.is_(None),
                            and_(
                                Submission.status.in_(("accepted", "indexing", "ready", "failed")),
                                ~Run.status.in_(("cancel_requested", "cancelled")),
                                Run.cancel_requested.is_(False),
                            ),
                        ),
                    )
                    .order_by(Attachment.created_at, Attachment.id)
                    .limit(1)
                )
            ).first()
            if candidate is None:
                return None
            # All send/retry/completion mutations lock the chat before file rows.
            # Use the same order to avoid deadlocks when several files finish together.
            conversation = await session.scalar(
                select(Conversation)
                .where(
                    Conversation.id == candidate.conversation_id,
                    Conversation.deleted_at.is_(None),
                )
                .with_for_update(skip_locked=True)
            )
            if conversation is None:
                return None
            attachment = await session.scalar(
                select(Attachment)
                .where(
                    Attachment.id == candidate.id,
                    Attachment.status == "pending",
                    Attachment.deleted_at.is_(None),
                )
                .with_for_update(skip_locked=True)
            )
            if attachment is None:
                return None
            if attachment.submission_id is not None:
                submission = await session.get(Submission, attachment.submission_id)
                run = await session.scalar(
                    select(Run).where(Run.submission_id == attachment.submission_id)
                )
                if (
                    submission is None
                    or submission.status == "cancelled"
                    or run is None
                    or run.cancel_requested
                    or run.status in ("cancel_requested", "cancelled")
                ):
                    return None
            attachment.status = "indexing"
            attachment.index_owner = f"{worker_id}:{uuid4().hex}"
            attachment.index_generation = uuid4()
            attachment.index_attempt += 1
            attachment.index_expires_at = func.clock_timestamp() + timedelta(seconds=lease_seconds)
            attachment.error = None
            if attachment.submission_id is not None:
                await session.execute(
                    update(Submission)
                    .where(
                        Submission.id == attachment.submission_id,
                        Submission.status != "failed",
                    )
                    .values(status="indexing", error=None)
                )
            await session.flush()
            if attachment.submission_id is not None and run.status == "pending":
                run = await session.scalar(select(Run).where(Run.id == run.id).with_for_update())
                await update_state_and_add_run_event(
                    session, run, "documents.indexing", {"document_id": str(attachment.id)}
                )
            return IndexingClaim(
                attachment.id,
                attachment.conversation_id,
                attachment.submission_id,
                Path(attachment.file_path),
                attachment.file_name,
                attachment.index_generation,
                attachment.index_owner,
                attachment.index_attempt,
            )

    async def heartbeat(self, claim: IndexingClaim, *, lease_seconds: float) -> bool:
        """Extend an indexing reservation only while its file/chat/send remain eligible.

        Args:
            claim: Current indexing attempt.
            lease_seconds: Seconds to add from PostgreSQL's present clock time.

        Returns:
            True if renewed; False if expired, replaced, deleted or cancelled.
        """
        async with self.sessions.begin() as session:
            chat_live = (
                select(Conversation.id)
                .where(
                    Conversation.id == claim.conversation_id,
                    Conversation.deleted_at.is_(None),
                )
                .exists()
            )
            send_live = or_(
                Attachment.submission_id.is_(None),
                select(Run.id)
                .where(
                    Run.submission_id == Attachment.submission_id,
                    ~Run.status.in_(("cancel_requested", "cancelled")),
                    Run.cancel_requested.is_(False),
                )
                .exists(),
            )
            renewed = await session.scalar(
                update(Attachment)
                .where(
                    owned_indexing_claim(claim),
                    chat_live,
                    send_live,
                )
                .values(index_expires_at=func.clock_timestamp() + timedelta(seconds=lease_seconds))
                .returning(Attachment.id)
            )
            return renewed is not None

    async def finish(
        self, claim: IndexingClaim, *, chunk_count: int, error_code: str | None = None
    ) -> bool:
        """Commit this attempt's ready/failed state and update its waiting submission/run.

        Args:
            claim: Attempt whose vectors have finished writing or whose work failed.
            chunk_count: Number of usable chunks, positive on success and zero on failure.
            error_code: Safe public failure code; None means success.

        Returns:
            True after committing a current attempt. False without making vectors
            visible if ownership/expiry/chat/run status changed during computation.

        Raises:
            ValueError: If success reports no chunks or failure reports stored chunks.
        """
        if (error_code is None and chunk_count <= 0) or (
            error_code is not None and chunk_count != 0
        ):
            raise ValueError("Invalid indexing completion count")
        async with self.sessions.begin() as session:
            conversation = await session.scalar(
                select(Conversation)
                .where(
                    Conversation.id == claim.conversation_id,
                    Conversation.deleted_at.is_(None),
                )
                .with_for_update()
            )
            if conversation is None:
                return False
            attachment = await session.scalar(
                select(Attachment)
                .where(
                    owned_indexing_claim(claim),
                )
                .with_for_update()
            )
            if attachment is None:
                return False
            if claim.submission_id is not None:
                run = await session.scalar(
                    select(Run).where(Run.submission_id == claim.submission_id)
                )
                if (
                    run is None
                    or run.cancel_requested
                    or run.status in ("cancel_requested", "cancelled")
                ):
                    return False
            attachment.status = "ready" if error_code is None else "failed"
            attachment.chunk_count = chunk_count
            attachment.error = (
                None
                if error_code is None
                else {
                    "code": error_code,
                    "message": "Document indexing failed",
                }
            )
            attachment.index_owner = None
            attachment.index_expires_at = None
            await session.flush()
            if claim.submission_id is not None:
                run = await session.scalar(
                    select(Run).where(Run.submission_id == claim.submission_id).with_for_update()
                )
                if run is not None and run.status == "pending":
                    await update_state_and_add_run_event(
                        session,
                        run,
                        "documents.ready" if error_code is None else "documents.failed",
                        {"document_id": str(attachment.id)},
                    )
                await self._update_waiting_submission(session, claim.submission_id)
            return True

    async def _update_waiting_submission(self, session: AsyncSession, submission_id: UUID) -> None:
        """Queue the reply only after all send files are ready, or fail it if any file failed.

        Args:
            session: Current transaction holding the conversation lock.
            submission_id: Send whose attachments were just updated.

        Returns:
            None. Save submission/run states together; leave partially indexed sends waiting.
        """
        attachments = list(
            (
                await session.scalars(
                    select(Attachment).where(
                        Attachment.submission_id == submission_id,
                    )
                )
            ).all()
        )
        submission = await session.get(Submission, submission_id)
        run = await session.scalar(select(Run).where(Run.submission_id == submission_id))
        failed_file = next(
            (
                file
                for file in attachments
                if file.status in ("failed", "deleted") or file.deleted_at is not None
            ),
            None,
        )
        if failed_file is not None:
            submission.status = "failed"
            submission.error = {
                "code": "DOCUMENT_INDEXING_FAILED",
                "message": "An attached document is not ready",
                "attachment_id": str(failed_file.id),
            }
            if run.status == "pending":
                run.status = "failed"
                run.error = dict(submission.error)
                run.finished_at = func.clock_timestamp()
                await update_state_and_add_run_event(
                    session, run, "run.failed", {"error": run.error}
                )
        elif attachments and all(
            file.status == "ready" and file.index_generation is not None for file in attachments
        ):
            submission.status = "ready"
            submission.error = None
            if run.status == "pending":
                run.status = "queued"
                run.error = None
                await update_state_and_add_run_event(
                    session,
                    run,
                    "documents.ready",
                    {"document_ids": [str(file.id) for file in attachments]},
                )
                await update_state_and_add_run_event(session, run, "run.queued", {})

    async def recover_expired(self) -> int:
        """Return expired indexing attempts to pending so a worker can rebuild them.

        Returns:
            Number of updated files. Clear owner/expiry; the next claim receives
            a fresh generation, so vectors from the expired attempt stay invisible.
            Do not retry failed documents automatically or change AI execution attempts.
        """
        async with self.sessions.begin() as session:
            recovered = (
                await session.execute(
                    update(Attachment)
                    .where(
                        Attachment.status == "indexing",
                        Attachment.index_expires_at <= func.clock_timestamp(),
                    )
                    .values(status="pending", index_owner=None, index_expires_at=None)
                    .returning(Attachment.id)
                )
            ).all()
            return len(recovered)

    async def release(self, claim: IndexingClaim) -> bool:
        """Return this owned attempt to pending after confirmed local work has stopped.

        Args:
            claim: Attempt being abandoned because of shutdown, lease loss or a busy file lock.

        Returns:
            True if the matching indexing row was reset, False if another attempt
            or deletion already changed it. This does not touch ready/failed rows.
        """
        async with self.sessions.begin() as session:
            released = await session.scalar(
                update(Attachment)
                .where(
                    Attachment.id == claim.attachment_id,
                    Attachment.index_owner == claim.owner,
                    Attachment.index_attempt == claim.attempt,
                    Attachment.status == "indexing",
                )
                .values(status="pending", index_owner=None, index_expires_at=None)
                .returning(Attachment.id)
            )
            return released is not None


async def retry_submission_indexing(
    session: AsyncSession, conversation_id: UUID, submission_id: UUID
) -> Submission:
    """Retry failed files in one send without recreating messages or already-ready vectors.

    Args:
        session: Request session with no open transaction.
        conversation_id: Live conversation that must own the send.
        submission_id: Failed send whose indexing should be retried explicitly.

    Returns:
        Submission reset to accepted with its same user message/run/file IDs.

    Raises:
        ApiError: 404 for missing resources; 409 for a send that is not an indexing
            failure, indexing still in progress, cancellation, or another active run.
    """
    async with session.begin():
        await get_conversation(session, conversation_id, lock=True)
        submission = await session.scalar(
            select(Submission).where(
                Submission.id == submission_id,
                Submission.conversation_id == conversation_id,
            )
        )
        if submission is None:
            raise ApiError(404, "SUBMISSION_NOT_FOUND", "Submission not found")
        attachments = list(
            (
                await session.scalars(
                    select(Attachment)
                    .where(
                        Attachment.submission_id == submission_id,
                    )
                    .with_for_update()
                )
            ).all()
        )
        run = await session.scalar(select(Run).where(Run.submission_id == submission_id))
        if (
            submission.status != "failed"
            or run is None
            or run.status != "failed"
            or run.cancel_requested
        ):
            raise ApiError(
                409, "INDEXING_NOT_FAILED", "Only failed document indexing can be retried"
            )
        if any(file.status == "indexing" for file in attachments):
            raise ApiError(409, "INDEXING_BUSY", "Wait for the remaining files to finish indexing")
        if any(file.deleted_at is not None or file.status == "deleted" for file in attachments):
            raise ApiError(409, "DOCUMENT_DELETED", "A deleted attachment cannot be indexed again")
        failed_files = [file for file in attachments if file.status == "failed"]
        if not failed_files:
            raise ApiError(409, "INDEXING_NOT_FAILED", "No failed documents to retry")
        other_run = await session.scalar(
            select(Run.id)
            .where(
                Run.conversation_id == conversation_id,
                Run.id != run.id,
                Run.status.in_(ACTIVE_RUN_STATUSES),
            )
            .limit(1)
        )
        if other_run is not None:
            raise ApiError(409, "RUN_ACTIVE", "Finish the active run before retrying this send")
        for file in failed_files:
            file.status = "pending"
            file.error = None
            file.chunk_count = 0
        submission.status = "accepted"
        submission.error = None
        run.status = "pending"
        run.error = None
        run.finished_at = None
        await session.flush()
        await update_state_and_add_run_event(
            session, run, "run.retried", {"next_attempt": run.attempt + 1}
        )
        await update_state_and_add_run_event(session, run, "run.pending", {})
        return submission
