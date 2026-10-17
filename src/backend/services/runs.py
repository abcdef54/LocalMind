"""Read a run snapshot and accept scoped cancellation or explicit retry requests."""

from uuid import UUID

from sqlalchemy import func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, noload

from src.backend.api.errors import ApiError
from src.backend.api.v1.schemas import MessageResponse, CitationResponse
from src.backend.database.models import (
    ACTIVE_RUN_STATUSES,
    Attachment,
    Message,
    Run,
    RunEvent,
    Submission,
)
from src.backend.execution.events import update_state_and_add_run_event
from src.backend.services.conversation_history import get_conversation

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted"})


async def get_run(
    session: AsyncSession, conversation_id: UUID | None, run_id: UUID, *, lock: bool = False
) -> Run:
    """Read a run only within its live conversation, optionally locking conversation then run.

    Args:
        session: Request transaction; caller owns commits.
        conversation_id: Live chat UUID from the server route.
        run_id: Run UUID that must belong to this chat.
        lock: Serialize lifecycle changes against send/completion/deletion.

    Returns:
        Run row; wrong-chat and missing IDs are indistinguishable HTTP 404 errors.
    """
    if conversation_id is None:
        conversation_id = await session.scalar(select(Run.conversation_id).where(Run.id == run_id))
        if conversation_id is None:
            raise ApiError(404, "RUN_NOT_FOUND", "Run not found")
    await get_conversation(session, conversation_id, lock=lock)
    query = select(Run).where(Run.id == run_id, Run.conversation_id == conversation_id)
    if lock:
        query = query.with_for_update()
    run = await session.scalar(query)
    if run is None:
        raise ApiError(404, "RUN_NOT_FOUND", "Run not found")
    return run


async def snapshot(session: AsyncSession, conversation_id: UUID | None, run_id: UUID) -> dict:
    """Return canonical run state and the visualization at last_event_sequence.

    Args:
        session: Request read session.
        conversation_id: Chat that must own the run.
        run_id: Run to inspect, including terminal runs with no event history.

    Returns:
        Public IDs/settings/status/error, attempt, compute confirmation, projection
        and sequence. Document processing is read separately from their current rows.
    """
    run = await get_run(session, conversation_id, run_id)
    documents = (
        await session.execute(
            select(Attachment.id, Attachment.status, Attachment.error).where(
                Attachment.submission_id == run.submission_id
            )
        )
    ).all()
    first = await session.scalar(
        select(func.min(RunEvent.sequence)).where(RunEvent.run_id == run.id)
    )
    return {
        "id": run.id,
        "conversation_id": run.conversation_id,
        "submission_id": run.submission_id,
        "message_id": run.message_id,
        "result_message_id": run.result_message_id,
        "result_url": f"/api/v1/runs/{run.id}/result" if run.result_message_id else None,
        "status": run.status,
        "attempt": run.attempt,
        "settings_snapshot": run.settings_snapshot,
        "cancel_requested": run.cancel_requested,
        "compute_stopped": run.lease_owner is None
        and run.status in (TERMINAL_STATUSES | {"pending", "queued"}),
        "error": run.error,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "last_event_sequence": run.event_sequence,
        "first_event_sequence": first,
        "visualization": run.visualization,
        "documents": [
            {"id": row.id, "status": row.status, "error": row.error} for row in documents
        ],
    }


async def request_cancel(session: AsyncSession, conversation_id: UUID | None, run_id: UUID) -> None:
    """Record cancellation; finish immediately only when inference has not been claimed.

    Args:
        session: Fresh request session; commits before returning.
        conversation_id: Live chat owning the task.
        run_id: Task to cancel. Repeating cancellation is an idempotent no-op.

    Completed/failed runs keep their outcome. Running work remains cancel_requested
    until cooperative checkpoints finish all siblings. An interrupted slot is never
    released by this endpoint because it cannot verify external compute stoppage.
    """
    async with session.begin():
        run = await get_run(session, conversation_id, run_id, lock=True)
        if run.status in TERMINAL_STATUSES or run.status == "cancel_requested":
            return
        run.cancel_requested = True
        if run.status in ("pending", "queued") and run.lease_owner is None:
            run.status = "cancelled"
            run.finished_at = func.clock_timestamp()
            submission = await session.get(Submission, run.submission_id)
            submission.status = "cancelled"
            await update_state_and_add_run_event(
                session, run, "run.cancelled", {"compute_stopped": True}
            )
        else:
            run.status = "cancel_requested"
            await update_state_and_add_run_event(
                session, run, "run.cancel_requested", {"compute_stopped": False}
            )
        run.updated_at = func.clock_timestamp()


async def retry_run(
    session: AsyncSession, conversation_id: UUID | None, run_id: UUID, expected_attempt: int
) -> None:
    """Explicitly retry the same saved input after confirming its previous compute stopped.

    Args:
        session: Fresh request session; locks conversation/run before changing state.
        conversation_id: Chat owning the failed/cancelled/interrupted run.
        run_id: Existing task; IDs/settings/input stay unchanged.
        expected_attempt: Attempt shown by the client snapshot, guarding stale retry.

    Raises:
        ApiError: 409 for active/completed runs, unknown compute, stale attempt,
            deleted files, indexing still active, another active run, or a newer
            question that superseded this input.

    Keep sequence monotonic; reset visualization and queue again, or wait for pending
    files. Never retry automatically or clear an unconfirmed interrupted owner.
    """
    async with session.begin():
        run = await get_run(session, conversation_id, run_id, lock=True)
        conversation_id = run.conversation_id
        if run.attempt != expected_attempt:
            raise ApiError(409, "STALE_RUN_ATTEMPT", "Refresh the run before retrying")
        if run.lease_owner is not None:
            raise ApiError(
                409, "COMPUTE_UNCONFIRMED", "Verify previous compute stopped before retrying"
            )
        if (
            run.status not in ("failed", "cancelled", "interrupted")
            or run.result_message_id is not None
        ):
            raise ApiError(409, "RUN_NOT_RETRYABLE", "This run cannot be retried")
        active = await session.scalar(
            select(Run.id)
            .where(
                Run.conversation_id == conversation_id,
                Run.id != run.id,
                Run.status.in_(ACTIVE_RUN_STATUSES),
            )
            .limit(1)
        )
        if active is not None:
            raise ApiError(409, "RUN_ACTIVE", "Another run is active in this conversation")
        input_position = (
            await session.execute(
                select(Message.created_at, Message.id).where(Message.id == run.message_id)
            )
        ).one()
        newer_input = await session.scalar(
            select(Message.id)
            .where(
                Message.conversation_id == conversation_id,
                Message.role == "user",
                Message.deleted_at.is_(None),
                tuple_(Message.created_at, Message.id)
                > (input_position.created_at, input_position.id),
            )
            .limit(1)
        )
        if newer_input is not None:
            raise ApiError(
                409, "RUN_INPUT_SUPERSEDED", "Retry the latest send or submit a new message"
            )
        files = list(
            (
                await session.scalars(
                    select(Attachment)
                    .where(Attachment.submission_id == run.submission_id)
                    .with_for_update()
                )
            ).all()
        )
        if any(file.deleted_at is not None for file in files):
            raise ApiError(409, "DOCUMENT_DELETED", "A document for this send was deleted")
        if any(file.status == "indexing" for file in files):
            raise ApiError(
                409, "INDEXING_ACTIVE", "Wait for the previous indexing operation to stop"
            )
        for file in files:
            if file.status != "ready" or file.index_generation is None:
                file.status = "pending"
                file.error = None
                file.chunk_count = 0
                file.index_owner = None
                file.index_expires_at = None
        ready = all(file.status == "ready" and file.index_generation is not None for file in files)
        run.status = "queued" if ready else "pending"
        run.cancel_requested = False
        run.lease_expires_at = None
        run.started_at = None
        run.finished_at = None
        run.error = None
        submission = await session.get(Submission, run.submission_id)
        submission.status = "ready" if ready else "accepted"
        submission.error = None
        await update_state_and_add_run_event(
            session, run, "run.retried", {"next_attempt": run.attempt + 1}
        )
        await update_state_and_add_run_event(
            session, run, "run.queued" if ready else "run.pending", {}
        )


async def get_result(
    session: AsyncSession, conversation_id: UUID | None, run_id: UUID
) -> MessageResponse:
    """Read one saved final answer without loading conversation history pages.

    Args:
        session: Request database session used only for scoped reads.
        conversation_id: Optional expected chat, checked against the live run.
        run_id: Completed task whose result message is requested.

    Returns:
        Public assistant message with settings, metadata and same-chat citations.
        Deleted citations remain identifiable; document disk paths are never returned.

    Raises:
        ApiError: 409 if no completed result exists, 404 for unavailable/wrong-scope resources.
    """

    run = await get_run(session, conversation_id, run_id)
    if run.status != "completed" or run.result_message_id is None:
        raise ApiError(409, "RUN_RESULT_NOT_READY", "This run has no completed answer")
    message = await session.scalar(
        select(Message)
        .where(
            Message.id == run.result_message_id,
            Message.run_id == run.id,
            Message.conversation_id == run.conversation_id,
            Message.deleted_at.is_(None),
        )
        .options(
            selectinload(Message.citations.and_(Attachment.conversation_id == run.conversation_id)),
            noload(Message.attachments),
        )
    )
    if message is None:
        raise ApiError(404, "RUN_RESULT_NOT_FOUND", "The saved answer is unavailable")
    return MessageResponse(
        id=message.id,
        conversation_id=message.conversation_id,
        content=message.content,
        role=message.role,
        personality=message.personality,
        model_name=message.model_name,
        reasoning_mode=message.reasoning_mode,
        temperature=message.temperature,
        created_at=message.created_at,
        submission_id=message.submission_id,
        run_id=message.run_id,
        generation_metadata=message.generation_metadata,
        citations=[
            CitationResponse(
                id=file.id,
                file_name=file.file_name,
                file_type=file.file_type,
                file_size=file.file_size,
                deleted=file.deleted_at is not None,
            )
            for file in message.citations
        ],
    )
