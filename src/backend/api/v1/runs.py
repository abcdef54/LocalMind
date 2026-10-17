"""Expose durable run snapshots, replay/SSE and explicit cancel/retry operations."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Header, Query, Request
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

from src.backend.api.errors import ApiError
from src.backend.api.v1.conversations import SessionDependency
from src.backend.api.v1.schemas import RunEventPage, RunRetryInput, RunSnapshot, MessageResponse
from src.backend.execution.replay import read_event_page, stream_events
from src.backend.services import runs

SSE_MAX_CONNECTIONS = 32

router = APIRouter(prefix="/runs", tags=["runs"])
ExpectedConversation = Annotated[
    UUID | None, Query(description="Optional expected chat; mismatched scope returns 404.")
]
AfterSequence = Annotated[int, Query(ge=0)]


@router.get("/{run_id}", response_model=RunSnapshot)
async def get_run_snapshot(
    run_id: UUID, session: SessionDependency, conversation_id: ExpectedConversation = None
) -> RunSnapshot:
    """Read canonical run state without starting or retrying inference.

    Args:
        run_id: Saved task UUID returned by submission acceptance.
        session: Request read session.
        conversation_id: Optional expected live chat; mismatched scope returns 404.

    Returns:
        Status, result URL, error, attempt, settings and visualization at one sequence.
    """
    return RunSnapshot.model_validate(await runs.snapshot(session, conversation_id, run_id))


@router.post("/{run_id}/cancel", status_code=202, response_model=RunSnapshot)
async def cancel_run(
    run_id: UUID, session: SessionDependency, conversation_id: ExpectedConversation = None
) -> RunSnapshot:
    """Request cancellation of run_id in the optional chat scope and return its committed snapshot.

    Args:
        run_id: Saved task to stop; repeated requests preserve terminal outcomes.
        session: Fresh session used to commit cancellation before responding.
        conversation_id: Optional expected live chat scope.

    Returns:
        HTTP 202 snapshot. This does not mean running compute stopped;
    check status and compute_stopped. Repeated requests preserve terminal outcomes.
    """
    await runs.request_cancel(session, conversation_id, run_id)
    return RunSnapshot.model_validate(await runs.snapshot(session, conversation_id, run_id))


@router.post("/{run_id}/retry", status_code=202, response_model=RunSnapshot)
async def retry_run(
    run_id: UUID,
    payload: RunRetryInput,
    session: SessionDependency,
    conversation_id: ExpectedConversation = None,
) -> RunSnapshot:
    """Queue run_id again only if payload.expected_attempt matches and prior compute stopped.

    Args:
        run_id: Failed/cancelled/interrupted task; input and settings are reused.
        payload: expected_attempt from the client's last run snapshot.
        session: Fresh session committing the reset and retry events together.
        conversation_id: Optional expected live chat scope.

    Returns:
        HTTP 202 snapshot after queueing or waiting for file indexing. Wrong scope returns 404;
    active/completed/unsafe/stale retries return 409. No new message or file is created.
    """
    await runs.retry_run(session, conversation_id, run_id, payload.expected_attempt)
    return RunSnapshot.model_validate(await runs.snapshot(session, conversation_id, run_id))


@router.get("/{run_id}/event-log", response_model=RunEventPage)
async def get_event_log(
    run_id: UUID,
    session: SessionDependency,
    conversation_id: ExpectedConversation = None,
    after: AfterSequence = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
) -> RunEventPage:
    """Read saved events for research replay without holding an SSE connection.

    Args:
        run_id: Task whose persisted history is requested.
        session: Request-scoped read session.
        conversation_id: Optional expected live chat scope.
        after: Last applied sequence; return only greater sequence numbers.
        limit: Maximum events to return, from 1 to 100.

    Returns:
        Ascending events with replay cursor and current sequence/terminal metadata.

    Raises:
        ApiError: 410 for expired retention, 409 for a future cursor, 404 for scope failure.
    """
    return await read_event_page(session, conversation_id, run_id, after, limit)


@router.get("/{run_id}/events", response_class=StreamingResponse)
async def subscribe_events(
    request: Request,
    run_id: UUID,
    session: SessionDependency,
    conversation_id: ExpectedConversation = None,
    after: AfterSequence = 0,
    last_event_id: Annotated[str | None, Header(max_length=128)] = None,
) -> StreamingResponse:
    """Stream committed events and resume from after or a scoped Last-Event-ID.

    Args:
        request: Client connection; disconnect only stops this subscription.
        run_id: Run whose persisted events are streamed, including completed runs.
        session: Read session, released between each bounded poll.
        conversation_id: Optional expected chat scope.
        after: Last applied sequence, initially zero or snapshot.last_event_sequence.
        last_event_id: '<run UUID>:<sequence>'; takes precedence over after on automatic reconnect.

    Returns:
        SSE response with stable IDs, heartbeats, no-store and proxy buffering disabled.
        At most 32 simultaneous subscribers per API process are accepted.

    Raises:
        ApiError: 422 for malformed/wrong-run IDs, 410 for expired replay, 409 for
            future cursors or 429 for subscriber capacity. No inference is started.
    """
    if last_event_id is not None:
        try:
            identity, sequence = last_event_id.rsplit(":", 1)
            resumed = int(sequence)
            if UUID(identity) != run_id or resumed < 0:
                raise ValueError("Cursor mismatch")
            after = resumed
        except (ValueError, TypeError):
            raise ApiError(422, "INVALID_EVENT_CURSOR", "Invalid Last-Event-ID") from None
    await read_event_page(session, conversation_id, run_id, after, 1)
    await session.rollback()
    active = getattr(request.app.state, "run_stream_count", 0)
    if active >= SSE_MAX_CONNECTIONS:
        raise ApiError(
            429, "STREAM_CAPACITY", "Too many event subscriptions", headers={"Retry-After": "1"}
        )
    request.app.state.run_stream_count = active + 1
    released = False

    def release_slot():
        """Release once, including when disconnect occurs before the iterator starts."""
        nonlocal released
        if not released:
            request.app.state.run_stream_count -= 1
            released = True

    return StreamingResponse(
        stream_events(request, session, conversation_id, run_id, after, release_slot),
        media_type="text/event-stream",
        background=BackgroundTask(release_slot),
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.get("/{run_id}/result", response_model=MessageResponse)
async def get_run_result(
    run_id: UUID, session: SessionDependency, conversation_id: ExpectedConversation = None
) -> MessageResponse:
    """Read a saved final answer directly, without fetching history pages.

    Args:
        run_id: Completed task whose result message is requested.
        session: Read session; no generation or retry is performed.
        conversation_id: Optional expected live chat scope.

    Returns:
        MessageResponse with answer, saved settings, metadata and scoped citations.

    Raises:
        ApiError: 409 until a completed result exists, 404 for unavailable/wrong-scope resources.
    """
    return await runs.get_result(session, conversation_id, run_id)
