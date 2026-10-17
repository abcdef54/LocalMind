"""Read bounded persisted event pages and stream them without involving inference tasks."""

import asyncio
import json
import time
from uuid import UUID

from fastapi import Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from contextlib import suppress

from src.backend.api.errors import ApiError
from src.backend.api.v1.schemas import RunEventPage, RunEventResponse
from src.backend.database.models import RunEvent
from src.backend.services.runs import TERMINAL_STATUSES, get_run

EVENT_PAGE_SIZE = 100
SSE_POLL_SECONDS = 0.5
SSE_HEARTBEAT_SECONDS = 15


async def read_event_page(
    session: AsyncSession, conversation_id: UUID | None, run_id: UUID, after: int, limit: int
) -> RunEventPage:
    """Read retained events strictly after a scoped sequence, in ascending order.

    Args:
        session: Short read session; caller closes its transaction before waiting.
        conversation_id: Optional expected chat, checked against the live run.
        run_id: Run to replay; sequences never reset across retry attempts.
        after: Last sequence the client already applied, initially zero.
        limit: Validated page size, at most 100 events.

    Returns:
        Versioned events, replay cursor and terminal flag from the same run snapshot.

    Raises:
        ApiError: 404 for wrong scope, 409 for a future cursor, or 410 if retention
            removed needed events. Refetch the run snapshot before reconnecting.
    """
    run = await get_run(session, conversation_id, run_id)
    last = run.event_sequence
    first = await session.scalar(
        select(func.min(RunEvent.sequence)).where(RunEvent.run_id == run_id)
    )
    if after > last:
        raise ApiError(409, "EVENT_CURSOR_AHEAD", "Event cursor is newer than this run")
    if first is not None and after < first - 1:
        raise ApiError(
            410,
            "EVENT_HISTORY_EXPIRED",
            "Refetch the run snapshot before replaying",
            details={"snapshot_url": f"/api/v1/runs/{run_id}", "last_event_sequence": last},
        )
    rows = list(
        (
            await session.scalars(
                select(RunEvent)
                .where(
                    RunEvent.run_id == run_id, RunEvent.sequence > after, RunEvent.sequence <= last
                )
                .order_by(RunEvent.sequence)
                .limit(limit)
            )
        ).all()
    )
    items = [
        RunEventResponse(
            id=f"{row.run_id}:{row.sequence}",
            sequence=row.sequence,
            run_id=row.run_id,
            conversation_id=row.conversation_id,
            attempt=row.attempt,
            type=row.type,
            timestamp=row.created_at,
            payload=row.payload,
        )
        for row in rows
    ]
    next_after = items[-1].sequence if items else after
    return RunEventPage(
        items=items,
        next_after=next_after,
        last_event_sequence=last,
        first_event_sequence=first,
        has_more=next_after < last,
        terminal=run.status in TERMINAL_STATUSES,
    )


async def stream_events(
    request: Request,
    session: AsyncSession,
    conversation_id: UUID | None,
    run_id: UUID,
    after: int,
    release_slot,
):
    """Yield persisted SSE frames, closing only this subscription on disconnect.

    Args:
        request: Used to detect browser disconnect; never cancels the run itself.
        session: Request read session; rollback releases DB connections between polls.
        conversation_id: Optional expected chat scope.
        run_id: Run whose events are streamed.
        after: Last event sequence already applied by this client.
        release_slot: Idempotent callback releasing the per-process subscriber slot.

    Slow consumers block only their own response. Every poll reads at most 100
    events; no unbounded viewer queue exists. Falling behind retention emits
    stream.reset and closes. Heartbeats have no event ID and change no run state.
    """
    heartbeat_at = time.monotonic()
    try:
        while not await request.is_disconnected():
            try:
                page = await read_event_page(
                    session, conversation_id, run_id, after, EVENT_PAGE_SIZE
                )
            except ApiError as error:
                await session.rollback()
                yield "event: stream.reset\ndata: " + json.dumps(
                    {"code": error.code, "snapshot_url": f"/api/v1/runs/{run_id}"}
                ) + "\n\n"
                return
            except SQLAlchemyError:
                with suppress(SQLAlchemyError):
                    await session.rollback()
                yield 'event: stream.reset\ndata: {"code":"EVENT_STREAM_UNAVAILABLE"}\n\n'
                return
            await session.rollback()
            for event in page.items:
                yield f"id: {event.id}\nevent: {event.type}\ndata: {event.model_dump_json()}\n\n"
                after = event.sequence
                heartbeat_at = time.monotonic()
            if page.terminal and after >= page.last_event_sequence:
                return
            if page.has_more:
                continue
            if time.monotonic() - heartbeat_at >= SSE_HEARTBEAT_SECONDS:
                yield ": heartbeat\n\n"
                heartbeat_at = time.monotonic()
            await asyncio.sleep(SSE_POLL_SECONDS)
    finally:
        release_slot()
