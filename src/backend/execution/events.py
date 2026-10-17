"""Commit ordered events and their compact visualization state in one database transaction."""

import asyncio
from copy import deepcopy
import json
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.backend.database.models import Conversation, Run, RunEvent
from src.backend.execution.control import RunCancellationRequested

EVENT_RETENTION = 100
EVENT_TYPES = frozenset(
    {
        "run.pending",
        "run.queued",
        "run.started",
        "run.completed",
        "run.failed",
        "run.cancel_requested",
        "run.cancelled",
        "run.interrupted",
        "run.retried",
        "run.compute_stopped",
        "documents.indexing",
        "documents.ready",
        "documents.failed",
        "generation.started",
        "candidate.completed",
        "generation.completed",
        "tournament.started",
        "round.started",
        "group.started",
        "group.completed",
        "candidate.bye",
        "ranking.fallback",
        "round.completed",
        "tournament.completed",
    }
)


def build_updated_visualization_state_from_event(
    projection: dict, event_type: str, payload: dict
) -> dict:
    """Return a new compact visualization state without mutating the stored JSON in place.

    Args:
        projection: Current state of this run attempt; contains no answer/tool text.
        event_type: Validated lifecycle or graph event name.
        payload: Candidate identities, brackets, winners and measured counters.

    Returns:
        Updated candidate/group/survivor state. Retry resets it; failed/cancelled/
        interrupted runs do not expose a selected candidate as a final answer winner.
    """
    state = deepcopy(projection)
    if event_type == "run.retried":
        return {}
    if event_type == "generation.started":
        state = {
            "phase": "generating",
            "expected_candidates": payload["candidate_count"],
            "completed_candidates": 0,
            "ranking_group_size": payload["ranking_group_size"],
            "candidates": {
                item["candidate_id"]: {
                    "display_index": item["display_index"],
                    "status": "generating",
                }
                for item in payload["candidates"]
            },
            "rounds": {},
            "groups": {},
            "survivors": [],
            "winner_id": None,
        }
    elif event_type == "generation.completed":
        state["phase"] = "generation_completed"
    elif event_type == "candidate.completed":
        candidate = state["candidates"][payload["candidate_id"]]
        if candidate["status"] == "generating":
            state["completed_candidates"] += 1
        candidate["status"] = "ready"
    elif event_type == "tournament.started":
        state["phase"] = "ranking"
        state["seeding"] = payload["candidate_ids"]
        state["survivors"] = payload["candidate_ids"]
    elif event_type == "round.started":
        state.setdefault("rounds", {})[payload["round_id"]] = deepcopy(payload)
    elif event_type == "group.started":
        state.setdefault("groups", {})[payload["group_id"]] = {**payload, "status": "ranking"}
    elif event_type == "group.completed":
        state.setdefault("groups", {})[payload["group_id"]] = {**payload, "status": "completed"}
        for identity in payload["eliminated_ids"]:
            state["candidates"][identity]["status"] = "eliminated"
        state["candidates"][payload["winner_id"]]["status"] = "advanced"
    elif event_type == "candidate.bye":
        state["candidates"][payload["candidate_id"]]["status"] = "bye"
        state.setdefault("groups", {})[payload["group_id"]] = {**payload, "status": "bye"}
    elif event_type == "round.completed":
        state["survivors"] = payload["survivor_ids"]
        state["rounds"][payload["round_id"]]["status"] = "completed"
        state["rounds"][payload["round_id"]]["survivor_ids"] = payload["survivor_ids"]
    elif event_type == "tournament.completed":
        state["phase"] = "selected"
        state["winner_id"] = payload["winner_id"]
        state["survivors"] = [payload["winner_id"]]
        state["candidates"][payload["winner_id"]]["status"] = "selected"
        state["tournament"] = deepcopy(payload)
    elif event_type in (
        "run.queued",
        "run.pending",
        "run.started",
        "run.cancel_requested",
        "run.completed",
    ):
        state["phase"] = {"run.pending": "indexing", "run.started": "preparing"}.get(
            event_type, event_type.removeprefix("run.")
        )
    elif event_type in ("run.failed", "run.cancelled", "run.interrupted"):
        state["phase"] = event_type.removeprefix("run.")
        state["winner_id"] = None
        for candidate in state.get("candidates", {}).values():
            if candidate["status"] != "eliminated":
                candidate["status"] = event_type.removeprefix("run.")
    return state


async def update_state_and_add_run_event(
    session: AsyncSession, run: Run, event_type: str, payload: dict[str, Any] | None = None
) -> None:
    """Update the visualization state and add an event under the caller's locked run row.

    Args:
        session: Existing transaction; caller must hold this run's row lock or own
            its uncommitted insert. This function flushes but never commits it.
        run: Mutable ORM run whose sequence/projection change with the event.
        event_type: One supported lifecycle or graph event type.
        payload: JSON object of bounded public metadata, default empty.

    Sequence never resets across retries. Keep the newest 1000 events per run;
    the latest projection survives retention. Failure rolls back state and event
    together. Full candidate answers, tool transcripts and model thoughts are omitted.
    """
    payload = payload or {}
    if event_type not in EVENT_TYPES or len(json.dumps(payload, allow_nan=False).encode()) > 16384:
        raise ValueError("Unsupported or oversized run event")
    run.event_sequence = (run.event_sequence or 0) + 1
    run.visualization = build_updated_visualization_state_from_event(
        run.visualization or {}, event_type, payload
    )
    session.add(
        RunEvent(
            run_id=run.id,
            conversation_id=run.conversation_id,
            sequence=run.event_sequence,
            attempt=run.attempt or 0,
            type=event_type,
            payload=payload,
        )
    )
    await session.flush()
    if run.event_sequence > EVENT_RETENTION:
        await session.execute(
            delete(RunEvent).where(
                RunEvent.run_id == run.id, RunEvent.sequence <= run.event_sequence - EVENT_RETENTION
            )
        )


async def emit_graph_event(config: dict, event_type: str, payload: dict) -> None:
    """Forward graph metadata to the injected progress recorder; benchmarks without a recorder do nothing.

    Args:
        config: Graph execution config containing an optional progress_recorder.
        event_type: Supported candidate/tournament event type.
        payload: Bounded public identities and metrics, never raw generated text.
    """
    progress_recorder = config.get("configurable", {}).get("progress_recorder")
    if progress_recorder is not None:
        await progress_recorder.check_run_ownership_and_save_progress_event(event_type, payload)


class RunProgressRecorder:
    """Persist progress only while this run attempt still owns a live reservation.

    Args:
        sessions: Factory for short DB transactions, independent of inference.
        claim: RunClaim identifying this exact owner and attempt.
        timeout_seconds: Maximum wait for each ownership check or event database write.

    Client connections are not involved in writes. Concurrent candidate completions
    serialize on the run row, producing one gap-free sequence and atomic snapshot.
    """

    def __init__(self, sessions: async_sessionmaker, claim, timeout_seconds: float = 5) -> None:
        self.sessions = sessions
        self.claim = claim
        self.timeout_seconds = timeout_seconds

    async def _owned_run(self, session: AsyncSession, *, lock: bool = False) -> Run:
        """Return the current run or reject lost ownership/cancellation before more work.

        Args:
            session: Short read/write session used to check ownership or save a progress event.
            lock: Whether to serialize a progress write against cancel/completion.

        Returns:
            Owned unexpired running run; never an attempt replaced by retry.
        """
        from src.backend.execution.store import RunStore

        query = (
            select(Run)
            .join(Conversation, Conversation.id == Run.conversation_id)
            .where(RunStore._owned_lease(self.claim), Conversation.deleted_at.is_(None))
        )
        if lock:
            query = query.with_for_update(of=Run)
        run = await session.scalar(query)
        if run is None:
            raise ValueError("Run reservation lost")
        if run.cancel_requested or run.status == "cancel_requested":
            raise RunCancellationRequested()
        return run

    async def check_run_ownership_and_cancellation(self) -> None:
        """Check cancellation between external calls; never claim that an in-flight HTTP request stopped."""

        async def check():
            """Read current ownership without retaining a database session during inference."""
            async with self.sessions() as session:
                await self._owned_run(session)

        await asyncio.wait_for(check(), self.timeout_seconds)

    async def check_run_ownership_and_save_progress_event(
        self, event_type: str, payload: dict
    ) -> None:
        """Commit one ordered graph event after checking ownership and cancellation.

        Args:
            event_type: Supported generation/tournament event type.
            payload: Small visualization metadata object without answer/tool contents.

        Database failures propagate; missing telemetry is never presented as a
        complete research trace. This does not wait for any SSE reader or animation.
        """

        async def write():
            """Lock the run and save event/projection in the same short transaction."""
            async with self.sessions() as session, session.begin():
                run = await self._owned_run(session, lock=True)
                await update_state_and_add_run_event(session, run, event_type, payload)

        await asyncio.wait_for(write(), self.timeout_seconds)
