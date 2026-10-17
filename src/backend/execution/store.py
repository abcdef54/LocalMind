"""Read and update AI run records in PostgreSQL.

A run is the work needed to produce one assistant reply. A worker must reserve
that run before starting it. The reservation has an owner token, attempt number
and expiry time; these are checked again before saving an answer. Database updates
use short transactions, so model calls never keep a database lock open.
"""

from dataclasses import dataclass, field
from datetime import timedelta
import json
from typing import Any, Literal
from uuid import UUID, uuid4

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.backend.database.models import Attachment, Conversation, Message, Run, Submission
from src.backend.execution.events import update_state_and_add_run_event

ADMISSION_LOCK_KEY = 1280134734
LEASED_STATUSES = ("running", "cancel_requested")


@dataclass(frozen=True)
class RunClaim:
    """Store the IDs and permission token for one reserved AI run.

    run_id identifies the work, message_id identifies the user message, and
    settings_snapshot contains the settings saved when Send was pressed. lease_owner
    and attempt identify this specific execution. Pass this object back to RunStore
    when extending the reservation or saving a result; knowing run_id alone is not enough.
    """

    run_id: UUID
    conversation_id: UUID
    submission_id: UUID
    message_id: UUID
    settings_snapshot: dict[str, Any]
    lease_owner: str
    attempt: int


@dataclass(frozen=True)
class RunResult:
    """Carry the assistant text returned by a run handler.

    Args:
        content: Nonblank answer text; original whitespace is preserved.
        citation_ids: Document UUIDs used by the winner. Completion rechecks their
            conversation, deletion and readiness before linking them to the answer.
        generation_metadata: Small JSON object containing tournament counts, winner
            identity and timing, never full prompts, candidate answers or file paths.
    """

    content: str
    citation_ids: tuple[UUID, ...] = ()
    generation_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject blank answers, invalid citation UUIDs and oversized/non-JSON metadata.

        Copy metadata so later caller mutations cannot change the result to be saved.
        Deduplicate citations while preserving their first occurrence.
        """
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("RunResult requires non-empty assistant content")
        if any(not isinstance(identity, UUID) for identity in self.citation_ids):
            raise ValueError("Citation IDs must be UUIDs")
        if not isinstance(self.generation_metadata, dict):
            raise ValueError("Generation metadata must be a JSON object")
        encoded = json.dumps(self.generation_metadata, allow_nan=False)
        if len(encoded.encode("utf-8")) > 8192:
            raise ValueError("Generation metadata exceeds 8 KiB")
        object.__setattr__(self, "citation_ids", tuple(dict.fromkeys(self.citation_ids)))
        object.__setattr__(self, "generation_metadata", json.loads(encoded))


class RunStore:
    """Save run status and assistant replies through short PostgreSQL transactions.

    sessions is the async session factory used for every operation. Methods commit
    successful changes when their transaction exits and propagate database errors.
    They do not call the model or determine whether external compute has stopped.
    """

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        """Save the session factory; do not open a database connection yet."""
        self.sessions = sessions

    async def reserve_next_queued_run_for_worker(
        self, *, worker_id: str, lease_seconds: float, max_concurrency: int
    ) -> RunClaim | None:
        """Reserve the oldest eligible queued run for this worker.

        worker_id names the worker; lease_seconds is how long its reservation lasts;
        max_concurrency limits occupied slots across executors sharing this database
        and the same limit. Only ready submissions in live conversations are eligible.

        Return a RunClaim after setting running, recording a new owner token and
        incrementing attempt. Return None if another worker is reserving work, the
        limit is reached, or no eligible row is available. Locked run rows are skipped.
        Interrupted runs still occupy a slot until compute stoppage is confirmed.
        Raise ValueError for invalid worker ID or nonpositive limits.
        """
        if not worker_id or len(worker_id) > 64:
            raise ValueError("worker_id must have 1–64 characters")
        if lease_seconds <= 0 or max_concurrency < 1:
            raise ValueError("Lease and concurrency limits must be positive")
        async with self.sessions.begin() as session:
            acquired = await session.scalar(
                select(func.pg_try_advisory_xact_lock(ADMISSION_LOCK_KEY))
            )
            if not acquired:
                return None
            occupied = await session.scalar(
                select(func.count())
                .select_from(Run)
                .where(or_(Run.lease_owner.is_not(None), Run.status.in_(LEASED_STATUSES)))
            )
            if occupied >= max_concurrency:
                return None
            run = await session.scalar(
                select(Run)
                .join(Submission, Submission.id == Run.submission_id)
                .join(Conversation, Conversation.id == Run.conversation_id)
                .where(
                    Run.status == "queued",
                    Run.cancel_requested.is_(False),
                    Run.lease_owner.is_(None),
                    Submission.status == "ready",
                    Conversation.deleted_at.is_(None),
                )
                .order_by(Run.created_at, Run.id)
                .limit(1)
                .with_for_update(of=Run, skip_locked=True)
            )
            if run is None:
                return None
            owner = f"{worker_id}:{uuid4().hex}"
            run.status = "running"
            run.lease_owner = owner
            run.lease_expires_at = func.clock_timestamp() + timedelta(seconds=lease_seconds)
            run.started_at = func.clock_timestamp()
            run.updated_at = func.clock_timestamp()
            run.attempt += 1
            await session.flush()
            await update_state_and_add_run_event(
                session, run, "run.started", {"attempt": run.attempt}
            )
            return RunClaim(
                run_id=run.id,
                conversation_id=run.conversation_id,
                submission_id=run.submission_id,
                message_id=run.message_id,
                settings_snapshot=dict(run.settings_snapshot),
                lease_owner=owner,
                attempt=run.attempt,
            )

    @staticmethod
    def _owned_lease(claim: RunClaim):
        """Build the SQL conditions that identify a claim still allowed to update its run.

        The run ID, owner token and attempt must match. The run must still be running
        or cancel_requested, and its expiry must be later than PostgreSQL's current time.
        Return a SQL expression; this function does not query or change the database.
        """
        return and_(
            Run.id == claim.run_id,
            Run.lease_owner == claim.lease_owner,
            Run.attempt == claim.attempt,
            Run.status.in_(LEASED_STATUSES),
            Run.lease_expires_at > func.clock_timestamp(),
        )

    async def extend_run_reservation_and_get_cancellation_status(
        self, claim: RunClaim, *, lease_seconds: float
    ) -> Literal["active", "cancel_requested", "lost"]:
        """Extend the claim's reservation and report whether the worker should continue.

        claim identifies the current execution; lease_seconds sets the new duration
        from PostgreSQL's current time. Return active after a valid renewal,
        cancel_requested if cancellation was recorded, or lost if ownership/status/expiry
        no longer matches. Expired reservations are never renewed. Changes are committed
        before return; the method itself does not cancel model computation.
        """
        async with self.sessions.begin() as session:
            heartbeat = (
                await session.execute(
                    update(Run)
                    .where(self._owned_lease(claim))
                    .values(
                        lease_expires_at=func.clock_timestamp() + timedelta(seconds=lease_seconds),
                        updated_at=func.clock_timestamp(),
                    )
                    .returning(Run.status, Run.cancel_requested)
                )
            ).first()
            if heartbeat is None:
                return "lost"
            if heartbeat.status == "cancel_requested" or heartbeat.cancel_requested:
                return "cancel_requested"
            return "active"

    async def save_assistant_reply_and_mark_run_completed(
        self, claim: RunClaim, result: RunResult
    ) -> bool:
        """Save one assistant reply and mark its run completed in the same transaction.

        claim identifies the execution; result contains answer text, citation UUIDs
        and compact generation metadata. Recheck ready citations in this chat and
        save the original settings on the assistant message. Save only
        while the claim is unexpired, the run is running, and cancellation is not set.
        Return True after inserting the assistant message, linking it to the run and
        releasing the execution slot. Return False without saving an answer when those
        checks fail. A late or repeated handler result cannot create another reply.
        """
        async with self.sessions.begin() as session:
            # Conversation-first order matches document deletion and submission
            # acceptance, preventing lock inversion when citations are rechecked.
            conversation = await session.scalar(
                select(Conversation)
                .where(Conversation.id == claim.conversation_id, Conversation.deleted_at.is_(None))
                .with_for_update()
            )
            if conversation is None:
                return False
            run = await session.scalar(
                select(Run)
                .where(
                    self._owned_lease(claim),
                    Run.status == "running",
                    Run.cancel_requested.is_(False),
                    Run.conversation_id == claim.conversation_id,
                    Run.submission_id == claim.submission_id,
                    Run.message_id == claim.message_id,
                )
                .with_for_update()
            )
            if run is None:
                return False
            citations = []
            if result.citation_ids:
                citations = list(
                    (
                        await session.scalars(
                            select(Attachment)
                            .where(
                                Attachment.id.in_(result.citation_ids),
                                Attachment.conversation_id == run.conversation_id,
                                Attachment.deleted_at.is_(None),
                                Attachment.status == "ready",
                                Attachment.index_generation.is_not(None),
                            )
                            .order_by(Attachment.id)
                            .with_for_update()
                        )
                    ).all()
                )
            settings = run.settings_snapshot
            message = Message(
                conversation_id=run.conversation_id,
                run_id=run.id,
                content=result.content,
                role="assistant",
                citations=citations,
                model_name=settings.get("model_name"),
                personality=settings.get("personality"),
                temperature=settings.get("temperature"),
                reasoning_mode=settings.get("reasoning_mode"),
                generation_metadata=result.generation_metadata,
            )
            session.add(message)
            await session.flush()
            # A row lock cannot freeze time. Check the lease again after citation
            # reads/insertion; rolling back also removes the uncommitted message.
            still_owned = await session.scalar(
                select(Run.id).where(
                    self._owned_lease(claim),
                    Run.cancel_requested.is_(False),
                    Run.status == "running",
                )
            )
            if still_owned is None:
                await session.rollback()
                return False
            conversation.message_count += 1
            conversation.unsummarized_message_count += 1
            run.result_message_id = message.id
            run.status = "completed"
            run.finished_at = func.clock_timestamp()
            run.updated_at = func.clock_timestamp()
            run.lease_owner = None
            run.lease_expires_at = None
            await update_state_and_add_run_event(
                session,
                run,
                "run.completed",
                {
                    "result_message_id": str(message.id),
                    "result_url": f"/api/v1/runs/{run.id}/result",
                    "winner_id": result.generation_metadata.get("winner_candidate_id"),
                },
            )
            return True

    async def fail(self, claim: RunClaim) -> bool:
        """Mark the claim's run failed and release its slot after a confirmed handler failure.

        Call only when the handler has stopped its compute. Return True if the claim
        still owns an unexpired running run with no cancellation request; otherwise
        return False and leave it unchanged. Save a generic HANDLER_FAILED error,
        never the raw exception or user content. Do not queue an automatic retry.
        """
        async with self.sessions.begin() as session:
            failed = await session.scalar(
                update(Run)
                .where(
                    self._owned_lease(claim),
                    Run.status == "running",
                    Run.cancel_requested.is_(False),
                )
                .values(
                    status="failed",
                    error={"code": "HANDLER_FAILED", "message": "Run handler failed"},
                    finished_at=func.clock_timestamp(),
                    updated_at=func.clock_timestamp(),
                    lease_owner=None,
                    lease_expires_at=None,
                )
                .returning(Run.id)
            )
            if failed is not None:
                run = await session.get(Run, failed)
                await update_state_and_add_run_event(
                    session, run, "run.failed", {"error": run.error}
                )
            return failed is not None

    async def interrupt(self, claim: RunClaim, *, code: str) -> bool:
        """Mark this execution interrupted while keeping its slot occupied.

        claim supplies the owner token and attempt; code is the public error code to
        record. Return True if a matching running/cancel_requested run was updated,
        or False if it has already changed. An expired claim can still be interrupted.

        This stops later results from being saved. It does not stop the model or prove
        compute ended, so the owner token and expiry are deliberately kept.
        """
        async with self.sessions.begin() as session:
            interrupted = await session.scalar(
                update(Run)
                .where(
                    Run.id == claim.run_id,
                    Run.lease_owner == claim.lease_owner,
                    Run.attempt == claim.attempt,
                    Run.status.in_(LEASED_STATUSES),
                )
                .values(
                    status="interrupted",
                    error={
                        "code": code,
                        "message": "Execution interrupted; verify compute stopped before releasing its slot",
                    },
                    finished_at=func.clock_timestamp(),
                    updated_at=func.clock_timestamp(),
                )
                .returning(Run.id)
            )
            if interrupted is not None:
                run = await session.get(Run, interrupted)
                await update_state_and_add_run_event(
                    session, run, "run.interrupted", {"error": run.error, "compute_stopped": False}
                )
            return interrupted is not None

    async def recover_expired_leases(self) -> int:
        """Mark running jobs interrupted when their reservation expired or has no expiry.

        Check PostgreSQL's time and skip rows another transaction has locked. Keep each
        run's owner token, assigning an orphan marker if missing, so its slot stays
        occupied. Return the number of runs updated and commit those changes together.
        Do not rerun the model or assume a crashed worker's compute has stopped.
        """
        async with self.sessions.begin() as session:
            expired = list(
                (
                    await session.scalars(
                        select(Run)
                        .where(
                            Run.status.in_(LEASED_STATUSES),
                            or_(
                                Run.lease_expires_at <= func.clock_timestamp(),
                                Run.lease_expires_at.is_(None),
                            ),
                        )
                        .with_for_update(skip_locked=True)
                    )
                ).all()
            )
            for run in expired:
                run.status = "interrupted"
                run.lease_owner = run.lease_owner or f"orphaned:{run.id}"
                run.error = {
                    "code": "LEASE_EXPIRED",
                    "message": "Worker lease expired; compute stoppage is not confirmed",
                }
                run.finished_at = func.clock_timestamp()
                run.updated_at = func.clock_timestamp()
                await update_state_and_add_run_event(
                    session, run, "run.interrupted", {"error": run.error, "compute_stopped": False}
                )
            return len(expired)

    async def release_interrupted_run_slot_after_compute_stop_confirmed(
        self, run_id: UUID, *, expected_owner: str
    ) -> bool:
        """Release an interrupted run's slot after the caller confirms compute has stopped.

        run_id identifies the run; expected_owner must match its saved owner token.
        Return True after clearing the owner and expiry, or False if status/owner no
        longer matches. Keep the run interrupted; do not retry it or change its answer.

        This method performs no compute check itself. The operator or inference adapter
        must confirm stoppage first. It is not exposed as an HTTP endpoint.
        """
        async with self.sessions.begin() as session:
            released = await session.scalar(
                update(Run)
                .where(
                    Run.id == run_id,
                    Run.status == "interrupted",
                    Run.lease_owner == expected_owner,
                )
                .values(
                    lease_owner=None,
                    lease_expires_at=None,
                    updated_at=func.clock_timestamp(),
                )
                .returning(Run.id)
            )
            if released is not None:
                run = await session.get(Run, released)
                await update_state_and_add_run_event(
                    session, run, "run.compute_stopped", {"compute_stopped": True}
                )
            return released is not None

    async def confirm_cancellation(self, claim: RunClaim) -> bool:
        """Mark a requested cancellation complete only after the handler has stopped safely.

        Args:
            claim: Current owner/attempt whose handler finished all model/tool siblings.

        Returns:
            True after committing cancelled, releasing the slot and appending its event.
            False if cancellation was not requested or ownership/expiry changed.

        Do not call after forced task cancellation or uncertain remote inference.
        Those cases must remain interrupted with the slot held.
        """
        async with self.sessions() as session, session.begin():
            conversation = await session.scalar(
                select(Conversation)
                .where(Conversation.id == claim.conversation_id)
                .with_for_update()
            )
            run = await session.scalar(
                select(Run)
                .where(self._owned_lease(claim), Run.cancel_requested.is_(True))
                .with_for_update()
            )
            if conversation is None or run is None:
                return False
            run.status = "cancelled"
            run.finished_at = func.clock_timestamp()
            run.updated_at = func.clock_timestamp()
            run.lease_owner = None
            run.lease_expires_at = None
            submission = await session.get(Submission, run.submission_id)
            submission.status = "cancelled"
            await update_state_and_add_run_event(
                session, run, "run.cancelled", {"compute_stopped": True}
            )
            return True
