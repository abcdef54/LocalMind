"""Short PostgreSQL transactions for claims, lease fencing and result publication."""

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal
from uuid import UUID, uuid4

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.backend.database.models import Conversation, Message, Run, Submission

ADMISSION_LOCK_KEY = 1280134734
LEASED_STATUSES = ("running", "cancel_requested")


@dataclass(frozen=True)
class RunClaim:
    run_id: UUID
    conversation_id: UUID
    submission_id: UUID
    message_id: UUID
    settings_snapshot: dict[str, Any]
    lease_owner: str
    attempt: int


@dataclass(frozen=True)
class RunResult:
    """Minimal handler output; LangGraph/citation integration is added in BE-10."""

    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("RunResult requires non-empty assistant content")


class RunStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def claim_next(
        self, *, worker_id: str, lease_seconds: float, max_concurrency: int
    ) -> RunClaim | None:
        """Claim ready queued work, admitting at most the shared concurrency limit.

        The advisory lock protects the short count-and-claim transaction across
        executor instances. Row locks are never held while a handler is running.
        Interrupted runs with an unresolved owner keep their slot occupied.
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
        return and_(
            Run.id == claim.run_id,
            Run.lease_owner == claim.lease_owner,
            Run.attempt == claim.attempt,
            Run.status.in_(LEASED_STATUSES),
            Run.lease_expires_at > func.clock_timestamp(),
        )

    async def heartbeat(
        self, claim: RunClaim, *, lease_seconds: float
    ) -> Literal["active", "cancel_requested", "lost"]:
        """An expired lease cannot be revived, even by its former owner."""
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

    async def complete(self, claim: RunClaim, result: RunResult) -> bool:
        """Publish exactly one assistant result atomically with run completion.

        A stale handler may still finish computing, but cannot write a result
        after ownership/lease loss. Full answer metadata/citations are BE-10 work.
        """
        async with self.sessions.begin() as session:
            run = await session.scalar(
                select(Run)
                .where(
                    self._owned_lease(claim),
                    Run.status == "running",
                    Run.cancel_requested.is_(False),
                )
                .with_for_update()
            )
            if run is None:
                return False
            message = Message(
                conversation_id=run.conversation_id,
                run_id=run.id,
                content=result.content,
                role="assistant",
            )
            session.add(message)
            await session.flush()
            run.result_message_id = message.id
            run.status = "completed"
            run.finished_at = func.clock_timestamp()
            run.updated_at = func.clock_timestamp()
            run.lease_owner = None
            run.lease_expires_at = None
            return True

    async def fail(self, claim: RunClaim) -> bool:
        """Record a returned handler failure; no automatic inference retry."""
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
            return failed is not None

    async def interrupt(self, claim: RunClaim, *, code: str) -> bool:
        """Fence writes but retain the occupied slot until compute stoppage is confirmed."""
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
            return interrupted is not None

    async def recover_expired_leases(self) -> int:
        """Mark expired/orphaned execution interrupted, never queued for an implicit retry."""
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
            return len(expired)

    async def acknowledge_interruption(self, run_id: UUID, *, expected_owner: str) -> bool:
        """Release an interrupted slot only AFTER the caller confirms compute has stopped.

        This operator/adapter primitive is not an HTTP endpoint and does not
        requeue the interrupted run. Checking the expected owner prevents stale
        acknowledgements from clearing a newer execution's slot.
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
            return released is not None
