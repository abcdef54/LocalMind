"""Load a claimed send, invoke LangGraph and return an answer for guarded persistence."""

from uuid import UUID
from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.backend.ai.errors import InferenceStateUnknown
from src.backend.api.v1.schemas import ChatSettings
from src.backend.constants import RECENT_MESSAGES_COUNT
from src.backend.database.models import Attachment, Conversation, Message, Run, Submission
from src.backend.execution.store import RunClaim, RunResult, RunStore
from src.backend.execution.worker import ComputeStateUnknown


class GenerateReply:
    """Generate a reply using saved settings and a bounded backend-selected history.

    Args:
        sessions: Database factory for reading input in a short transaction.
        ai: Shared LocalMindAI instance, or a fake exposing the same async chat method.

    This handler never saves messages or holds a DB session during inference.
    Existing summaries are read but not updated here; watermark/token-budget memory
    improvements belong to BE-11. Browser pagination never controls model context.
    """

    def __init__(self, sessions: async_sessionmaker[AsyncSession], ai) -> None:
        self.sessions = sessions
        self.ai = ai

    async def _load_input(self, claim: RunClaim) -> tuple[list[dict[str, str]], ChatSettings]:
        """Load current input exactly once, preceded by recent history and optional summary.

        Args:
            claim: Reserved run and user-message identity, including owner and attempt.

        Returns:
            Chronological role/content messages and the validated stored settings.
            Current uploaded filenames are included so file-only sends have context.

        Raises:
            ValueError: If the lease/chat/input is invalid or a submitted file is no
                longer ready. No model call is started for these cases.
        """
        async with self.sessions() as session:
            run = await session.scalar(
                select(Run).where(
                    RunStore._owned_lease(claim),
                    Run.status == "running",
                    Run.cancel_requested.is_(False),
                    Run.conversation_id == claim.conversation_id,
                    Run.submission_id == claim.submission_id,
                    Run.message_id == claim.message_id,
                )
            )
            conversation = await session.get(Conversation, claim.conversation_id)
            submission = await session.get(Submission, claim.submission_id)
            if (
                run is None
                or conversation is None
                or conversation.deleted_at is not None
                or submission is None
                or submission.status != "ready"
            ):
                raise ValueError("Run input is no longer eligible")
            settings = ChatSettings.model_validate(run.settings_snapshot)
            # Select columns, not relationship-loaded ORM messages: context does not
            # need citation/blob rows and must not trigger hidden full-history reads.
            current = (
                await session.execute(
                    select(Message.content, Message.created_at, Message.id).where(
                        Message.id == claim.message_id,
                        Message.conversation_id == claim.conversation_id,
                        Message.submission_id == claim.submission_id,
                        Message.role == "user",
                        Message.deleted_at.is_(None),
                    )
                )
            ).one_or_none()
            if current is None:
                raise ValueError("Run user message is missing")
            attachments = list(
                (
                    await session.scalars(
                        select(Attachment).where(Attachment.submission_id == claim.submission_id)
                    )
                ).all()
            )
            if any(
                file.conversation_id != claim.conversation_id
                or file.deleted_at is not None
                or file.status != "ready"
                or file.index_generation is None
                for file in attachments
            ):
                raise ValueError("Submitted document is no longer ready")
            previous = (
                await session.execute(
                    select(Message.role, Message.content)
                    .where(
                        Message.conversation_id == claim.conversation_id,
                        Message.deleted_at.is_(None),
                        Message.role.in_(("user", "assistant")),
                        tuple_(Message.created_at, Message.id) < (current.created_at, current.id),
                    )
                    .order_by(Message.created_at.desc(), Message.id.desc())
                    .limit(RECENT_MESSAGES_COUNT)
                )
            ).all()
            messages = []
            if conversation.summary:
                messages.append(
                    {
                        "role": "system",
                        "content": "Previous conversation summary:\n" + conversation.summary,
                    }
                )
            messages.extend(
                {"role": row.role, "content": row.content} for row in reversed(previous)
            )
            content = current.content
            if attachments:
                names = "\n".join(f"- {file.file_name}" for file in attachments)
                content += "\n\nFiles uploaded with this message:\n" + names
            messages.append({"role": "user", "content": content})
            return messages, settings

    async def __call__(self, claim: RunClaim) -> RunResult:
        """Generate one result without publishing it or releasing the execution slot.

        Args:
            claim: Current execution reservation supplied by RunExecutor.

        Returns:
            Nonblank answer, deduplicated document UUIDs and compact generation metadata.
            RunStore revalidates document visibility and lease ownership at commit.

        Raises:
            ComputeStateUnknown: Remote inference may still be running after a transport
                failure; the executor retains its admission slot.
            Exception: Invalid inputs, model responses or ordinary generation failures
                propagate so the executor records failure without a fake assistant answer.
        """
        messages, settings = await self._load_input(claim)
        try:
            response = await self.ai.chat(
                messages=messages,
                model_name=settings.model_name,
                personality=settings.personality,
                reasoning_mode=settings.reasoning_mode,
                temperature=settings.temperature,
                tools_enabled=settings.tools_enabled,
                conversation_id=claim.conversation_id,
            )
        except InferenceStateUnknown as error:
            raise ComputeStateUnknown() from error
        answer = response["final_answer"]
        citation_ids = tuple(UUID(str(identity)) for identity in response.get("citations", []))
        return RunResult(
            content=answer,
            citation_ids=citation_ids,
            generation_metadata=response.get("generation_metadata", {}),
        )
