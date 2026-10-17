"""Summarize older history once and assemble a bounded, chronological model context."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import func, select, tuple_
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.backend.ai.system_prompt import get_localmind_system_instruction_with_personality
from src.backend.api.v1.schemas import ChatSettings
from src.backend.database.models import Conversation, Message, Run
from src.backend.execution.store import RunClaim, RunStore
from src.backend.memory.budget import (
    estimate_messages_tokens,
    estimate_text_tokens,
    split_text_for_summary,
)
from src.backend.memory.config import MemorySettings


class ConversationMemory:
    """Prepare model context and save summaries only after successful generation.

    Args:
        sessions: Factory for short database reads and summary commit transactions.
        ai: AI engine exposing async summarize_conversation; calls use the run's slot.
        settings: Context reserves and bounded history/summary batch sizes.

    A saved (created_at, message_id) pair identifies the last summarized message.
    Older messages are never also included as raw context. Legacy summaries without
    a boundary are ignored and rebuilt from stored history. No DB session remains
    open while the model summarizes, and stale runs cannot advance the boundary.
    """

    def __init__(
        self, sessions: async_sessionmaker, ai, settings: MemorySettings | None = None
    ) -> None:
        self.sessions = sessions
        self.ai = ai
        self.settings = settings or MemorySettings.from_environment()

    async def _save_summary(
        self, claim: RunClaim, version: int, summary: str, boundary: tuple[datetime, UUID]
    ) -> None:
        """Commit summary and coverage together while this run still owns its lease.

        Args:
            claim: Current generation reservation supplied by the executor.
            version: Summary version read before inference; reject concurrent changes.
            summary: Validated compact summary covering all history through boundary.
            boundary: Timestamp/UUID of the last whole message summarized.

        Raises:
            ValueError: Chat/version/lease/cancellation changed; no summary is saved.
        """
        async with self.sessions() as session, session.begin():
            conversation = await session.scalar(
                select(Conversation)
                .where(Conversation.id == claim.conversation_id, Conversation.deleted_at.is_(None))
                .with_for_update()
            )
            owned_run = await session.scalar(
                select(Run.id)
                .where(
                    RunStore._owned_lease(claim),
                    Run.status == "running",
                    Run.cancel_requested.is_(False),
                )
                .with_for_update()
            )
            if conversation is None or conversation.summary_version != version or owned_run is None:
                raise ValueError("Conversation summary or run reservation changed")
            conversation.summary = summary
            conversation.summary_through_created_at, conversation.summary_through_message_id = (
                boundary
            )
            conversation.summary_version += 1
            conversation.unsummarized_message_count = await session.scalar(
                select(func.count())
                .select_from(Message)
                .where(
                    Message.conversation_id == claim.conversation_id,
                    Message.deleted_at.is_(None),
                    tuple_(Message.created_at, Message.id) > boundary,
                )
            )
            if (
                await session.scalar(
                    select(Run.id).where(
                        RunStore._owned_lease(claim),
                        Run.status == "running",
                        Run.cancel_requested.is_(False),
                    )
                )
                is None
            ):
                raise ValueError("Run reservation expired before summary commit")

    async def summarize_old_history_and_build_messages_for_inference(
        self,
        claim: RunClaim,
        chat_settings: ChatSettings,
        current_content: str,
        current_position: tuple[datetime, UUID],
    ) -> list[dict[str, str]]:
        """Summarize older history and return summary + recent messages + current input.

        Args:
            claim: Owned run identifying the conversation and generation attempt.
            chat_settings: Saved model/personality/tool settings for this send.
            current_content: Current user text with accepted uploaded filenames.
            current_position: Timestamp/UUID of the current message; later rows are excluded.

        Returns:
            Chronological messages within the conservative input budget. The current
            message appears exactly once and is never silently truncated.

        Raises:
            ValueError: Current input is too large, summary output is invalid, or
                reservation/version checks fail. Failed summary batches do not advance
                their saved boundary; previously committed batches remain usable.
            InferenceStateUnknown: Summary computation may still be running remotely.
        """
        system = get_localmind_system_instruction_with_personality(chat_settings.personality)
        current_message = {"role": "user", "content": current_content}
        available = (
            self.settings.context_tokens - self.settings.reply_tokens - self.settings.tool_tokens
        )
        # Reserve the maximum summary and its wrapper before choosing raw history.
        raw_budget = (
            available
            - estimate_messages_tokens([{"role": "system", "content": system}, current_message])
            - self.settings.summary_tokens
            - 128
        )
        if raw_budget < 0:
            raise ValueError("Current question exceeds the configured context budget")
        async with self.sessions() as session:
            conversation = await session.get(Conversation, claim.conversation_id)
            if conversation is None or conversation.deleted_at is not None:
                raise ValueError("Conversation no longer exists")
            version = conversation.summary_version
            boundary = None
            summary = ""
            if conversation.summary_through_message_id is not None:
                boundary = (
                    conversation.summary_through_created_at,
                    conversation.summary_through_message_id,
                )
                if boundary >= current_position:
                    raise ValueError("Summary includes this or a later message")
                summary = conversation.summary or ""
                if estimate_text_tokens(summary) > self.settings.summary_tokens:
                    raise ValueError("Saved summary exceeds the configured summary budget")
            query = self._history_query(claim.conversation_id, current_position, boundary)
            recent = (
                await session.execute(
                    query.order_by(Message.created_at.desc(), Message.id.desc()).limit(
                        self.settings.recent_messages
                    )
                )
            ).all()
        raw_messages = []
        raw_positions = []
        used = 0
        for row in recent:
            message = {"role": row.role, "content": row.content}
            cost = estimate_messages_tokens([message])
            if used + cost > raw_budget:
                break
            used += cost
            raw_messages.append(message)
            raw_positions.append((row.created_at, row.id))
        raw_messages.reverse()
        cutoff = raw_positions[-1] if raw_positions else current_position
        while True:
            async with self.sessions() as session:
                rows = (
                    await session.execute(
                        self._history_query(claim.conversation_id, cutoff, boundary)
                        .order_by(Message.created_at, Message.id)
                        .limit(self.settings.batch_messages)
                    )
                ).all()
            if not rows:
                break
            # Leave room for summary instructions, prior summary and JSON framing.
            batch_capacity = available - self.settings.summary_tokens - 2048
            if batch_capacity < 128:
                raise ValueError("Context budget leaves no room for summary input")
            batch = []
            covered = []
            for row in rows:
                message = {"role": row.role, "content": row.content}
                if estimate_messages_tokens(batch + [message]) > batch_capacity:
                    break
                batch.append(message)
                covered.append(row)
            next_summary = summary
            if batch:
                next_summary = await self._summarize(next_summary, batch, chat_settings.model_name)
            else:
                # An oversized old message is processed in full across bounded
                # requests. Only commit its boundary after every piece succeeds.
                row = rows[0]
                for piece in split_text_for_summary(row.content, batch_capacity - 128):
                    next_summary = await self._summarize(
                        next_summary,
                        [{"role": row.role, "content": piece}],
                        chat_settings.model_name,
                    )
                covered = [row]
            last = covered[-1]
            next_boundary = (last.created_at, last.id)
            await self._save_summary(claim, version, next_summary, next_boundary)
            summary = next_summary
            boundary = next_boundary
            version += 1
        messages = []
        if summary:
            messages.append(
                {"role": "system", "content": "Previous conversation summary:\n" + summary}
            )
        messages.extend(raw_messages)
        messages.append(current_message)
        if estimate_messages_tokens([{"role": "system", "content": system}] + messages) > available:
            raise ValueError("Prepared context exceeds the configured input budget")
        return messages

    @staticmethod
    def _history_query(
        conversation_id: UUID, upper: tuple[datetime, UUID], boundary: tuple[datetime, UUID] | None
    ):
        """Build a scoped history query after boundary and strictly before upper.

        Args:
            conversation_id: Chat whose live user/assistant messages are selected.
            upper: Exclusive timestamp/UUID upper bound; excludes current/future input.
            boundary: Last summarized position, or None to start at oldest history.

        Returns:
            Unexecuted query; callers must add ordering and a bounded LIMIT.
        """
        query = select(Message.id, Message.created_at, Message.role, Message.content).where(
            Message.conversation_id == conversation_id,
            Message.deleted_at.is_(None),
            Message.role.in_(("user", "assistant")),
            tuple_(Message.created_at, Message.id) < upper,
        )
        if boundary is not None:
            query = query.where(tuple_(Message.created_at, Message.id) > boundary)
        return query

    async def _summarize(
        self, previous: str, messages: list[dict[str, str]], model_name: str
    ) -> str:
        """Generate and validate a compact replacement summary without saving it yet.

        Args:
            previous: Summary already covering earlier messages, possibly empty.
            messages: Next contiguous batch or piece of historical conversation.
            model_name: Saved model name used under the currently reserved run slot.

        Returns:
            Nonblank summary within the configured conservative token estimate.

        Raises:
            ValueError: Output is blank, non-text or too large; coverage is not advanced.
        """
        summary = await self.ai.summarize_conversation(
            previous, messages, model_name, max_tokens=min(512, self.settings.summary_tokens)
        )
        if (
            not isinstance(summary, str)
            or not summary.strip()
            or estimate_text_tokens(summary) > self.settings.summary_tokens
        ):
            raise ValueError("Model returned an invalid or oversized conversation summary")
        return summary
