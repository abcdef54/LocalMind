"""Conversation mutations and bounded history queries without AI dependencies."""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, tuple_, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.backend.api.errors import ApiError
from src.backend.api.v1.pagination import HistoryCursor, decode_cursor, encode_cursor
from src.backend.api.v1.schemas import (
    CitationResponse,
    ConversationPage,
    ConversationResponse,
    MessagePage,
    MessageResponse,
)
from src.backend.database.models import ACTIVE_RUN_STATUSES, Attachment, Conversation, Message, Run


async def get_conversation(
    session: AsyncSession, conversation_id: UUID, *, lock: bool = False
) -> Conversation:
    query = select(Conversation).where(
        Conversation.id == conversation_id, Conversation.deleted_at.is_(None)
    )
    if lock:
        query = query.with_for_update()
    conversation = await session.scalar(query)
    if conversation is None:
        raise ApiError(404, "CONVERSATION_NOT_FOUND", "Conversation not found")
    return conversation


async def list_conversations(
    session: AsyncSession, limit: int, before: str | None
) -> ConversationPage:
    query = select(Conversation).where(Conversation.deleted_at.is_(None))
    if before is not None:
        cursor = decode_cursor(before, resource="conversations")
        query = query.where(
            tuple_(Conversation.created_at, Conversation.id) < (cursor.created_at, cursor.id)
        )
    query = query.order_by(Conversation.created_at.desc(), Conversation.id.desc()).limit(limit + 1)
    conversations = list((await session.scalars(query)).all())
    has_more = len(conversations) > limit
    conversations = conversations[:limit]
    next_cursor = None
    if has_more:
        oldest = conversations[-1]
        next_cursor = encode_cursor(
            HistoryCursor(resource="conversations", created_at=oldest.created_at, id=oldest.id)
        )
    return ConversationPage(
        items=[ConversationResponse.model_validate(conversation) for conversation in conversations],
        next_cursor=next_cursor,
        has_more=has_more,
    )


async def create_conversation(session: AsyncSession, title: str) -> Conversation:
    async with session.begin():
        conversation = Conversation(title=title)
        session.add(conversation)
        await session.flush()
        await session.refresh(conversation)
    return conversation


async def rename_conversation(
    session: AsyncSession, conversation_id: UUID, title: str
) -> Conversation:
    async with session.begin():
        conversation = await get_conversation(session, conversation_id, lock=True)
        conversation.title = title
        await session.flush()
    return conversation


async def delete_conversation(session: AsyncSession, conversation_id: UUID) -> None:
    """Archive metadata atomically; physical document cleanup is added in BE-09.

    Reject recorded active runs. Future run creation must lock/check the same
    conversation. Retrieval scoping and blob/index cleanup are separate tasks.
    """
    async with session.begin():
        conversation = await get_conversation(session, conversation_id, lock=True)
        active_run = await session.scalar(
            select(Run.id)
            .where(
                Run.conversation_id == conversation_id,
                Run.status.in_(ACTIVE_RUN_STATUSES),
            )
            .limit(1)
        )
        if active_run is not None:
            raise ApiError(
                409, "RUN_ACTIVE", "Stop or finish the active run before deleting this conversation"
            )
        deleted_at = datetime.now(timezone.utc).replace(tzinfo=None)
        conversation.deleted_at = deleted_at
        for model in (Message, Attachment):
            await session.execute(
                update(model)
                .where(model.conversation_id == conversation_id, model.deleted_at.is_(None))
                .values(deleted_at=deleted_at)
            )


async def list_messages(
    session: AsyncSession, conversation_id: UUID, limit: int, before: str | None
) -> MessagePage:
    await get_conversation(session, conversation_id)
    query = select(Message).where(
        Message.conversation_id == conversation_id, Message.deleted_at.is_(None)
    )
    if before is not None:
        cursor = decode_cursor(before, resource="messages", conversation_id=conversation_id)
        # Message.created_at remains timestamp without time zone in the legacy schema.
        anchor_time = cursor.created_at.replace(tzinfo=None)
        query = query.where(tuple_(Message.created_at, Message.id) < (anchor_time, cursor.id))
    query = (
        query.options(
            selectinload(Message.citations.and_(Attachment.conversation_id == conversation_id))
        )
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(limit + 1)
    )
    messages = list((await session.scalars(query)).all())
    has_more = len(messages) > limit
    messages = messages[:limit]
    next_cursor = None
    if has_more:
        oldest = messages[-1]
        next_cursor = encode_cursor(
            HistoryCursor(
                resource="messages",
                conversation_id=conversation_id,
                created_at=oldest.created_at.replace(tzinfo=timezone.utc),
                id=oldest.id,
            )
        )
    items = [
        MessageResponse(
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
            citations=[
                CitationResponse(
                    id=attachment.id,
                    file_name=attachment.file_name,
                    file_type=attachment.file_type,
                    file_size=attachment.file_size,
                    deleted=attachment.deleted_at is not None,
                )
                for attachment in message.citations
            ],
        )
        for message in reversed(messages)
    ]
    return MessagePage(items=items, next_cursor=next_cursor, has_more=has_more)
