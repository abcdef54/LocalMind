"""Create, rename and hide conversations, and read their history one page at a time."""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, tuple_, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.backend.api.errors import ApiError
from src.backend.api.v1.pagination import HistoryCursor, decode_cursor, encode_cursor
from src.backend.api.v1.schemas import (
    CitationResponse,
    DocumentResponse,
    ConversationPage,
    ConversationResponse,
    MessagePage,
    MessageResponse,
)
from src.backend.database.models import ACTIVE_RUN_STATUSES, Attachment, Conversation, Message, Run


async def get_conversation(
    session: AsyncSession, conversation_id: UUID, *, lock: bool = False
) -> Conversation:
    """Return the non-deleted Conversation with the given UUID, or raise ApiError 404.

    session supplies the database connection. With lock=True, lock the row until
    the caller's transaction ends so sends, renames and deletion can be coordinated.
    Do not commit or close the supplied session here.
    """
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
    """Read at most limit visible conversations, newest first, and return page metadata.

    session supplies the read transaction. before is None for the first page or a
    conversation cursor for an older page. Compare created_at and ID together so
    equal timestamps remain ordered. Fetch one extra row to set has_more, without
    counting/loading the full table. Raise ApiError 422 for an invalid cursor.
    """
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
    """Insert a conversation with the supplied title and return the saved ORM row.

    session must have no open transaction. Commit the insert before returning;
    propagate DB failures so the transaction rolls back. Title validation happens
    in the API schema before this service is called.
    """
    async with session.begin():
        conversation = Conversation(title=title)
        session.add(conversation)
        await session.flush()
        await session.refresh(conversation)
    return conversation


async def rename_conversation(
    session: AsyncSession, conversation_id: UUID, title: str
) -> Conversation:
    """Lock a live conversation, update its title and commit the change.

    Use conversation_id to select the row and title as the validated replacement.
    Return the updated Conversation. Raise ApiError 404 if missing/deleted; propagate
    database errors and roll back rather than returning an unsaved rename.
    """
    async with session.begin():
        conversation = await get_conversation(session, conversation_id, lock=True)
        conversation.title = title
        await session.flush()
    return conversation


async def delete_conversation(session: AsyncSession, conversation_id: UUID) -> None:
    """Hide a conversation and its messages/files by saving deletion timestamps.

    session must have no open transaction. Lock conversation_id and reject recorded
    active runs with ApiError 409. Raise 404 if the chat is missing/already deleted.
    Commit metadata changes together and return None. Queue documents marked as deleted for background file/vector cleanup.
    Physical cleanup completion is tracked separately; historical citations remain.
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
            values = {"deleted_at": deleted_at}
            if model is Attachment:
                values["status"] = "deleted"
            await session.execute(
                update(model)
                .where(model.conversation_id == conversation_id, model.deleted_at.is_(None))
                .values(**values)
            )


async def list_messages(
    session: AsyncSession, conversation_id: UUID, limit: int, before: str | None
) -> MessagePage:
    """Return a limited page of messages and their own conversation's file references.

    session reads conversation_id after checking the chat is live. limit is the
    validated page size; before is None or a message cursor for this same chat.
    Select the newest eligible rows with limit+1, then return items oldest to newest
    with has_more/next_cursor. Hide deleted messages but retain deleted-file markers.
    Raise ApiError 404 for a missing chat or 422 for an invalid/cross-chat cursor.
    This query does not load full history or determine the model's context window.
    """
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
            selectinload(Message.citations.and_(Attachment.conversation_id == conversation_id)),
            selectinload(Message.attachments.and_(Attachment.conversation_id == conversation_id)),
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
            attachments=[
                DocumentResponse(
                    id=attachment.id,
                    client_file_id=attachment.client_file_id,
                    file_name=attachment.file_name,
                    file_type=attachment.file_type,
                    file_size=attachment.file_size,
                    status=attachment.status,
                    deleted=attachment.deleted_at is not None,
                    error=attachment.error,
                    chunk_count=attachment.chunk_count,
                )
                for attachment in message.attachments
            ],
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
