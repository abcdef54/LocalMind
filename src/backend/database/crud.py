import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID

from src.backend.database.models import Conversation, Message, Attachment
from src.backend.database.exceptions import ConversationNotFound


def coerce_uuid(val):
    if isinstance(val, str):
        return uuid.UUID(val)
    return val


async def create_conversation(title: str, session: AsyncSession) -> Conversation:
    try:
        new_conversation = Conversation(
            title=title,
        )

        session.add(new_conversation)
        await session.commit()
        await session.refresh(new_conversation)

        return new_conversation

    except Exception:
        await session.rollback()
        raise


async def get_conversations(session: AsyncSession) -> list[Conversation]:
    stmt = (
        select(Conversation)
        .where(Conversation.deleted_at == None)
        .order_by(Conversation.created_at.desc())
    )
    results = await session.scalars(stmt)
    return results.all()


async def get_conversation(id: UUID, session: AsyncSession) -> Conversation:
    id = coerce_uuid(id)
    conversation = await session.execute(
        select(Conversation).where(Conversation.id == id, Conversation.deleted_at == None)
    )
    conversation = conversation.scalar_one_or_none()
    if not conversation:
        raise ConversationNotFound(f"Conversation {id} not found.")
    return conversation


async def get_conversation_messages(
    conversation_id: UUID, session: AsyncSession, limit: int = None
) -> list[Message]:
    conversation_id = coerce_uuid(conversation_id)

    conversation = await session.execute(
        select(Conversation).where(
            Conversation.id == conversation_id, Conversation.deleted_at == None
        )
    )
    conversation = conversation.scalar_one_or_none()

    if conversation is None:
        raise ConversationNotFound(f"Conversation {conversation_id} not found")

    if limit is None:
        results = await session.scalars(
            select(Message).where(
                Message.conversation_id == conversation_id, Message.deleted_at == None
            )
        )
    else:
        results = await session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id, Message.deleted_at == None)
            .limit(limit)
        )
    return results.all()


async def rename_conversation(
    conversation_id: UUID, new_name: str, session: AsyncSession
) -> Conversation:
    conversation_id = coerce_uuid(conversation_id)
    conversation = await session.execute(
        select(Conversation).where(
            Conversation.id == conversation_id, Conversation.deleted_at == None
        )
    )
    conversation = conversation.scalar_one_or_none()
    if not conversation:
        raise ConversationNotFound(f"Conversation {conversation_id} not found.")

    try:
        conversation.title = new_name
        await session.commit()
        await session.refresh(conversation)

        return conversation
    except Exception:
        await session.rollback()
        raise


async def get_infomation_source(
    message_id: UUID, conversation_id: UUID, session: AsyncSession
) -> list[Attachment] | None:
    """
    Returns the collection of file attachments that were cited/used to generate
    a specific assistant response message.
    """
    message_id = coerce_uuid(message_id)
    conversation_id = coerce_uuid(conversation_id)
    conversation = await session.execute(
        select(Conversation).where(
            Conversation.id == conversation_id, Conversation.deleted_at == None
        )
    )
    conversation = conversation.scalar_one_or_none()
    if not conversation:
        raise ConversationNotFound(f"Conversation {conversation_id} not found.")

    stmt = select(Message).where(
        Message.id == message_id, Message.conversation_id == conversation_id
    )

    result = await session.execute(stmt)
    message = result.scalar_one_or_none()
    if not message:
        raise ValueError(f"Message {message_id} not found in conversation {conversation_id}.")

    return getattr(message, "citations", [])
