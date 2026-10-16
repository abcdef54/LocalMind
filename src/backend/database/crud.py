from datetime import datetime
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from uuid import UUID

from src.backend.database.models import Conversation, Message, Attachment, InstalledModel
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


async def create_message(
    content: str,
    conversation_id: UUID,
    role: str,
    session: AsyncSession,
    model_name: str,
    temperature: float,
    reasoning_mode: str,
    personality: str,
    citations: list[Attachment] = [],
) -> Message:
    conversation_id = coerce_uuid(conversation_id)
    conversation = await session.execute(
        select(Conversation).where(
            Conversation.id == conversation_id, Conversation.deleted_at == None
        )
    )
    conversation = conversation.scalar_one_or_none()
    if not conversation:
        raise ConversationNotFound(f"Conversation {conversation_id} not found")

    new_message = Message(
        conversation_id=conversation_id,
        content=content,
        role=role,
        model_name=model_name,
        temperature=temperature,
        reasoning_mode=reasoning_mode,
        personality=personality,
        citations=citations or [],
    )
    try:
        session.add(new_message)
        await session.commit()
        await session.refresh(new_message)

        return new_message

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


async def get_citations(
    conversation_id: UUID, attachments_ids: list[str], session: AsyncSession
) -> list[Attachment]:
    """Get SqlAlchemy Attachment table objects from ids"""
    conversation_id = coerce_uuid(conversation_id)
    if not attachments_ids:
        return []

    stmt = (
        select(Attachment)
        .join(Conversation, Conversation.id == Attachment.conversation_id)
        .where(
            Attachment.conversation_id == conversation_id,
            Attachment.id.in_(attachments_ids),
            Conversation.deleted_at == None,  # <-- Guard against archived context indexing
        )
    )
    result = await session.execute(stmt)
    return result.scalars().all()


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


async def create_installed_model(
    display_name: str,
    model_name: str,
    hf_repo: str,
    gguf_file: str,
    local_path: str,
    quantization: str,
    size_bytes: int,
    is_default: bool,
    session: AsyncSession,
) -> InstalledModel:
    new_model = InstalledModel(
        display_name=display_name,
        model_name=model_name,
        hf_repo=hf_repo,
        gguf_file=gguf_file,
        local_path=local_path,
        quantization=quantization,
        size_bytes=size_bytes,
        is_default=is_default,
    )
    try:
        session.add(new_model)
        await session.commit()
        await session.refresh(new_model)
        return new_model
    except Exception:
        await session.rollback()
        raise


async def get_installed_models(session: AsyncSession) -> list[InstalledModel]:
    stmt = select(InstalledModel).order_by(InstalledModel.installed_at.desc())
    results = await session.execute(stmt)
    return list(results.scalars().all())


async def get_installed_model(id: UUID, session: AsyncSession) -> InstalledModel | None:
    id = coerce_uuid(id)
    stmt = select(InstalledModel).where(InstalledModel.id == id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def delete_installed_model(id: UUID, session: AsyncSession) -> InstalledModel:
    id = coerce_uuid(id)
    model = await get_installed_model(id, session)
    if not model:
        raise ValueError(f"Installed model with ID {id} not found.")
    try:
        await session.delete(model)
        await session.commit()
        return model
    except Exception:
        await session.rollback()
        raise


async def update_installed_model(
    id: UUID, update_data: dict, session: AsyncSession
) -> InstalledModel:
    id = coerce_uuid(id)
    model = await get_installed_model(id, session)
    if not model:
        raise ValueError(f"Installed model with ID {id} not found.")
    try:
        for key, val in update_data.items():
            if val is not None:
                setattr(model, key, val)
        await session.commit()
        await session.refresh(model)
        return model
    except Exception:
        await session.rollback()
        raise
