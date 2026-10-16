"""Conversation v1 routes; messages are always fetched as bounded cursor pages."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.api.v1.dependencies import get_session
from src.backend.api.v1.schemas import (
    ConversationInput,
    ConversationPage,
    ConversationResponse,
    MessagePage,
)
from src.backend.services import conversation_history

router = APIRouter(prefix="/conversations", tags=["conversations"])
SessionDependency = Annotated[AsyncSession, Depends(get_session)]
PageLimit = Annotated[int, Query(ge=1, le=100)]
BeforeCursor = Annotated[str | None, Query(min_length=1, max_length=1024)]


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    payload: ConversationInput, session: SessionDependency
) -> ConversationResponse:
    conversation = await conversation_history.create_conversation(session, payload.title)
    return ConversationResponse.model_validate(conversation)


@router.get("", response_model=ConversationPage)
async def list_conversations(
    session: SessionDependency, limit: PageLimit = 50, before: BeforeCursor = None
) -> ConversationPage:
    return await conversation_history.list_conversations(session, limit, before)


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: UUID, session: SessionDependency
) -> ConversationResponse:
    conversation = await conversation_history.get_conversation(session, conversation_id)
    return ConversationResponse.model_validate(conversation)


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def rename_conversation(
    conversation_id: UUID, payload: ConversationInput, session: SessionDependency
) -> ConversationResponse:
    conversation = await conversation_history.rename_conversation(
        session, conversation_id, payload.title
    )
    return ConversationResponse.model_validate(conversation)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: UUID, session: SessionDependency) -> Response:
    await conversation_history.delete_conversation(session, conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{conversation_id}/messages", response_model=MessagePage)
async def list_messages(
    conversation_id: UUID,
    session: SessionDependency,
    limit: PageLimit = 50,
    before: BeforeCursor = None,
) -> MessagePage:
    return await conversation_history.list_messages(session, conversation_id, limit, before)
