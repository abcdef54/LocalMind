"""Expose conversation CRUD and paginated message history under /api/v1."""

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
    """Create a conversation from payload.title and return its ID/title/creation time.

    session is the request's database session. The service commits before this route
    returns HTTP 201; invalid or empty titles are rejected by ConversationInput.
    """
    conversation = await conversation_history.create_conversation(session, payload.title)
    return ConversationResponse.model_validate(conversation)


@router.get("", response_model=ConversationPage)
async def list_conversations(
    session: SessionDependency, limit: PageLimit = 50, before: BeforeCursor = None
) -> ConversationPage:
    """Return one newest-first page of conversations that have not been deleted.

    limit is the page size (default 50, maximum 100); before is the previous page's
    cursor or None for the first page. Return items, has_more and next_cursor.
    Do not load the full conversation list. Invalid cursors return HTTP 422.
    """
    return await conversation_history.list_conversations(session, limit, before)


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: UUID, session: SessionDependency
) -> ConversationResponse:
    """Return the live conversation identified by conversation_id using the request session.

    Return HTTP 404 when the conversation is missing or has been deleted.
    """
    conversation = await conversation_history.get_conversation(session, conversation_id)
    return ConversationResponse.model_validate(conversation)


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def rename_conversation(
    conversation_id: UUID, payload: ConversationInput, session: SessionDependency
) -> ConversationResponse:
    """Save payload.title for the live conversation and return its updated metadata.

    conversation_id selects the chat; session supplies the transaction. The service
    locks the row and commits the rename. Missing/deleted conversations return 404.
    """
    conversation = await conversation_history.rename_conversation(
        session, conversation_id, payload.title
    )
    return ConversationResponse.model_validate(conversation)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: UUID, session: SessionDependency) -> Response:
    """Mark a conversation and its visible messages/files deleted, then return HTTP 204.

    conversation_id selects the chat and session supplies the transaction. Return
    404 if missing/deleted, or 409 if a recorded run is active. This changes metadata;
    physical document and vector deletion is implemented separately in BE-09.
    """
    await conversation_history.delete_conversation(session, conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{conversation_id}/messages", response_model=MessagePage)
async def list_messages(
    conversation_id: UUID,
    session: SessionDependency,
    limit: PageLimit = 50,
    before: BeforeCursor = None,
) -> MessagePage:
    """Return one page of older messages from the selected live conversation.

    limit defaults to 50 and is capped at 100. before selects messages older than
    the previous page's timestamp/ID; omit it to get the newest messages. Items within
    each page run oldest to newest. Include has_more/next_cursor, attachments and
    citations. Return 404 for a missing chat or 422 for an invalid/mismatched cursor.
    """
    return await conversation_history.list_messages(session, conversation_id, limit, before)
