from fastapi import HTTPException, Depends, APIRouter
from src.backend.database.schemas import (
    ConversationCreate,
    ConversationResponse,
    ChatResponse,
    AttachmentReponse,
)
from src.backend.database.exceptions import ConversationNotFound
from src.backend.services.conversation_services import ConversationService
from src.backend.routes.dependencies import get_conversation_service

router = APIRouter(prefix="/conversation", tags=["conversations"])


@router.post("", response_model=ConversationResponse)
async def create_conversation(
    conversation_create: ConversationCreate,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationResponse:
    try:
        conversation = await conversation_service.create_conversation(
            title=conversation_create.title
        )
        return ConversationResponse.model_validate(conversation)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=list[ConversationResponse])
async def get_conversations(
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> list[ConversationResponse]:
    try:
        conversations = await conversation_service.get_conversations()
        return conversations
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{id}", response_model=ConversationResponse)
async def get_conversation(
    id: str, conversation_service: ConversationService = Depends(get_conversation_service)
) -> ConversationResponse:
    try:
        conversation = await conversation_service.get_conversation(id)
        return ConversationResponse.model_validate(conversation)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}/messages", response_model=list[ChatResponse])
async def get_conversation_messages(
    conversation_id: str,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> list[ChatResponse]:
    try:
        conversation = await conversation_service.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=404, detail=f"Conversation {conversation_id} not found."
            )

        messages = await conversation_service.get_conversation_messages(
            conversation_id=conversation_id
        )
        return [ChatResponse.model_validate(m) for m in messages]
    except ConversationNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{conversation_id}/messages/{message_id}/sources", response_model=list[AttachmentReponse]
)
async def get_message_sources(
    conversation_id: str,
    message_id: str,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> list[AttachmentReponse]:
    try:
        citations = await conversation_service.get_information_source(
            message_id=message_id, conversation_id=conversation_id
        )

        return citations
    except ConversationNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while fetching citation sources: {str(e)}",
        )


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def rename_conversation(
    conversation_id: str,
    new_name: str,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationResponse:
    try:
        updated_conversation = await conversation_service.rename_conversation(
            conversation_id=conversation_id, new_name=new_name
        )
        return ConversationResponse.model_validate(updated_conversation)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
