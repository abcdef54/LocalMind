import traceback
from fastapi import Depends, HTTPException, APIRouter

from src.backend.database.exceptions import ConversationNotFound
from src.backend.database.schemas import ChatRequest, ChatResponse
from src.backend import constants as CONST
from src.backend.services.conversation_services import ConversationService
from src.backend.routes.dependencies import get_conversation_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
async def chat(
    request: ChatRequest,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ChatResponse:
    print(f"\n[DEBUG /chat] Starting chat invocation...")
    print(f"  - conversation_id: {request.conversation_id}")
    print(f"  - content: '{request.content[:50]}...'")
    print(f"  - reasoning_mode: {request.reasoning_mode}")
    print(f"  - personality: {request.personality}")
    print(f"  - model_name: {request.model_name}")
    print(f"  - temperature: {request.temperature}")
    try:
        local_mind = conversation_service.ai
        full_chat = []
        existing_conversation = await conversation_service.get_conversation(request.conversation_id)
        summary = existing_conversation.summary

        recent_messages = await conversation_service.get_conversation_messages(
            conversation_id=request.conversation_id, limit=CONST.RECENT_MESSAGES_COUNT
        )

        if summary:
            full_chat.append(
                {
                    "role": "system",
                    "content": f"""
                Previous conversation summary:
                {summary}
                """,
                }
            )

        for msg in recent_messages:
            if msg.role == "user":
                full_chat.append({"role": "user", "content": msg.content})
            else:
                full_chat.append({"role": "assistant", "content": msg.content})

        full_chat.append({"role": "user", "content": request.content})

        print("[DEBUG /chat] Creating new user message in DB...")
        message = await conversation_service.create_message(
            content=request.content,
            conversation_id=request.conversation_id,
            role="user",
            personality=None,
            model_name=None,
            temperature=None,
            reasoning_mode=None,
        )
        print(f"[DEBUG /chat] User message created successfully with ID: {message.id}")

        print("[DEBUG /chat] Invoking LangGraph agent...")
        agent_response = await local_mind(
            messages=full_chat,
            model_name=request.model_name,
            personality=request.personality,
            reasoning_mode=request.reasoning_mode,
            temperature=request.temperature,
            tools_enabled=request.tools_enabled,
            conversation_id=request.conversation_id,
        )
        print("[DEBUG /chat] Agent invocation completed successfully.")
        print(
            f"[DEBUG /chat] Extracted response content (first 80 chars): '{agent_response['final_answer'][:80]}...'"
        )
        print("[DEBUG /chat] Fetching citations/sources...")
        citations = await conversation_service.get_citations(
            conversation_id=request.conversation_id, attachments_ids=agent_response["citations"]
        )

        print("[DEBUG /chat] Creating assistant response message in DB...")
        assistant_message = await conversation_service.create_message(
            content=agent_response["final_answer"],
            conversation_id=request.conversation_id,
            role="assistant",
            citations=citations,
            personality=request.personality,
            model_name=request.model_name,
            temperature=request.temperature,
            reasoning_mode=request.reasoning_mode,
        )
        print(f"[DEBUG /chat] Assistant message saved successfully with ID: {assistant_message.id}")

        conversation = await conversation_service.get_conversation(request.conversation_id)
        if (
            conversation.message_count >= CONST.RECENT_MESSAGES_COUNT + CONST.SUMMARY_BATCH
            and conversation.unsummarized_message_count >= CONST.SUMMARY_BATCH
        ):
            await conversation_service.update_summary(
                conversation_id=request.conversation_id, model_name=request.model_name
            )

        return ChatResponse(
            id=assistant_message.id,
            conversation_id=assistant_message.conversation_id,
            content=assistant_message.content,
            role=assistant_message.role,
            personality=assistant_message.personality,
            citations=assistant_message.citations,
            created_at=assistant_message.created_at,
        )

    except HTTPException:
        raise
    except ConversationNotFound as e:
        print(f"[DEBUG /chat] EXCEPTION: ConversationNotFound: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"[DEBUG /chat] CRITICAL RUNTIME EXCEPTION:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
