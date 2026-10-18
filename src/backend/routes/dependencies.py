from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.backend.database.session import get_async_session
from src.backend.ai import LocalMindAI
from src.backend.services.conversation_services import ConversationService

# Singleton instances for heavy resources
_ai = None


def get_ai() -> LocalMindAI:
    global _ai
    if _ai is None:
        _ai = LocalMindAI()
    return _ai


def get_conversation_service(
    session: AsyncSession = Depends(get_async_session),
) -> ConversationService:
    return ConversationService(session=session)
