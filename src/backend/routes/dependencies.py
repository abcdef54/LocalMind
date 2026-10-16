from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.backend.database.session import get_async_session
from src.backend.ai import LocalMindAI
from src.backend.services.conversation_services import ConversationService
from src.backend.ai.llms.model_manager import LocalMindModelManager
from src.backend.services.model_services import ModelService

# Singleton instances for heavy resources
_ai = None
_model_manager = None


def get_ai() -> LocalMindAI:
    global _ai
    if _ai is None:
        _ai = LocalMindAI()
    return _ai


def get_model_manager() -> LocalMindModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = LocalMindModelManager()
    return _model_manager


def get_conversation_service(
    session: AsyncSession = Depends(get_async_session), ai: LocalMindAI = Depends(get_ai)
) -> ConversationService:
    return ConversationService(ai=ai, session=session)


def get_model_service(
    session: AsyncSession = Depends(get_async_session),
    manager: LocalMindModelManager = Depends(get_model_manager),
) -> ModelService:
    return ModelService(manager=manager, session=session)
