from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.database import crud
from src.backend.database.models import Conversation, Message, Attachment


class ConversationService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_conversation(self, title: str) -> Conversation:
        return await crud.create_conversation(title=title, session=self.session)

    async def get_conversations(self) -> list[Conversation]:
        return await crud.get_conversations(session=self.session)

    async def get_conversation(self, id: UUID) -> Conversation:
        return await crud.get_conversation(id=id, session=self.session)

    async def get_conversation_messages(
        self, conversation_id: UUID, limit: int = None
    ) -> list[Message]:
        return await crud.get_conversation_messages(
            conversation_id=conversation_id, session=self.session, limit=limit
        )

    async def rename_conversation(self, conversation_id: UUID, new_name: str) -> Conversation:
        return await crud.rename_conversation(
            conversation_id=conversation_id, new_name=new_name, session=self.session
        )

    async def get_information_source(
        self, message_id: UUID, conversation_id: UUID
    ) -> list[Attachment] | None:
        return await crud.get_infomation_source(
            message_id=message_id, conversation_id=conversation_id, session=self.session
        )
