from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend import constants as CONST
from src.backend.ai import LocalMindAI
from src.backend.database import crud
from src.backend.database.models import Conversation, Message, Attachment
from src.backend.database.exceptions import ConversationNotFound


class ConversationService:
    def __init__(self, ai: LocalMindAI, session: AsyncSession) -> None:
        self.ai = ai
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

    async def create_message(
        self,
        content: str,
        conversation_id: UUID,
        role: str,
        model_name: str = None,
        temperature: float = None,
        reasoning_mode: str = None,
        personality: str = None,
        citations: list[Attachment] = [],
    ) -> Message:
        return await crud.create_message(
            content=content,
            conversation_id=conversation_id,
            role=role,
            model_name=model_name,
            temperature=temperature,
            reasoning_mode=reasoning_mode,
            personality=personality,
            citations=citations,
            session=self.session,
        )

    async def get_citations(
        self, conversation_id: UUID, attachments_ids: list[str]
    ) -> list[Attachment]:
        return await crud.get_citations(
            conversation_id=conversation_id, attachments_ids=attachments_ids, session=self.session
        )

    async def get_information_source(
        self, message_id: UUID, conversation_id: UUID
    ) -> list[Attachment] | None:
        return await crud.get_infomation_source(
            message_id=message_id, conversation_id=conversation_id, session=self.session
        )

    async def update_summary(self, conversation_id: UUID, model_name: str) -> None:
        conversation = await crud.get_conversation(conversation_id, self.session)
        if not conversation:
            raise ConversationNotFound(f"Conversation {conversation_id} not found.")

        summary = conversation.summary
        recent_30 = await crud.get_conversation_messages(
            conversation_id, self.session, limit=CONST.SUMMARY_BATCH + CONST.RECENT_MESSAGES_COUNT
        )

        messages_to_summarize = recent_30[: -CONST.RECENT_MESSAGES_COUNT]

        full = []
        if summary is not None:
            full.append(
                {
                    "role": "system",
                    "content": f"""
                Previous conversation summary:
                {summary}
                """,
                }
            )
        else:
            full.append(
                {
                    "role": "system",
                    "content": f"""
                Previous conversation summary:
                - None
                """,
                }
            )

        for msg in messages_to_summarize:
            if msg.role == "user":
                full.append({"role": "user", "content": msg.content})
            if msg.role == "assistant":
                full.append({"role": "assistant", "content": msg.content})

        full.append(
            {
                "role": "user",
                "content": """
        Update the conversation summary.

        Requirements:
        - Preserve important facts.
        - Preserve user preferences.
        - Preserve ongoing projects.
        - Preserve unfinished tasks.
        - Remove small talk.
        - Keep the summary under 500 words.
        - Return only the updated summary.
        """,
            }
        )

        new_summary = await self.ai.baseline_chat(full, model_name)
        conversation.summary = new_summary
        conversation.unsummarized_message_count = 0
        try:
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
