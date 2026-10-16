"""Choose eligible documents from PostgreSQL before and after a scoped vector search."""

from uuid import UUID

from langchain_core.documents import Document
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.backend.database.models import Attachment, Conversation
from src.backend.indexing.vectors import DocumentVectorIndex
from src.backend.uploads.storage import run_file_operation


async def ready_document_keys(session: AsyncSession, conversation_id: UUID) -> set[str]:
    """Read ready, undeleted attachment generations from a live conversation.

    Args:
        session: Read session; the caller controls its transaction lifetime.
        conversation_id: Server-selected chat UUID, never a model-selected scope.

    Returns:
        Attachment:generation strings eligible for retrieval, or an empty set for
        deleted/missing chats or documents that have not committed ready yet.
    """
    rows = (
        await session.execute(
            select(Attachment.id, Attachment.index_generation)
            .join(Conversation, Conversation.id == Attachment.conversation_id)
            .where(
                Conversation.id == conversation_id,
                Conversation.deleted_at.is_(None),
                Attachment.deleted_at.is_(None),
                Attachment.status == "ready",
                Attachment.index_generation.is_not(None),
            )
        )
    ).all()
    return {f"{row.id}:{row.index_generation}" for row in rows}


class ConversationRetriever:
    """Search only documents currently ready in the selected conversation.

    Args:
        sessions: Factory for fresh DB reads before/after vector search. A second
            read rejects documents deleted or replaced while the query was running.
        vector_index: Shared scoped Chroma index, with no unfiltered query API.
    """

    def __init__(
        self, sessions: async_sessionmaker[AsyncSession], vector_index: DocumentVectorIndex
    ) -> None:
        self.sessions = sessions
        self.vector_index = vector_index

    async def search(self, conversation_id: UUID, query: str, *, limit: int = 4) -> list[Document]:
        """Retrieve nearby chunks, checking both conversation and ready generation.

        Args:
            conversation_id: Chat UUID supplied by backend request/run context.
            query: Text to embed for similarity search, nonblank and at most 8000 characters.
            limit: Maximum returned chunks, between one and twenty.

        Returns:
            Eligible Documents with original filename metadata, never filesystem paths.
            An empty list requires no embedding call when no ready files exist.

        Raises:
            ValueError: If scope/query/limit is invalid.
            Exception: DB/embedding/vector errors propagate rather than broadening scope.
        """
        if (
            not isinstance(conversation_id, UUID)
            or not query.strip()
            or len(query) > 8000
            or not 1 <= limit <= 20
        ):
            raise ValueError("Invalid conversation retrieval request")
        async with self.sessions() as session:
            keys = await ready_document_keys(session, conversation_id)
        if not keys:
            return []
        documents = await run_file_operation(
            lambda: self.vector_index.search(
                query, conversation_id=conversation_id, ready_index_keys=sorted(keys), limit=limit
            )
        )
        async with self.sessions() as session:
            still_ready = await ready_document_keys(session, conversation_id)
        return [
            document
            for document in documents
            if (
                document.metadata.get("conversation_id") == str(conversation_id)
                and document.metadata.get("index_key") in still_ready
            )
        ]
