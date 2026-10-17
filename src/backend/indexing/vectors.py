"""Store attempt-specific document chunks and require conversation filters for every search."""

from pathlib import Path
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from src.backend.indexing.config import IndexingSettings
from src.backend.indexing.errors import DocumentIndexingError
from src.backend.indexing.parser_process import DocumentSection
from src.backend.indexing.store import IndexingClaim

COLLECTION_NAME = "localmind_documents_v1"
CODE_LANGUAGES = {
    ".py": Language.PYTHON,
    ".cpp": Language.CPP,
    ".c": Language.CPP,
    ".h": Language.CPP,
    ".hpp": Language.CPP,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".html": Language.HTML,
}


def split_extracted_sections_into_document_chunks(
    sections: list[DocumentSection], file_name: str, settings: IndexingSettings
) -> list[Document]:
    """Split extracted text into bounded chunks, retaining original filename/page metadata.

    Args:
        sections: Bounded text sections produced by the parser subprocess.
        file_name: Original basename used for language-aware splitting and display.
        settings: Character-based chunk size, overlap and maximum chunk count.

    Returns:
        Nonblank LangChain Documents ready to embed, never including disk paths.

    Raises:
        DocumentIndexingError: If no text remains or chunk count exceeds the limit.
    """
    language = CODE_LANGUAGES.get(Path(file_name).suffix.lower())
    options = dict(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap, add_start_index=True
    )
    splitter = (
        RecursiveCharacterTextSplitter.from_language(language=language, **options)
        if language
        else RecursiveCharacterTextSplitter(**options)
    )
    chunks = []
    for section in sections:
        metadata = {"source": file_name}
        if section.page is not None:
            metadata["page"] = section.page
        for chunk in splitter.split_documents(
            [Document(page_content=section.text, metadata=metadata)]
        ):
            if chunk.page_content.strip():
                chunks.append(chunk)
                if len(chunks) > settings.max_chunks:
                    raise DocumentIndexingError("TOO_MANY_CHUNKS")
    if not chunks:
        raise DocumentIndexingError("EMPTY_DOCUMENT")
    return chunks


class DocumentVectorIndex:
    """Embed document chunks in a dedicated Chroma collection with explicit scope.

    Args:
        directory: Persistent local Chroma storage root. The old unscoped collection
            is left untouched but is never queried by this implementation.
        embeddings: Model used to embed documents and queries; tests inject a local fake.

    No public search method can run without conversation and ready-generation keys.
    The caller obtains those keys from PostgreSQL, never from LLM tool arguments.
    """

    def __init__(self, directory: Path, embeddings: Embeddings) -> None:
        import chromadb
        from chromadb.config import Settings

        self.embeddings = embeddings
        self.client = chromadb.PersistentClient(
            path=str(directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            COLLECTION_NAME, embedding_function=None
        )

    def generate_embeddings_and_upsert_chunk_vectors(
        self, claim: IndexingClaim, chunks: list[Document], start_index: int
    ) -> None:
        """Embed and upsert one bounded batch using stable IDs within this attempt.

        Args:
            claim: Attachment/conversation/generation identifiers for these vectors.
            chunks: Nonempty document chunks to embed together.
            start_index: Zero-based position of this batch in the full document.

        Returns:
            None after Chroma writes succeed. Repeating the same batch replaces its
            IDs rather than duplicating them. Library errors propagate to the worker.
        """
        index_key = f"{claim.attachment_id}:{claim.generation}"
        metadata = [
            {
                **chunk.metadata,
                "conversation_id": str(claim.conversation_id),
                "source_file_id": str(claim.attachment_id),
                "index_generation": str(claim.generation),
                "index_key": index_key,
            }
            for chunk in chunks
        ]
        text = [chunk.page_content for chunk in chunks]
        vectors = self.embeddings.embed_documents(text)
        self.collection.upsert(
            ids=[f"{index_key}:{start_index + offset}" for offset in range(len(chunks))],
            documents=text,
            metadatas=metadata,
            embeddings=vectors,
        )

    def search(
        self, text: str, *, conversation_id: UUID, ready_index_keys: list[str], limit: int
    ) -> list[Document]:
        """Search only vectors belonging to the chat and DB-approved ready attempts.

        Args:
            text: User query to embed for similarity search.
            conversation_id: Server-selected live chat UUID.
            ready_index_keys: Attachment:generation pairs read from ready, undeleted DB rows.
            limit: Maximum nearest chunks to return.

        Returns:
            Documents without private storage paths, or an empty list without
            embedding/querying Chroma when no ready documents are eligible.
        """
        if not ready_index_keys:
            return []
        matches = self.collection.query(
            query_embeddings=[self.embeddings.embed_query(text)],
            n_results=limit,
            where={
                "$and": [
                    {"conversation_id": str(conversation_id)},
                    {"index_key": {"$in": ready_index_keys}},
                ]
            },
            include=["documents", "metadatas"],
        )
        return [
            Document(page_content=content, metadata=metadata)
            for content, metadata in zip(
                matches["documents"][0],
                matches["metadatas"][0],
            )
        ]

    def delete_attempt(self, attachment_id: UUID, generation: UUID) -> None:
        """Delete only one attempt's vectors after confirmed failure or stale completion.

        Args:
            attachment_id: Document UUID that owns the vectors.
            generation: Attempt namespace to remove without affecting newer attempts.
        """
        self.collection.delete(where={"index_key": f"{attachment_id}:{generation}"})

    def delete_obsolete_attempts(self, attachment_id: UUID, keep_generation: UUID) -> None:
        """Remove old attempts after the current generation has safely committed ready.

        Args:
            attachment_id: Document whose previous/partial vectors can be removed.
            keep_generation: Current ready generation that must remain intact.
        """
        self.collection.delete(
            where={
                "$and": [
                    {"source_file_id": str(attachment_id)},
                    {"index_generation": {"$ne": str(keep_generation)}},
                ]
            }
        )

    def delete_document_vectors(self, conversation_id: UUID, document_id: UUID) -> None:
        """Remove every indexing generation of one document in the selected chat.

        Args:
            conversation_id: Chat UUID recorded on the deleted attachment.
            document_id: Attachment UUID whose partial and ready vectors are removed.

        Repeating deletion is safe. This does not remove PostgreSQL metadata or files.
        """
        self.collection.delete(
            where={
                "$and": [
                    {"conversation_id": str(conversation_id)},
                    {"source_file_id": str(document_id)},
                ]
            }
        )

        # The removed LangChain Chroma wrapper used its default "langchain" collection.
        # UUIDs are globally unique even though that legacy metadata has no chat field.
        collection_names = {
            getattr(collection, "name", collection) for collection in self.client.list_collections()
        }
        if "langchain" in collection_names:
            legacy = self.client.get_collection("langchain", embedding_function=None)
            legacy.delete(where={"source_file_id": str(document_id)})
