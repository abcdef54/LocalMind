"""Load one CPU embedding model and share one persistent document vector index."""

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from src.backend.constants import VECTOR_DB_PERSIST_DIR
from src.backend.indexing.vectors import DocumentVectorIndex


@lru_cache(maxsize=1)
def get_vector_embedding_model() -> Embeddings:
    """Load the CPU model that converts document/query text into numeric vectors.

    Returns:
        Shared MiniLM embedding model with normalized outputs. CPU execution keeps
        document indexing from competing with the LLM for GPU memory.

    Raises:
        Exception: If model download or initialization fails. Do not substitute
            fake vectors or mark unindexed documents ready when loading fails.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_document_vector_index() -> DocumentVectorIndex:
    """Open the persistent Chroma document index using the shared embedding model.

    Returns:
        Shared DocumentVectorIndex used by background indexing and scoped retrieval.
        It provides vector storage/search, not just text-to-vector model inference.

    Raises:
        Exception: If model initialization or opening the Chroma collection fails.
    """
    return DocumentVectorIndex(VECTOR_DB_PERSIST_DIR, get_vector_embedding_model())
