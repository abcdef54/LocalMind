"""Provide deterministic embeddings so real Chroma tests never download or run a model."""

from langchain_core.embeddings import Embeddings


class TinyEmbeddings(Embeddings):
    """Map text to small deterministic vectors using character counts.

    No network or GPU is used. These vectors test storage/filtering and retries,
    not the semantic quality of the production MiniLM model.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one fixed-size vector for each supplied document string.

        Args:
            texts: Chunk strings in their embedding-batch order.

        Returns:
            Numeric vectors with matching count/order, using the query mapping below.
        """
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return character-count features for a query or document chunk.

        Args:
            text: Input whose characters determine a nonzero, four-dimensional vector.

        Returns:
            Four floats sufficient for exercising actual Chroma nearest-neighbor queries.
        """
        return [
            1.0,
            float(text.lower().count("a")),
            float(text.lower().count("e")),
            len(text) / 100,
        ]
