from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

from src.config.settings import settings


class SparseAdapter:
    """Wraps FastEmbed BM25 for sparse text embedding.

    Produces SparseVector objects compatible with Qdrant's native sparse index.
    Loads the model once at construction time.
    """

    def __init__(self) -> None:
        # Downloads model on first run, cached locally by FastEmbed.
        self._model = SparseTextEmbedding(model_name=settings.sparse_model_name)

    def embed(self, text: str) -> SparseVector:
        """Produce a sparse BM25 vector for a single text.

        Args:
            text: chunk text or query string.

        Returns:
            SparseVector with indices (token IDs) and values (BM25 weights).
        """
        results = list(self._model.embed([text]))
        sparse = results[0]
        return SparseVector(
            indices=sparse.indices.tolist(),
            values=sparse.values.tolist(),
        )

    def embed_batch(self, texts: list[str]) -> list[SparseVector]:
        """Produce sparse BM25 vectors for a list of texts.

        Args:
            texts: list of chunk texts. Used during ingestion.

        Returns:
            One SparseVector per input text, in the same order.
        """
        results = list(self._model.embed(texts))
        return [
            SparseVector(
                indices=r.indices.tolist(),
                values=r.values.tolist(),
            )
            for r in results
        ]
