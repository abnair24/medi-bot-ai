from sentence_transformers import SentenceTransformer

from src.config.settings import settings


class EmbeddingAdapter:
    """Wraps sentence-transformers for dense text embedding.

    Loads the model once at construction time. Thread-safe for read-only inference.
    """

    def __init__(self) -> None:
        # Model downloaded on first run, cached in ~/.cache/huggingface/
        self._model = SentenceTransformer(settings.dense_model_name)

    def embed(self, text: str) -> list[float]:
        """Encode a chunk text into a dense vector.

        Args:
            text: chunk text from document. No instruction prefix — chunks are indexed as-is.

        Returns:
            384-dimensional unit-normalised float vector.
        """
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Encode a user query with BGE retrieval instruction prefix.

        BGE models support asymmetric retrieval: the query gets an instruction prefix
        that steers the model toward matching document-style text rather than
        finding similar-sounding sentences.

        Args:
            query: user's natural-language question.

        Returns:
            384-dimensional unit-normalised float vector.
        """
        prefixed = f"Represent this question for searching relevant passages: {query}"
        vector = self._model.encode(prefixed, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts in one forward pass (more efficient than looping).

        Args:
            texts: list of chunk texts. Used during ingestion for batch throughput.

        Returns:
            List of 384-dim unit-normalised vectors, one per input text.
        """
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=settings.embedding_batch_size,
        )
        return vectors.tolist()
