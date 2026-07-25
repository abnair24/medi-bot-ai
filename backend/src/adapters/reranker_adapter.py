from sentence_transformers import CrossEncoder

from src.config.settings import settings
from src.models.chunk import ChunkResult


class RerankerAdapter:
    """Wraps sentence-transformers CrossEncoder for joint (query, chunk) relevance scoring.

    The cross-encoder sees query and chunk text together in one forward pass —
    more accurate than bi-encoder cosine similarity but too slow for full-corpus search.
    Intended for two-stage retrieval: run after QdrantAdapter narrows to top-k candidates.

    Loads model once at construction. Thread-safe for read-only inference.
    """

    def __init__(self) -> None:
        # ~80MB download on first run, cached in ~/.cache/huggingface/
        self._model = CrossEncoder(settings.reranker_model_name)

    def score(
        self,
        query: str,
        chunks: list[ChunkResult],
    ) -> list[tuple[ChunkResult, float]]:
        """Score all (query, chunk) pairs and return them with reranker scores.

        Returns all input chunks paired with their scores — does NOT sort or
        truncate. Top-k selection is the service layer's responsibility.
        All scores are returned so the service can log the full before/after
        distribution for the retrieval eval harness.

        Args:
            query: user's natural-language question.
            chunks: candidate chunks from Qdrant hybrid search (up to retrieval_top_k).

        Returns:
            List of (ChunkResult, reranker_score) in the same order as input.
            Scores are raw logits (not calibrated probabilities) — compare
            relative ordering, not absolute values.

        Raises:
            RuntimeError: if model inference fails (propagated from CrossEncoder).
        """
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        # predict() processes all pairs in one forward pass — more efficient than looping.
        # Returns numpy.ndarray; .tolist() converts to plain Python floats before
        # the adapter boundary.
        scores: list[float] = self._model.predict(pairs).tolist()
        return list(zip(chunks, scores))
