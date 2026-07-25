import asyncio

from src.adapters.reranker_adapter import RerankerAdapter
from src.config.settings import settings
from src.models.chunk import ChunkResult
from src.utils.logger import log_request_event
from src.utils.timing import stopwatch


class RerankService:
    """Reranks retrieval candidates using a cross-encoder, returns top-k.

    Receives RerankerAdapter via constructor injection.
    Sits between RetrievalService (Phase 2) and GenerationService (Phase 4)
    in the query pipeline.
    """

    def __init__(self, reranker: RerankerAdapter) -> None:
        self._reranker = reranker

    async def rerank(
        self,
        query: str,
        chunks: list[ChunkResult],
        request_id: str,
        role: str,
        final_k: int | None = None,
    ) -> list[ChunkResult]:
        """Score all candidates with the cross-encoder and return the top-k.

        Logs the full before/after score distribution so the eval harness can
        compute reranker lift (MRR, P@3 delta) without re-running retrieval.

        Args:
            query: user's natural-language question. Never logged (PHI risk).
            chunks: candidate chunks from RetrievalService (up to retrieval_top_k).
            request_id: trace ID for correlating log records across pipeline stages.
            role: authenticated user role for log correlation.
            final_k: how many top chunks to return. When None, uses
                settings.reranker_final_k. The two-axis router node passes a
                specificity-derived value here (point_lookup vs exploratory).

        Returns:
            Top `final_k` ChunkResult objects, sorted by reranker score descending.
            The .score field on each result carries the reranker score, not the
            original retrieval score.
        """
        if not chunks:
            return []

        # Resolve the cutoff: explicit override (two-axis router) or the default.
        cutoff = final_k if final_k is not None else settings.reranker_final_k

        with stopwatch() as elapsed_ms:
            # Cross-encoder inference is CPU-bound — offload to thread pool.
            # Returns all (chunk, reranker_score) pairs in input order, unsorted.
            scored: list[tuple[ChunkResult, float]] = await asyncio.to_thread(
                self._reranker.score, query, chunks
            )

            # Sort descending by reranker score. Policy decision — lives here, not in the adapter.
            scored.sort(key=lambda pair: pair[1], reverse=True)

        # Log full before/after score distribution for the eval harness.
        # Key by "source_document :: section_title" — NOT source_document alone, which
        # collapses multiple chunks from the same file (last-write-wins) and hides the
        # true per-chunk scores. Query text is never logged — it may contain PHI.
        def _cid(c: ChunkResult) -> str:
            return f"{c.source_document} :: {c.section_title}"

        log_request_event(
            request_id=request_id,
            role=role,
            event="rerank",
            latency_ms=elapsed_ms(),
            candidate_count=len(scored),
            final_k=cutoff,
            retrieval_scores={_cid(c): round(c.score, 4) for c, _ in scored},
            reranker_scores={_cid(c): round(s, 4) for c, s in scored},
            final_chunk_ids=[_cid(c) for c, _ in scored[:cutoff]],
        )

        # Replace retrieval score with reranker score before returning.
        # Downstream consumers (GenerationService) always receive the highest-quality
        # relevance signal in .score — not the RRF score from Phase 2.
        return [
            chunk.model_copy(update={"score": round(reranker_score, 4)})
            for chunk, reranker_score in scored[:cutoff]
        ]
