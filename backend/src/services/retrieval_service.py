import asyncio
from typing import TYPE_CHECKING

from src.adapters.embedding_adapter import EmbeddingAdapter
from src.adapters.qdrant_adapter import QdrantAdapter
from src.adapters.reranker_adapter import RerankerAdapter
from src.adapters.sparse_adapter import SparseAdapter
from src.config.access import ROLE_COLLECTIONS
from src.config.settings import settings
from src.models.chunk import ChunkResult
from src.utils.exceptions import RBACError
from src.utils.logger import log_request_event
from src.utils.timing import stopwatch

if TYPE_CHECKING:
    from src.services.hyde_service import HyDEService


class RetrievalService:
    """Orchestrates hybrid RAG retrieval: validate role → embed query → Qdrant search.

    Receives adapters via constructor injection — never constructs them internally.
    The Phase 3 reranker consumes the list[ChunkResult] this returns.

    Optional HyDE expansion: when a HyDEService is injected AND settings.hyde_enabled
    is True, the dense embedding uses a hypothetical answer document instead of the raw
    query. The sparse (BM25) path always uses the original query — keyword matching must
    stay literal.
    """

    def __init__(
        self,
        qdrant: QdrantAdapter,
        embedder: EmbeddingAdapter,
        sparse: SparseAdapter,
        hyde: "HyDEService | None" = None,
        reranker: RerankerAdapter | None = None,
    ) -> None:
        self._qdrant = qdrant
        self._embedder = embedder
        self._sparse = sparse
        self._hyde = hyde
        # Used only by restricted_matches() for a cross-encoder relevance comparison
        # (RRF scores are rank-based, not relevance-based, so they can't be compared
        # across two searches). Optional — when absent, the access probe is skipped.
        self._reranker = reranker

    async def search(
        self,
        query: str,
        role: str,
        request_id: str,
    ) -> list[ChunkResult]:
        """Run hybrid retrieval for the given query, enforcing RBAC via pre-filter.

        Dense and sparse embeddings are generated concurrently (both are independent
        CPU-bound calls on local models). The Qdrant RBAC pre-filter is applied
        server-side — Python never receives a chunk outside the role's access list.

        HyDE path: when enabled, the dense embedding uses an LLM-generated hypothetical
        answer instead of the raw query. BM25 always receives the original query.

        Args:
            query: user's natural-language question. Never logged (may contain PHI).
            role: authenticated user role from the JWT payload.
            request_id: trace ID for correlating log records across pipeline stages.

        Returns:
            Up to settings.retrieval_top_k ChunkResult objects, sorted by fused RRF score.

        Raises:
            RBACError: if role is not in ROLE_COLLECTIONS.
            RetrievalError: if Qdrant is unreachable or the query fails.
        """
        if role not in ROLE_COLLECTIONS:
            raise RBACError(
                f"Role '{role}' is not recognised. "
                f"Valid roles: {list(ROLE_COLLECTIONS)}"
            )

        with stopwatch() as elapsed_ms:
            # HyDE expansion: generate a hypothetical answer for the dense path.
            # BM25 always uses the original query — keyword matching must stay literal.
            dense_query = query
            if self._hyde is not None and settings.hyde_enabled:
                dense_query = await self._hyde.expand(query, request_id)

            # Run dense and sparse embedding concurrently — both are independent.
            # asyncio.to_thread offloads CPU-bound model calls to the thread pool,
            # keeping the FastAPI event loop free for other incoming requests.
            dense_vector, sparse_vector = await asyncio.gather(
                asyncio.to_thread(self._embedder.embed_query, dense_query),
                asyncio.to_thread(self._sparse.embed, query),  # always original query
            )

            chunks = self._qdrant.hybrid_search(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                role=role,
                limit=settings.retrieval_top_k,
            )

        # Log operational metadata only — no query text, no chunk text (PHI risk).
        log_request_event(
            request_id=request_id,
            role=role,
            event="retrieval",
            latency_ms=elapsed_ms(),
            chunk_count=len(chunks),
            chunk_ids=[c.source_document for c in chunks],
            scores=[round(c.score, 4) for c in chunks],
            hyde_active=self._hyde is not None and settings.hyde_enabled,
        )

        return chunks

    async def restricted_matches(
        self,
        query: str,
        role: str,
        request_id: str,
        user_chunks: list[ChunkResult],
    ) -> list[str]:
        """Detect an ACCESS-scoped miss: does the query's BEST home lie in a collection the
        role can't see, and beat everything the role CAN see?

        Distinguishes "topic is behind RBAC" (→ show a denial) from "topic isn't in the
        corpus at all" (→ show 'no relevant protocol'). Runs one unrestricted (admin-scope)
        probe and compares its top hit against the user's own best RBAC-filtered score. Only
        fires when a restricted collection clearly wins — so a weak incidental match on an
        absent topic does NOT trigger a false denial.

        Security: restricted CONTENT never leaves this method — only collection names, and
        only to choose the denial message. The user's actual results stay pre-filtered.
        """
        if self._reranker is None:
            return []
        accessible = set(ROLE_COLLECTIONS.get(role, []))

        dense_vector, sparse_vector = await asyncio.gather(
            asyncio.to_thread(self._embedder.embed_query, query),
            asyncio.to_thread(self._sparse.embed, query),
        )
        probe = self._qdrant.hybrid_search(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            role="admin",  # admin sees every collection — an unrestricted probe
            limit=settings.retrieval_top_k,
        )
        if not probe:
            return []

        # Cross-encoder gives an ABSOLUTE relevance score (unlike RRF's rank-based score),
        # so we can compare "how relevant is the best restricted chunk" against "how relevant
        # is the best chunk the caller CAN see". Score BOTH sets with the same encoder.
        probe_scored = await asyncio.to_thread(self._reranker.score, query, probe)
        user_scored = (
            await asyncio.to_thread(self._reranker.score, query, user_chunks)
            if user_chunks
            else []
        )

        restricted_hits = [(c, s) for c, s in probe_scored if c.collection not in accessible]
        best_accessible = max((s for _, s in user_scored), default=float("-inf"))

        restricted: list[str] = []
        if restricted_hits:
            best_chunk, best_score = max(restricted_hits, key=lambda pair: pair[1])
            # Access miss: the query's answer is genuinely MORE relevant in a collection the
            # caller can't see. On an absent topic everything scores low, so restricted never
            # wins → no false denial. The margin keeps ties out.
            if best_score > best_accessible + settings.rbac_probe_margin:
                restricted.append(best_chunk.collection)

        log_request_event(
            request_id=request_id,
            role=role,
            event="rbac_access_probe",
            restricted_matches=restricted,
        )
        return restricted
