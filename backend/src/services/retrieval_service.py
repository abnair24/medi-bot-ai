import asyncio
from typing import TYPE_CHECKING

from src.adapters.embedding_adapter import EmbeddingAdapter
from src.adapters.qdrant_adapter import QdrantAdapter
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
    ) -> None:
        self._qdrant = qdrant
        self._embedder = embedder
        self._sparse = sparse
        self._hyde = hyde

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
    ) -> list[str]:
        """Probe which collections OUTSIDE the role's access strongly match the query.

        Used to distinguish an ACCESS-scoped miss from a genuine no-content miss when the
        RBAC-filtered retrieval returns nothing relevant. Runs an unrestricted (admin-scope)
        search and returns the collections that appear in the top results but are NOT in the
        caller's access list.

        Security: the restricted CONTENT is never returned to the user or the LLM — only the
        collection *names* leave this method, and only to decide whether to show an RBAC
        denial. This is not a post-filter of the user's results (those stay pre-filtered).
        """
        accessible = set(ROLE_COLLECTIONS.get(role, []))

        dense_vector, sparse_vector = await asyncio.gather(
            asyncio.to_thread(self._embedder.embed_query, query),
            asyncio.to_thread(self._sparse.embed, query),
        )
        # "admin" sees every collection — an unrestricted probe. Only the top few results
        # matter: if the query's best home is a collection the caller can't see, that's an
        # access miss. Keep top-3 to avoid weak/incidental matches triggering a false denial.
        probe = self._qdrant.hybrid_search(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            role="admin",
            limit=settings.retrieval_top_k,
        )
        restricted: list[str] = []
        for chunk in probe[:3]:
            if chunk.collection not in accessible and chunk.collection not in restricted:
                restricted.append(chunk.collection)

        log_request_event(
            request_id=request_id,
            role=role,
            event="rbac_access_probe",
            restricted_matches=restricted,
        )
        return restricted
