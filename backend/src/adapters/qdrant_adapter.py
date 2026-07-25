import hashlib
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    ScoredPoint,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.config.settings import settings
from src.models.chunk import ChunkMetadata, ChunkResult
from src.utils.exceptions import RetrievalError

# Names used for the two vector spaces in Qdrant.
# Defined as module constants — referenced in insert and search (Phase 2).
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class QdrantAdapter:
    """Qdrant client wrapper for MediBot vector storage.

    Phase 1: insert_batch() — store chunks with dense + sparse vectors.
    Phase 2: hybrid_search() — RBAC-filtered hybrid retrieval (added in Phase 2).
    """

    def __init__(self) -> None:
        self._client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self._collection = settings.collection_name
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection and payload indexes if they do not exist.

        Safe to call on every startup — no-op if collection already exists.
        Payload indexes on access_roles and collection are required for
        the RBAC pre-filter to run efficiently at query time.
        """
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    DENSE_VECTOR_NAME: VectorParams(
                        size=settings.dense_vector_size,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: SparseVectorParams(),
                },
            )

        # Always ensure indexes exist — idempotent, safe to call every startup.
        # Moving this outside the creation block guards against manually deleted indexes.
        # Without the access_roles index, RBAC filters degrade to O(n) payload scans —
        # functionally correct but silently 3-10x slower under load.
        for field in ("access_roles", "collection"):
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

    def insert_batch(
        self,
        chunks: list[ChunkMetadata],
        dense_vectors: list[list[float]],
        sparse_vectors: list[SparseVector],
    ) -> None:
        """Store a batch of chunks with their vectors in Qdrant.

        Uses deterministic point IDs so re-running ingest is idempotent —
        the same chunk always upserts to the same ID rather than duplicating.

        Args:
            chunks: validated ChunkMetadata objects (become the payload).
            dense_vectors: one 384-dim vector per chunk from EmbeddingAdapter.
            sparse_vectors: one SparseVector per chunk from SparseAdapter.

        Raises:
            RetrievalError: if the Qdrant upsert fails.
        """
        points = [
            PointStruct(
                id=_deterministic_id(chunk),
                vector={
                    DENSE_VECTOR_NAME: dense,
                    SPARSE_VECTOR_NAME: sparse,
                },
                payload=chunk.model_dump(),
            )
            for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors)
        ]

        try:
            self._client.upsert(
                collection_name=self._collection,
                points=points,
            )
        except Exception as e:
            raise RetrievalError(f"Qdrant batch insert failed: {e}") from e

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        role: str,
        limit: int,
    ) -> list[ChunkResult]:
        """Search Qdrant with dense+sparse hybrid fusion, filtered by role.

        Runs dense cosine search and sparse BM25 search as separate prefetch
        stages, then fuses results with Reciprocal Rank Fusion (RRF).
        The RBAC filter is evaluated server-side by Qdrant — Python never
        receives a chunk the requesting role cannot access.

        Args:
            dense_vector: 384-dim query embedding from EmbeddingAdapter.embed_query().
            sparse_vector: BM25 query vector from SparseAdapter.embed().
            role: authenticated user role; drives the access_roles pre-filter.
            limit: max candidates to return after fusion.

        Returns:
            Up to `limit` ChunkResult objects sorted by fused RRF score.

        Raises:
            RetrievalError: if Qdrant is unreachable or the query fails.
        """
        rbac_filter = Filter(
            must=[
                FieldCondition(
                    key="access_roles",
                    match=MatchAny(any=[role]),
                )
            ]
        )

        try:
            response = self._client.query_points(
                collection_name=self._collection,
                prefetch=[
                    Prefetch(
                        query=dense_vector,
                        using=DENSE_VECTOR_NAME,
                        limit=limit,
                    ),
                    Prefetch(
                        query=sparse_vector,
                        using=SPARSE_VECTOR_NAME,
                        limit=limit,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                query_filter=rbac_filter,
                limit=limit,
                with_payload=True,
            )
        except Exception as e:
            raise RetrievalError(f"Qdrant hybrid search failed: {e}") from e

        return [_point_to_chunk_result(p) for p in response.points]

    def dense_search(
        self,
        dense_vector: list[float],
        role: str,
        limit: int,
    ) -> list[ChunkResult]:
        """Search Qdrant with the dense vector only (no sparse fusion), RBAC-filtered.

        This is the dense-only baseline for the retrieval A/B eval (Track 1) — it lets
        the harness measure hybrid and hybrid+rerank lift against a pure semantic
        retriever. Same server-side RBAC pre-filter as hybrid_search.

        Args:
            dense_vector: 384-dim query embedding from EmbeddingAdapter.embed_query().
            role: authenticated user role; drives the access_roles pre-filter.
            limit: max candidates to return.

        Returns:
            Up to `limit` ChunkResult objects sorted by cosine similarity.

        Raises:
            RetrievalError: if Qdrant is unreachable or the query fails.
        """
        rbac_filter = Filter(
            must=[FieldCondition(key="access_roles", match=MatchAny(any=[role]))]
        )
        try:
            response = self._client.query_points(
                collection_name=self._collection,
                query=dense_vector,
                using=DENSE_VECTOR_NAME,
                query_filter=rbac_filter,
                limit=limit,
                with_payload=True,
            )
        except Exception as e:
            raise RetrievalError(f"Qdrant dense search failed: {e}") from e

        return [_point_to_chunk_result(p) for p in response.points]

    def scroll_all_payloads(self) -> list[dict]:
        """Return the raw, unvalidated payload of every point in the collection.

        Payloads are returned as plain dicts on purpose — a corpus audit (Track 6)
        must be able to INSPECT chunks that would fail ChunkResult validation (e.g.
        a missing citation) rather than raise on them. Routing through ChunkResult
        would crash on exactly the chunks the audit exists to find.

        Uses Qdrant scroll pagination to page through the entire collection.

        Raises:
            RetrievalError: if the Qdrant scroll fails.
        """
        payloads: list[dict] = []
        offset = None
        try:
            while True:
                points, offset = self._client.scroll(
                    collection_name=self._collection,
                    with_payload=True,
                    with_vectors=False,
                    limit=256,
                    offset=offset,
                )
                payloads.extend(point.payload or {} for point in points)
                if offset is None:  # None marks the last page
                    break
        except Exception as e:
            raise RetrievalError(f"Qdrant scroll failed: {e}") from e
        return payloads


def _deterministic_id(chunk: ChunkMetadata) -> str:
    """Produce a stable UUID from chunk content.

    Same source_document + section_title + text prefix always yields the same ID.
    Prevents duplicate points when ingest.py is re-run.
    """
    fingerprint = f"{chunk.source_document}:{chunk.section_title}:{chunk.text[:200]}"
    return str(uuid.UUID(hashlib.md5(fingerprint.encode()).hexdigest()))


def _point_to_chunk_result(point: ScoredPoint) -> ChunkResult:
    """Convert a Qdrant ScoredPoint into a validated ChunkResult.

    Validates via Pydantic on construction — raises ValidationError if the
    stored payload is missing a required field or has a type mismatch.
    That would indicate data corruption in Qdrant, not a code bug.
    """
    payload = point.payload or {}
    return ChunkResult(
        text=payload["text"],
        score=point.score,
        source_document=payload["source_document"],
        collection=payload["collection"],
        access_roles=payload["access_roles"],
        section_title=payload["section_title"],
        chunk_type=payload["chunk_type"],
    )
