"""Track 1 — retrieval quality A/B: dense-only vs hybrid vs hybrid+rerank.

Measures whether the answer-bearing chunks come back in the top ranks, and how much
hybrid (dense+BM25 RRF) and the cross-encoder reranker improve over a pure semantic
baseline. Deterministic — local embeddings + Qdrant + local cross-encoder, no LLM.

Metrics (numpy, no framework):
- P@3  : fraction of queries with a relevant chunk in the top 3 (a.k.a. success@3 —
         each question has one answer section, so strict precision/3 would cap at 0.33;
         the field measures hit-rate, and EVAL_PLAN's 0.70 gate assumes this reading).
- MRR  : mean of 1 / rank-of-first-relevant over all queries (0 if none retrieved).

Relevance: a retrieved chunk is relevant if its (source_document, section_title) matches
any entry in the case's `expected`. Role = 'admin' so the RBAC pre-filter never removes a
relevant chunk — this isolates retrieval quality from access control (Track 2's concern).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from evals.eval_config import RETRIEVAL_P3_THRESHOLD
from evals.eval_schemas import RetrievalCase
from src.adapters.embedding_adapter import EmbeddingAdapter
from src.adapters.qdrant_adapter import QdrantAdapter
from src.adapters.reranker_adapter import RerankerAdapter
from src.adapters.sparse_adapter import SparseAdapter
from src.config.settings import settings
from src.models.chunk import ChunkResult

_GOLDEN = Path(__file__).parent / "golden" / "factual.json"
_ROLE = "admin"  # full access — isolates retrieval quality from RBAC filtering
_METHODS = ("dense", "hybrid", "hybrid+rerank")

# A chunk's relevance identity: (source_document, section_title).
RelevanceKey = tuple[str, str]


def _load_cases() -> list[RetrievalCase]:
    """Load + validate every factual case. Module-level (see test_rbac for why)."""
    raw = json.loads(_GOLDEN.read_text())
    return [RetrievalCase(**r) for r in raw]


_CASES = _load_cases()


def _keys(chunks: list[ChunkResult]) -> list[RelevanceKey]:
    """Ranked list of relevance keys from a ranked list of chunks."""
    return [(c.source_document, c.section_title) for c in chunks]


def _first_relevant_rank(ranked: list[RelevanceKey], expected: set[RelevanceKey]) -> int:
    """1-indexed rank of the first relevant chunk, or 0 if none present."""
    for rank, key in enumerate(ranked, start=1):
        if key in expected:
            return rank
    return 0


def _p_at_3(ranked: list[RelevanceKey], expected: set[RelevanceKey]) -> float:
    """1.0 if any relevant chunk is in the top 3, else 0.0 (success@3)."""
    return 1.0 if any(key in expected for key in ranked[:3]) else 0.0


def _reciprocal_rank(ranked: list[RelevanceKey], expected: set[RelevanceKey]) -> float:
    rank = _first_relevant_rank(ranked, expected)
    return 1.0 / rank if rank else 0.0


@pytest.fixture(scope="session")
def retrieval_runs(
    qdrant: QdrantAdapter,
    embedder: EmbeddingAdapter,
    sparse: SparseAdapter,
    reranker: RerankerAdapter,
) -> dict[str, dict[str, list[RelevanceKey]]]:
    """Run all three retrieval methods for every case once; cache ranked key lists.

    Returns: case_id -> method -> ranked list of RelevanceKey.
    """
    top_k = settings.retrieval_top_k
    runs: dict[str, dict[str, list[RelevanceKey]]] = {}

    for case in _CASES:
        dense_vec = embedder.embed_query(case.question)
        sparse_vec = sparse.embed(case.question)

        dense_hits = qdrant.dense_search(dense_vec, role=_ROLE, limit=top_k)
        hybrid_hits = qdrant.hybrid_search(dense_vec, sparse_vec, role=_ROLE, limit=top_k)

        # Rerank the hybrid candidates with the cross-encoder, sort by score desc.
        scored = reranker.score(case.question, hybrid_hits)
        reranked = [chunk for chunk, _ in sorted(scored, key=lambda p: p[1], reverse=True)]

        runs[case.id] = {
            "dense": _keys(dense_hits),
            "hybrid": _keys(hybrid_hits),
            "hybrid+rerank": _keys(reranked),
        }
    return runs


def _metrics(runs: dict[str, dict[str, list[RelevanceKey]]]) -> dict[str, dict[str, float]]:
    """method -> {'p_at_3': mean, 'mrr': mean} over all cases."""
    result: dict[str, dict[str, float]] = {}
    for method in _METHODS:
        p3s, rrs = [], []
        for case in _CASES:
            expected = {(rc.source_document, rc.section_title) for rc in case.expected}
            ranked = runs[case.id][method]
            p3s.append(_p_at_3(ranked, expected))
            rrs.append(_reciprocal_rank(ranked, expected))
        result[method] = {"p_at_3": float(np.mean(p3s)), "mrr": float(np.mean(rrs))}
    return result


def _report(runs: dict[str, dict[str, list[RelevanceKey]]]) -> str:
    m = _metrics(runs)
    lines = [
        "",
        "Retrieval A/B (n=%d queries):" % len(_CASES),
        f"{'method':>14} {'P@3':>7} {'MRR':>7}",
    ]
    for method in _METHODS:
        lines.append(f"{method:>14} {m[method]['p_at_3']:>7.3f} {m[method]['mrr']:>7.3f}")

    # Per-query first-relevant rank for each method — surfaces which queries dense misses.
    lines.append("  first-relevant rank per query (0 = not in top-%d):" % settings.retrieval_top_k)
    for case in _CASES:
        expected = {(rc.source_document, rc.section_title) for rc in case.expected}
        ranks = {mth: _first_relevant_rank(runs[case.id][mth], expected) for mth in _METHODS}
        tag = "kw" if case.keyword_heavy else "  "
        lines.append(
            f"    {case.id} [{tag}] "
            f"dense={ranks['dense']} hybrid={ranks['hybrid']} rerank={ranks['hybrid+rerank']}"
        )
    return "\n".join(lines)


def test_hybrid_rerank_meets_p3_threshold(
    retrieval_runs: dict[str, dict[str, list[RelevanceKey]]],
) -> None:
    """hybrid+rerank P@3 must clear the gate. Dense-only is baseline (reported, not gated)."""
    metrics = _metrics(retrieval_runs)
    report = _report(retrieval_runs)
    print(report)
    p3 = metrics["hybrid+rerank"]["p_at_3"]
    assert p3 >= RETRIEVAL_P3_THRESHOLD, (
        f"hybrid+rerank P@3 {p3:.3f} < {RETRIEVAL_P3_THRESHOLD}.\n{report}"
    )
