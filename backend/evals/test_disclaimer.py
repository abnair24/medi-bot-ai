"""Track 13 — disclaimer keying + relevance floor (deterministic, no LLM, no network).

Guards two of the intent-gate fixes at the generation boundary:

1. Content-keyed disclaimer. `_clinical_is_central` decides whether the clinical
   disclaimer is appended. The bug it fixes: keying on "any clinical/nursing chunk
   retrieved" stamped the clinical disclaimer onto equipment/HR answers whenever a single
   stray nursing chunk (e.g. hand hygiene) survived into the context. The rule under test:
   clinical is central iff it grounds the TOP chunk OR is a strict plurality — a lone
   low-ranked stray does not trigger it.

2. Relevance floor. `_keep_above_floor` drops reranked chunks below
   settings.rerank_relevance_floor — the deterministic backstop under the LLM sentinel
   that kills keyword-collision noise before it reaches the prompt.

Both are pure functions, so this track is fast and runs first-class in CI with zero ML
deps — it must be PERFECT (DISCLAIMER_ACCURACY_THRESHOLD = 1.0) because a misplaced or
missing clinical disclaimer is a safety-guardrail regression.
"""

import json
from pathlib import Path

from evals.eval_config import DISCLAIMER_ACCURACY_THRESHOLD
from evals.eval_schemas import DisclaimerCase
from src.config.access import get_roles_for_collection
from src.config.settings import settings
from src.models.chunk import ChunkResult
from src.models.schemas import Collection
from src.services.generation_service import _clinical_is_central, _keep_above_floor

_GOLDEN = Path(__file__).parent / "golden" / "disclaimer.json"


def _load_cases() -> list[DisclaimerCase]:
    """Load + validate every disclaimer case (module-level, like the other tracks)."""
    raw = json.loads(_GOLDEN.read_text())
    return [DisclaimerCase(**r) for r in raw]


_CASES = _load_cases()


def _chunk(collection: Collection, score: float) -> ChunkResult:
    """Build a minimal valid ChunkResult stub for a given collection and score.

    access_roles is derived from the collection so ChunkResult's own validator passes —
    the disclaimer logic only reads .collection and .score (and list order).
    """
    return ChunkResult(
        text="stub",
        score=score,
        source_document="stub.pdf",
        collection=collection,
        access_roles=get_roles_for_collection(collection),
        section_title="stub section",
        chunk_type="text",
    )


def _chunks_in_rank_order(collections: list[Collection]) -> list[ChunkResult]:
    """Build chunks in score order — index 0 is the top-scored chunk (highest score).

    _clinical_is_central relies on chunks[0] being the top chunk (rerank_service returns
    them sorted descending), so we assign strictly decreasing scores by position.
    """
    n = len(collections)
    return [_chunk(coll, score=float(n - i)) for i, coll in enumerate(collections)]


def test_disclaimer_keying_matches_golden() -> None:
    """_clinical_is_central must match the golden label on EVERY case (fail-closed)."""
    misses: list[str] = []
    for case in _CASES:
        chunks = _chunks_in_rank_order(case.chunk_collections)
        got = _clinical_is_central(chunks)
        if got != case.expected_central:
            misses.append(
                f"  {case.id}: {case.chunk_collections} -> got central={got}, "
                f"expected {case.expected_central}  ({case.description})"
            )

    accuracy = (len(_CASES) - len(misses)) / len(_CASES)
    report = "\n".join(misses) if misses else "  (all cases correct)"
    print(f"\nDisclaimer keying accuracy: {accuracy:.3f} (n={len(_CASES)})\n{report}")
    assert accuracy >= DISCLAIMER_ACCURACY_THRESHOLD, (
        f"Disclaimer keying accuracy {accuracy:.3f} < {DISCLAIMER_ACCURACY_THRESHOLD}.\n"
        f"Mis-keyed cases (each is a safety-guardrail regression):\n{report}"
    )


def test_when_bug_stray_nursing_chunk_does_not_trigger_disclaimer() -> None:
    """Regression guard for the exact 'WHEN' transcript: equipment/HR answer + stray nursing."""
    chunks = _chunks_in_rank_order(["equipment", "equipment", "nursing"])
    assert _clinical_is_central(chunks) is False


def test_relevance_floor_drops_below_and_keeps_at_or_above() -> None:
    """_keep_above_floor keeps chunks >= floor and drops those strictly below it."""
    floor = settings.rerank_relevance_floor
    chunks = [
        _chunk("general", score=floor + 2.0),   # keep
        _chunk("general", score=floor),         # keep (boundary is inclusive)
        _chunk("billing", score=floor - 0.01),  # drop (just below)
        _chunk("billing", score=floor - 5.0),   # drop (well below)
    ]
    kept = _keep_above_floor(chunks)
    assert len(kept) == 2
    assert all(c.score >= floor for c in kept)


def test_relevance_floor_all_noise_returns_empty() -> None:
    """When every chunk is keyword-collision noise, the floor removes them all."""
    floor = settings.rerank_relevance_floor
    chunks = [_chunk("billing", score=floor - 1.0), _chunk("equipment", score=floor - 3.0)]
    assert _keep_above_floor(chunks) == []
