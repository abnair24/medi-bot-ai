"""Track 2 — RBAC red-team: prove the metadata pre-filter blocks restricted chunks
at the Qdrant query layer. Checks the RETURNED CHUNKS, not the LLM answer — a leaked
chunk in context is a failure even if the LLM would refuse. Run first, fail-fast.
"""

import json
from pathlib import Path

import pytest

from evals.eval_schemas import AdversarialCase
from src.services.retrieval_service import RetrievalService

_GOLDEN = Path(__file__).parent / "golden" / "adversarial.json"


def _load_cases() -> list[AdversarialCase]:
    """Load + validate every red-team case. Module-level (not a fixture) because
    @parametrize resolves at COLLECTION time, before fixtures exist."""
    raw = json.loads(_GOLDEN.read_text())
    return [AdversarialCase(**r) for r in raw]


_CASES = _load_cases()


@pytest.mark.parametrize("case", _CASES, ids=[c.id for c in _CASES])
@pytest.mark.asyncio
async def test_no_rbac_leak(case: AdversarialCase, retrieval_service: RetrievalService) -> None:
    chunks = await retrieval_service.search(
        query=case.question,
        role=case.attacker_role,
        request_id=case.id,
    )
    # Chunks whose collection matches the case's target_collection.
    hits = [c for c in chunks if c.collection == case.target_collection]

    if case.is_control:
        # Control: target_collection is ALLOWED — the filter must NOT over-block.
        # Assert PRESENCE, not an exact count: top-k and corpus contents vary, so we
        # can only guarantee at least one relevant chunk surfaces, never how many.
        assert hits, (
            f"OVER-BLOCK: {case.id} — role '{case.attacker_role}' received NO chunks "
            f"from allowed collection '{case.target_collection}'. Filter too restrictive."
        )
    else:
        # Attack: target_collection is BLOCKED — zero tolerance for leaks.
        # PHI-safe message: case id, attack type, role, collections, count — never the
        # question text or chunk text.
        assert not hits, (
            f"RBAC LEAK: {case.id} ({case.attack_type}) — role '{case.attacker_role}' "
            f"received {len(hits)} chunk(s) from BLOCKED collection "
            f"'{case.target_collection}'. Leaked from: {[c.collection for c in hits]}"
        )
