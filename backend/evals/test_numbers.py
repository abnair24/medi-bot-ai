"""Track 3b — number match: drug doses / lab values / thresholds must be quoted EXACTLY.

Deterministic guardrail (guardrails.md): "Drug doses, lab values, and clinical thresholds
must be quoted exactly as they appear in the retrieved context. Never round, interpolate,
or convert units." This is checked by exact string compare — NOT the LLM judge, which
might treat '300 mg' and '0.3 g' as equivalent (a patient-safety failure for a clinical
tool). Use a judge only where correctness needs judgment; a dose has an exact oracle.

Each case runs the real pipeline (role-appropriate, so RBAC grants the source collection)
and asserts every expected value appears verbatim in the answer. Matching is
whitespace-normalized ('300mg' == '300 mg') but otherwise exact, so a rounded or
unit-converted value ('0.3 g', '300000 units') still fails.
"""

import asyncio
import json
import re
from pathlib import Path

import pytest

from evals.eval_schemas import NumberMatchCase
from src.services.generation_service import GenerationService
from src.services.rerank_service import RerankService
from src.services.retrieval_service import RetrievalService

_GOLDEN = Path(__file__).parent / "golden" / "numbers.json"


def _load_cases() -> list[NumberMatchCase]:
    raw = json.loads(_GOLDEN.read_text())
    return [NumberMatchCase(**r) for r in raw]


_CASES = _load_cases()


def _norm(s: str) -> str:
    """Collapse all whitespace and lowercase — tolerates spacing, not rounding/conversion."""
    return re.sub(r"\s+", "", s).lower()


@pytest.mark.parametrize("case", _CASES, ids=[c.id for c in _CASES])
def test_values_quoted_exactly(
    case: NumberMatchCase,
    retrieval_service: RetrievalService,
    rerank_service: RerankService,
    generation_service: GenerationService,
) -> None:
    """Every expected value must appear verbatim (whitespace-normalized) in the answer."""

    async def _pipeline() -> str:
        chunks = await retrieval_service.search(
            case.question, role=case.role, request_id=case.id
        )
        reranked = await rerank_service.rerank(
            case.question, chunks, request_id=case.id, role=case.role
        )
        answer, _ = await generation_service.generate(
            case.question, reranked, role=case.role, request_id=case.id
        )
        return answer

    answer = asyncio.run(_pipeline())
    answer_norm = _norm(answer)

    missing = [v for v in case.expected_values if _norm(v) not in answer_norm]
    assert not missing, (
        f"{case.id}: answer did not quote {missing} exactly. "
        f"A rounded/converted value is a guardrail violation. "
        f"Answer (first 300 chars): {answer[:300]!r}"
    )
