"""Track 3 — faithfulness: are the answer's claims supported by the retrieved context?

This is the LLM-as-judge track. Unlike retrieval/RBAC/routing there is no deterministic
oracle — "is every claim grounded in the context?" needs judgment, so DeepEval's
FaithfulnessMetric decomposes the answer into atomic claims and checks each against the
retrieval context, scoring supported/total. The judge (gpt-oss-20b) is a DIFFERENT model
from the generator (llama-3.3-70b) to avoid self-preference bias.

Runs the real pipeline: retrieval -> rerank -> generation, using role='admin' so the RBAC
filter never starves the context (faithfulness is about grounding, not access control).

COST: every case is generator tokens (answer) + judge tokens (claim decomposition). The
free tier is 12k tokens/min, so run selectively with `-k F-002` during development; a full
run must be paced. Each test is sync — it runs the async pipeline via asyncio.run to
completion, THEN calls metric.measure(async_mode=False) so there is no nested event loop.
"""

import asyncio
import json
from pathlib import Path

import pytest
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from evals.eval_config import FAITHFULNESS_THRESHOLD
from evals.eval_schemas import RetrievalCase
from evals.groq_judge import GroqJudge
from src.services.generation_service import GenerationService
from src.services.rerank_service import RerankService
from src.services.retrieval_service import RetrievalService

_GOLDEN = Path(__file__).parent / "golden" / "factual.json"
_ROLE = "admin"  # full access — never starve the context; grounding is what we measure


def _load_cases() -> list[RetrievalCase]:
    """Reuse the factual golden questions (see test_rbac for the module-level pattern)."""
    raw = json.loads(_GOLDEN.read_text())
    return [RetrievalCase(**r) for r in raw]


_CASES = _load_cases()


@pytest.mark.parametrize("case", _CASES, ids=[c.id for c in _CASES])
def test_answer_is_faithful_to_context(
    case: RetrievalCase,
    retrieval_service: RetrievalService,
    rerank_service: RerankService,
    generation_service: GenerationService,
    judge: GroqJudge,
) -> None:
    """The generated answer's claims must be grounded in the reranked context."""

    async def _pipeline() -> tuple[str, list[str]]:
        chunks = await retrieval_service.search(
            case.question, role=_ROLE, request_id=case.id
        )
        reranked = await rerank_service.rerank(
            case.question, chunks, request_id=case.id, role=_ROLE
        )
        answer, _ = await generation_service.generate(
            case.question, reranked, role=_ROLE, request_id=case.id
        )
        return answer, [c.text for c in reranked]

    # Run the async pipeline to completion first, so the event loop is closed before
    # DeepEval's (sync) measure() runs — avoids "asyncio.run() from a running loop".
    answer, context = asyncio.run(_pipeline())

    test_case = LLMTestCase(
        input=case.question,
        actual_output=answer,
        retrieval_context=context,
    )
    metric = FaithfulnessMetric(
        threshold=FAITHFULNESS_THRESHOLD,
        model=judge,
        async_mode=False,  # serialize the judge's internal calls — stays under 12k TPM
        include_reason=True,
    )
    metric.measure(test_case)

    assert metric.score is not None and metric.score >= FAITHFULNESS_THRESHOLD, (
        f"{case.id} faithfulness {metric.score} < {FAITHFULNESS_THRESHOLD}. "
        f"Judge reason: {metric.reason}"
    )
