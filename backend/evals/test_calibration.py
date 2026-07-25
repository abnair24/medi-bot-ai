"""Track 7 — judge calibration: does the LLM judge agree with human labels?

This is meta-evaluation — testing the test. We grade each answer two ways:
  1. the JUDGE (gpt-oss-20b) via a G-Eval correctness rubric -> correct/partial/wrong
  2. a HUMAN (you) in evals/golden/calibration_labels.json
then measure agreement with Cohen's kappa. Reported only — NOT a CI gate (there is no
fixed threshold for "the judge is trustworthy"; kappa is the evidence you interpret):
  kappa > 0.8 strong | 0.6-0.8 moderate | < 0.6 the judge (or rubric) is unreliable.

Workflow: run this once to generate answers + judge labels (written to
evals/results/calibration.json and printed). Fill in human_label for each case in
calibration_labels.json, then re-run — kappa is computed over the labeled cases.
"""

import asyncio
import json
from pathlib import Path

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from sklearn.metrics import cohen_kappa_score

from evals.eval_schemas import RetrievalCase
from evals.groq_judge import GroqJudge
from src.services.generation_service import GenerationService
from src.services.rerank_service import RerankService
from src.services.retrieval_service import RetrievalService

_FACTUAL = Path(__file__).parent / "golden" / "factual.json"
_LABELS = Path(__file__).parent / "golden" / "calibration_labels.json"
_RESULTS = Path(__file__).parent / "results" / "calibration.json"
_ROLE = "admin"
_VALID_LABELS = {"correct", "partial", "wrong"}


def _cases() -> list[RetrievalCase]:
    return [RetrievalCase(**r) for r in json.loads(_FACTUAL.read_text())]


def _human_labels() -> dict[str, str]:
    """id -> human_label, only for cases the human has actually labeled."""
    rows = json.loads(_LABELS.read_text())
    return {r["id"]: r["human_label"] for r in rows if r.get("human_label") in _VALID_LABELS}


def _score_to_label(score: float) -> str:
    """Map a G-Eval correctness score (0-1) to the same 3-way scale as human labels."""
    if score >= 0.80:
        return "correct"
    if score >= 0.40:
        return "partial"
    return "wrong"


def _kappa_verdict(kappa: float) -> str:
    if kappa > 0.80:
        return "strong — judge is reliable"
    if kappa >= 0.60:
        return "moderate — investigate disagreements"
    return "weak — judge/rubric unreliable, fix the rubric or switch judge model"


def test_judge_calibration(
    retrieval_service: RetrievalService,
    rerank_service: RerankService,
    generation_service: GenerationService,
    judge: GroqJudge,
) -> None:
    """Generate + judge-label every factual answer, then report Cohen's kappa vs human."""
    correctness = GEval(
        name="Correctness",
        criteria=(
            "Given the input question and the retrieval context, is the actual output a "
            "correct and complete answer that is grounded in the context? Penalize "
            "hallucinated facts, missing key information, and any wrong or altered values."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=judge,
        async_mode=False,
    )

    records: list[dict] = []
    for case in _cases():

        async def _pipeline() -> tuple[str, list[str]]:
            chunks = await retrieval_service.search(case.question, role=_ROLE, request_id=case.id)
            reranked = await rerank_service.rerank(
                case.question, chunks, request_id=case.id, role=_ROLE
            )
            answer, _ = await generation_service.generate(
                case.question, reranked, role=_ROLE, request_id=case.id
            )
            return answer, [c.text for c in reranked]

        answer, context = asyncio.run(_pipeline())
        tc = LLMTestCase(input=case.question, actual_output=answer, retrieval_context=context)
        correctness.measure(tc)
        records.append(
            {
                "id": case.id,
                "question": case.question,
                "answer": answer,
                "judge_score": correctness.score,
                "judge_label": _score_to_label(correctness.score or 0.0),
            }
        )

    # Persist to gitignored results/ so you can review answers while labeling.
    _RESULTS.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS.write_text(json.dumps(records, indent=2))

    print("\nJudge labels (fill human_label in calibration_labels.json to compute kappa):")
    for r in records:
        print(f"  {r['id']}: judge={r['judge_label']:>7} (score={r['judge_score']:.2f})  {r['question'][:55]}")

    # Cohen's kappa over the cases the human has labeled.
    human = _human_labels()
    paired = [(human[r["id"]], r["judge_label"]) for r in records if r["id"] in human]
    if len(paired) < 2:
        print(f"\n[calibration] {len(paired)} human labels — fill calibration_labels.json, then re-run for kappa.")
        return

    h, j = zip(*paired)
    kappa = cohen_kappa_score(list(h), list(j), labels=sorted(_VALID_LABELS))
    print(f"\nCohen's kappa (n={len(paired)}): {kappa:.3f} — {_kappa_verdict(kappa)}")
    print("Disagreements:")
    for r in records:
        if r["id"] in human and human[r["id"]] != r["judge_label"]:
            print(f"  {r['id']}: human={human[r['id']]} judge={r['judge_label']} (score={r['judge_score']:.2f})")
