"""Track 12 — intent-gate accuracy (multiturn admission control).

The intent gate runs BEFORE the router and decides WHETHER a turn is answerable:
  - resolved  → self-contained or pinnable to one prior topic → proceed to retrieval/SQL.
  - ambiguous → subject-less follow-up with >=2 candidate topics → clarify (HITL).
  - empty     → contentless with no recoverable prior intent → refuse, never retrieve.

This is the guard for the confidently-wrong answers seen on bare inputs ("EXPLAIN",
"one more") and on ambiguous follow-ups. It runs every golden case through the SAME
IntentService.resolve() the production graph uses (resolve_intent_node), aggregates into
a 3x3 confusion matrix, and gates on overall accuracy.

Two gates:
  - Overall accuracy >= INTENT_ACCURACY_THRESHOLD (resolved/ambiguous/empty across all).
  - First-turn (empty-history) accuracy == INTENT_FIRST_TURN_THRESHOLD (1.0): those cases
    are decided deterministically (no LLM), so any miss is a hard logic bug.

Golden queries are synthetic — safe to print in reports. No PHI.
"""

import asyncio
import json
from pathlib import Path

from evals.eval_config import INTENT_ACCURACY_THRESHOLD, INTENT_FIRST_TURN_THRESHOLD
from evals.eval_schemas import IntentCase
from src.models.schemas import IntentState
from src.services.intent_service import IntentService

import pytest

_GOLDEN = Path(__file__).parent / "golden" / "intent.json"
_STATES: tuple[IntentState, ...] = ("resolved", "ambiguous", "empty")

# (case, predicted_state) — cached once, reused by every report/assert.
Prediction = tuple[IntentCase, IntentState]


def _load_cases() -> list[IntentCase]:
    raw = json.loads(_GOLDEN.read_text())
    return [IntentCase(**r) for r in raw]


_CASES = _load_cases()


@pytest.fixture(scope="session")
def intent_predictions(intent_service: IntentService) -> list[Prediction]:
    """Resolve every case once. First-turn cases skip the LLM; history cases are paced.

    Pacing mirrors the routing track: sequential with a small delay to stay under the
    free-tier tokens-per-minute cap. Only the history cases hit the API.
    """
    _PACING_SECONDS = 1.6

    async def _run() -> list[Prediction]:
        results: list[Prediction] = []
        for case in _CASES:
            history = [t.model_dump() for t in case.history]
            verdict = await intent_service.resolve(
                question=case.question,
                history=history,
                request_id=case.id,
                role="admin",
            )
            results.append((case, verdict.state))
            if history:  # only history cases made an API call worth pacing
                await asyncio.sleep(_PACING_SECONDS)
        return results

    return asyncio.run(_run())


def _accuracy(preds: list[Prediction]) -> float:
    if not preds:
        return 0.0
    correct = sum(1 for c, p in preds if p == c.expected_state)
    return correct / len(preds)


def _confusion(preds: list[Prediction]) -> dict[IntentState, dict[IntentState, int]]:
    m: dict[IntentState, dict[IntentState, int]] = {
        a: {p: 0 for p in _STATES} for a in _STATES
    }
    for case, pred in preds:
        m[case.expected_state][pred] += 1
    return m


def _report(preds: list[Prediction]) -> str:
    m = _confusion(preds)
    lines = [
        "",
        "Intent-gate confusion matrix (rows = expected, cols = predicted):",
        f"{'':>12} {'resolved':>10} {'ambiguous':>11} {'empty':>7}",
    ]
    for a in _STATES:
        lines.append(
            f"{a:>12} {m[a]['resolved']:>10} {m[a]['ambiguous']:>11} {m[a]['empty']:>7}"
        )
    lines.append("")
    lines.append(f"Overall accuracy: {_accuracy(preds):.3f}  (n={len(preds)})")

    first_turn = [p for p in preds if not p[0].history]
    multiturn = [p for p in preds if p[0].history]
    if first_turn:
        lines.append(f"  first-turn (deterministic): {_accuracy(first_turn):.3f} "
                     f"({len(first_turn)} cases)")
    if multiturn:
        lines.append(f"  multiturn (LLM): {_accuracy(multiturn):.3f} ({len(multiturn)} cases)")

    misses = [(c, p) for c, p in preds if p != c.expected_state]
    if misses:
        lines.append("  Misclassifications:")
        lines += [
            f"    {c.id} [{'first-turn' if not c.history else 'multiturn'}]: "
            f"expected {c.expected_state} -> predicted {p}"
            for c, p in misses
        ]
    return "\n".join(lines)


def test_first_turn_intent_is_perfect(intent_predictions: list[Prediction]) -> None:
    """First-turn (empty-history) verdicts are deterministic — they must all be correct."""
    first_turn = [p for p in intent_predictions if not p[0].history]
    accuracy = _accuracy(first_turn)
    report = _report(intent_predictions)
    print(report)
    assert accuracy >= INTENT_FIRST_TURN_THRESHOLD, (
        f"First-turn intent accuracy {accuracy:.3f} < {INTENT_FIRST_TURN_THRESHOLD} — "
        f"a deterministic-path logic bug.\n{report}"
    )


def test_intent_accuracy_meets_threshold(intent_predictions: list[Prediction]) -> None:
    """Overall intent-gate accuracy must clear the threshold. Prints the full matrix."""
    accuracy = _accuracy(intent_predictions)
    report = _report(intent_predictions)
    print(report)
    assert accuracy >= INTENT_ACCURACY_THRESHOLD, (
        f"Intent accuracy {accuracy:.3f} < {INTENT_ACCURACY_THRESHOLD}.\n{report}"
    )
