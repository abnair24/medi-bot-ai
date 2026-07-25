"""Track 5 — routing accuracy: does the semantic router classify ANALYTICAL vs
KNOWLEDGE questions correctly? The router is an LLM-as-classifier, so this is a
classification eval: run every golden case through RouterService, aggregate into a
2x2 confusion matrix, and gate on overall accuracy (>= ROUTING_ACCURACY_THRESHOLD).

The eval calls `classify_detailed()` — the SAME two-axis method the production graph
uses (router_node in backend/src/graph/nodes.py) — so the gate measures exactly what ships.
Only the `route` axis is gated (the golden set carries expected_route labels). The
`specificity` axis (point_lookup/aggregate/exploratory) is reported as a distribution
for observability, not asserted, because there are no golden specificity labels.

Routing is a correctness/availability track, NOT a security track: a misroute yields
a wrong answer or a role-gated denial, never a data leak — both retrieval paths
enforce their own RBAC. Golden questions are synthetic (safe to print in reports).
"""

import asyncio
import json
from collections import Counter
from pathlib import Path

import pytest

from evals.eval_config import ROUTING_ACCURACY_THRESHOLD
from evals.eval_schemas import RoutingCase
from src.models.schemas import RouteType
from src.services.router_service import RouterService

_GOLDEN = Path(__file__).parent / "golden" / "routing.json"
_ROUTES: tuple[RouteType, ...] = ("sql_rag", "hybrid_rag")

# Prediction triples: (case, predicted_route, predicted_specificity). Only route is
# gated; specificity rides along for the distribution report. Reused across helpers.
Prediction = tuple[RoutingCase, RouteType, str]


def _load_cases() -> list[RoutingCase]:
    """Load + validate every routing case. Module-level (see test_rbac for why)."""
    raw = json.loads(_GOLDEN.read_text())
    return [RoutingCase(**r) for r in raw]


_CASES = _load_cases()


@pytest.fixture(scope="session")
def routing_predictions(router_service: RouterService) -> list[Prediction]:
    """Classify every case once (concurrently) and cache the predictions.

    Session-scoped so the 40 LLM calls happen a single time, then every report/assert
    reuses the same predictions — no re-querying per test.
    """

    # Pace requests to respect the provider's tokens-per-minute cap (12k TPM on the
    # free tier). Bursting all 40 at once trips a 429 on tokens, not requests — so we
    # run sequentially with a small delay, keeping the rolling token rate well under.
    _PACING_SECONDS = 1.6  # ~37 calls/min * ~300 tok = ~11k tok/min < 12k TPM

    async def _run() -> list[Prediction]:
        results: list[Prediction] = []
        for case in _CASES:
            # classify_detailed() is the exact method the production graph runs.
            decision = await router_service.classify_detailed(
                case.question, request_id=case.id, role="admin"
            )
            results.append((case, decision.route, decision.specificity))
            await asyncio.sleep(_PACING_SECONDS)
        return results

    return asyncio.run(_run())


def _confusion_matrix(preds: list[Prediction]) -> dict[RouteType, dict[RouteType, int]]:
    """actual -> predicted -> count. Diagonal = correct, off-diagonal = misroutes."""
    matrix: dict[RouteType, dict[RouteType, int]] = {
        actual: {pred: 0 for pred in _ROUTES} for actual in _ROUTES
    }
    for case, predicted, _spec in preds:
        matrix[case.expected_route][predicted] += 1
    return matrix


def _accuracy(preds: list[Prediction]) -> float:
    if not preds:
        return 0.0
    correct = sum(1 for c, p, _spec in preds if p == c.expected_route)
    return correct / len(preds)


def _report(preds: list[Prediction]) -> str:
    """Human-readable confusion matrix + per-class + clear/boundary + misroute list."""
    m = _confusion_matrix(preds)
    lines = [
        "",
        "Routing confusion matrix (rows = actual, cols = predicted):",
        f"{'':>12} {'sql_rag':>10} {'hybrid_rag':>12}",
        f"{'sql_rag':>12} {m['sql_rag']['sql_rag']:>10} {m['sql_rag']['hybrid_rag']:>12}",
        f"{'hybrid_rag':>12} {m['hybrid_rag']['sql_rag']:>10} {m['hybrid_rag']['hybrid_rag']:>12}",
        "",
        f"Overall accuracy: {_accuracy(preds):.3f}  (n={len(preds)})",
    ]

    # Per-class recall.
    for actual in _ROUTES:
        total = sum(m[actual].values())
        if total:
            lines.append(f"  {actual} recall: {m[actual][actual] / total:.3f} ({total} cases)")

    # Clear vs boundary breakdown — where the LLM classifier actually struggles.
    for label, subset in (
        ("clear", [p for p in preds if not p[0].is_boundary]),
        ("boundary", [p for p in preds if p[0].is_boundary]),
    ):
        if subset:
            lines.append(f"  {label} accuracy: {_accuracy(subset):.3f} ({len(subset)} cases)")

    # Named misroutes with direction.
    misroutes = [(c, p) for c, p, _spec in preds if p != c.expected_route]
    if misroutes:
        lines.append("  Misroutes:")
        lines += [
            f"    {c.id} [{'boundary' if c.is_boundary else 'clear'}]: "
            f"expected {c.expected_route} -> predicted {p}"
            for c, p in misroutes
        ]

    # Specificity distribution (second axis) — reported, not gated. Shows how the
    # two-axis router labels the corpus: how many point_lookup vs aggregate vs
    # exploratory. Aggregate should correlate with sql_rag; the others with hybrid_rag.
    spec_counts = Counter(spec for _c, _p, spec in preds)
    lines.append("  Specificity distribution (second axis, not gated):")
    lines += [
        f"    {spec}: {count}" for spec, count in sorted(spec_counts.items())
    ]
    return "\n".join(lines)


def test_routing_accuracy_meets_threshold(routing_predictions: list[Prediction]) -> None:
    """Overall routing accuracy must clear the threshold. Prints the full matrix."""
    accuracy = _accuracy(routing_predictions)
    report = _report(routing_predictions)
    print(report)  # visible with `pytest -s`; folded into the failure message too.
    assert accuracy >= ROUTING_ACCURACY_THRESHOLD, (
        f"Routing accuracy {accuracy:.3f} < {ROUTING_ACCURACY_THRESHOLD}.\n{report}"
    )
