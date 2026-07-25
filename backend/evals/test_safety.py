"""Track 11 — Medical safety guardrail: deterministic violation detection.

Tests safety_guard.run_safety_checks() against a fixed golden set of pre-written
answers. No LLM, no network, no fixtures — these are pure function tests. Runtime
cost: microseconds per case.

Why deterministic over LLM-judged? The safety contract is binary: a phrase like
"you should administer" is ALWAYS prescriptive; a missing disclaimer is ALWAYS a
violation. There is no spectrum of partial compliance here — the guardrails.md rules
are explicit enough that a regex/string detector is the correct tool. Adding a
language model would introduce variance into a check that must be fail-closed.

Test strategy: two assertions per case:
  1. No unexpected violations  — a false positive (blocking a safe answer) is also
     a failure. The safety guard must not over-block compliant content.
  2. No missing violations     — a false negative (passing a violating answer) is
     a safety failure. The guard must catch everything expected_violations names.
"""

import json
from pathlib import Path

import pytest

from evals.eval_schemas import SafetyCase
from src.services.safety_guard import ViolationType, run_safety_checks

_GOLDEN = Path(__file__).parent / "golden" / "safety.json"

# The canonical disclaimer marker injected by GenerationService.
# Duplicating the exact substring here rather than importing from generation_service
# avoids a circular dependency. The string is stable — it's the canonical medical
# disclaimer the network has committed to.
_DISCLAIMER_MARKERS = ("Always apply clinical judgement",)


def _load_cases() -> list[SafetyCase]:
    raw = json.loads(_GOLDEN.read_text())
    return [SafetyCase(**r) for r in raw]


_CASES = _load_cases()


@pytest.mark.parametrize("case", _CASES, ids=[c.id for c in _CASES])
def test_safety_violations_exact(case: SafetyCase) -> None:
    """Violations found must exactly match expected_violations — no more, no less.

    Both false positives (over-blocking) and false negatives (under-blocking) are
    test failures. Order does not matter — sets are compared.

    The test intentionally does NOT call the full generation pipeline. The golden
    case supplies the answer string directly so the safety_guard is tested in
    isolation from LLM output variance. Same logic guards the production path in
    generation_service.py.
    """
    violations = run_safety_checks(
        case.answer,
        is_clinical=case.is_clinical,
        disclaimer_markers=_DISCLAIMER_MARKERS,
    )
    found_types = {v.type for v in violations}
    expected_types = set(case.expected_violations)

    false_positives = found_types - expected_types
    false_negatives = expected_types - found_types

    assert not false_positives, (
        f"{case.id} ({case.description}): "
        f"OVER-BLOCK — guard raised unexpected violations {false_positives}. "
        "A false positive withholds a safe answer from a clinician."
    )
    assert not false_negatives, (
        f"{case.id} ({case.description}): "
        f"UNDER-BLOCK — guard missed violations {false_negatives}. "
        "A false negative lets a non-compliant answer reach the user."
    )


def test_blockquote_shields_directive_verbs() -> None:
    """Directive verbs inside Markdown blockquotes must NOT trigger PRESCRIPTIVE_LANGUAGE.

    This is the critical pattern: MediBot quotes protocols verbatim in blockquotes.
    Protocol text uses imperative voice ('Administer X', 'Monitor Y'). The guard
    must distinguish the SOURCE speaking (inside blockquote) from the BOT speaking
    (outside blockquote).
    """
    answer = (
        "> Administer 300 mg Aspirin loading dose immediately.\n"
        "> Monitor ECG for 12 hours post-admission.\n"
        "> You should follow the NSTEMI pathway.\n\n"
        "*This information is from internal protocols. "
        "Always apply clinical judgement and consult the treating physician "
        "for patient-specific decisions.*"
    )
    violations = run_safety_checks(
        answer,
        is_clinical=True,
        disclaimer_markers=_DISCLAIMER_MARKERS,
    )
    prescriptive = [v for v in violations if v.type == ViolationType.PRESCRIPTIVE_LANGUAGE]
    assert not prescriptive, (
        "Blockquoted directive voice incorrectly flagged as prescriptive. "
        "The guard must strip blockquote lines before scanning for directive patterns."
    )


def test_unit_conversion_requires_both_units() -> None:
    """A single temperature unit alone must NOT trigger UNIT_CONVERSION.

    The guard fires when BOTH °F and °C appear — that pattern signals a conversion.
    An answer quoting only °C (the clinical standard in India) is compliant.
    """
    answer = (
        "> Maintain core temperature ≥ 36.5 °C during the post-operative period.\n\n"
        "*This information is from internal protocols. "
        "Always apply clinical judgement and consult the treating physician "
        "for patient-specific decisions.*"
    )
    violations = run_safety_checks(
        answer,
        is_clinical=True,
        disclaimer_markers=_DISCLAIMER_MARKERS,
    )
    unit_violations = [v for v in violations if v.type == ViolationType.UNIT_CONVERSION]
    assert not unit_violations, (
        "Single-unit temperature incorrectly flagged as unit conversion. "
        "UNIT_CONVERSION requires BOTH °F and °C to be present."
    )


def test_disclaimer_not_required_for_nonclinical() -> None:
    """MISSING_DISCLAIMER must NEVER fire when is_clinical=False.

    Non-clinical answers (billing, equipment, general) are not governed by the
    clinical disclaimer requirement. Flagging them would over-block administrative
    content that has no clinical implication.
    """
    answer = "Leave encashment is capped at 30 days per financial year per HR policy 4.2."
    violations = run_safety_checks(
        answer,
        is_clinical=False,
        disclaimer_markers=_DISCLAIMER_MARKERS,
    )
    disclaimer_violations = [
        v for v in violations if v.type == ViolationType.MISSING_DISCLAIMER
    ]
    assert not disclaimer_violations, (
        "MISSING_DISCLAIMER fired on a non-clinical answer. "
        "is_clinical=False must suppress the disclaimer check entirely."
    )
