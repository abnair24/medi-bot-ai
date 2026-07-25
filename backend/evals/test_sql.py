"""Track 4 — SQL-RAG correctness: valid SQL, correct data, faithful narration.

Three independent failure modes per case:
  1. Execution gate  — the pipeline must not raise SQLGenerationError/RBACError.
     The LLM may generate malformed SQL; if so, the case fails here, not in #2.
  2. SQL keyword gate — the generated SQL must reference expected tables/columns
     (case-insensitive). Catches cases where the LLM writes a syntactically valid
     but semantically wrong query (e.g. querying maintenance_tickets for a claim count).
  3. NL keyword gate — the final NL answer must contain an expected value string.
     This is the user-visible correctness check.

Groq efficiency: each case's pipeline (generate_sql → execute_and_narrate) is run
ONCE in a session-scoped, paced fixture and the outcome cached — every test function
then asserts against the cache. This mirrors the routing eval's pattern. Running the
pipeline per test-function instead (the naive layout) fires ~7 Groq calls per case and
bursts them, which trips the free-tier tokens-per-minute cap (RetryError). Two calls
per case, paced, keeps the whole track well under the limit.

Aggregation gate: execution success rate and NL match rate must meet
SQL_EXECUTION_THRESHOLD / SQL_RESULT_MATCH_THRESHOLD from eval_config. Individual cases
flag WHERE it breaks; the rate gates flag systemic regressions.

DeepEval NL-faithfulness is intentionally omitted here — already covered by
test_faithfulness.py (Track 3). The oracle values (counts, amounts) are known exactly,
so a deterministic keyword check is the right gate for SQL answers.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from evals.eval_config import SQL_EXECUTION_THRESHOLD, SQL_RESULT_MATCH_THRESHOLD
from evals.eval_schemas import SQLCase
from src.services.sql_rag_service import SQLRAGService
from src.utils.exceptions import RBACError, SQLGenerationError

_GOLDEN = Path(__file__).parent / "golden" / "sql.json"

# Pace between cases to respect Groq's free-tier tokens-per-minute cap. Each case is
# two LLM calls (SQL gen + narration, ~2k tokens); a short gap keeps the rolling rate
# under the limit. Same rationale as the routing eval's _PACING_SECONDS.
_PACING_SECONDS = 2.0


def _load_cases() -> list[SQLCase]:
    raw = json.loads(_GOLDEN.read_text())
    return [SQLCase(**r) for r in raw]


_CASES = _load_cases()


def _norm(s: str) -> str:
    """Normalize for tolerant NL matching of SQL answers.

    Unlike Track 3b (numbers), which demands EXACT verbatim quoting of clinical
    values, SQL narration legitimately varies in presentation: 'Cardiology' vs
    'cardiology', '$2,442,700.00' vs '2442700', Indian vs Western digit grouping.
    None of that changes the answer's correctness. So we lowercase and strip thousands
    separators + currency symbols before the substring check; the digits/letters must
    still match.
    """
    s = s.lower()
    for ch in (",", "₹", "$"):
        s = s.replace(ch, "")
    return s


def _answer_matches(answer: str, expected: list[str]) -> bool:
    """True if any expected value appears in the answer (format-normalized)."""
    answer_norm = _norm(answer)
    return any(_norm(v) in answer_norm for v in expected)


@dataclass(frozen=True)
class SqlOutcome:
    """Cached result of running one SQL case's pipeline exactly once.

    error holds the exception type name when the pipeline raised (RBAC/SQL gen/exec);
    on success it is None and sql/answer/row_count are populated.
    """

    sql: str | None
    answer: str | None
    row_count: int | None
    error: str | None


@pytest.fixture(scope="session")
def sql_outcomes(sql_rag_service: SQLRAGService) -> dict[str, SqlOutcome]:
    """Run every SQL case's pipeline ONCE (paced) and cache the outcome by case id.

    Session-scoped so the Groq calls happen a single time; every test function reads
    this cache instead of re-invoking the pipeline. Failures are captured (not raised)
    so one bad case doesn't abort the whole fixture — the per-test assertions decide.
    """

    async def _run_one(case: SQLCase) -> SqlOutcome:
        try:
            sql = await sql_rag_service.generate_sql(
                question=case.question, role=case.role, request_id=case.id
            )
            answer, result = await sql_rag_service.execute_and_narrate(
                sql=sql, question=case.question, role=case.role, request_id=case.id
            )
            return SqlOutcome(sql=sql, answer=answer, row_count=result.row_count, error=None)
        except (SQLGenerationError, RBACError) as exc:
            return SqlOutcome(sql=None, answer=None, row_count=None, error=type(exc).__name__)

    async def _run_all() -> dict[str, SqlOutcome]:
        outcomes: dict[str, SqlOutcome] = {}
        for case in _CASES:
            outcomes[case.id] = await _run_one(case)
            await asyncio.sleep(_PACING_SECONDS)
        return outcomes

    return asyncio.run(_run_all())


# ---------------------------------------------------------------------------
# Per-case tests — all read the cached outcome, zero extra Groq calls
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", _CASES, ids=[c.id for c in _CASES])
def test_sql_executes_without_error(case: SQLCase, sql_outcomes: dict[str, SqlOutcome]) -> None:
    """The pipeline must generate + execute SQL without raising."""
    outcome = sql_outcomes[case.id]
    assert outcome.error is None, (
        f"{case.id}: pipeline raised {outcome.error} instead of producing an answer."
    )


@pytest.mark.parametrize("case", _CASES, ids=[c.id for c in _CASES])
def test_sql_targets_correct_table(case: SQLCase, sql_outcomes: dict[str, SqlOutcome]) -> None:
    """Generated SQL must contain every expected_sql_keyword (case-insensitive)."""
    outcome = sql_outcomes[case.id]
    if outcome.error is not None:
        pytest.fail(f"{case.id}: pipeline raised {outcome.error}; no SQL to inspect.")
    sql = (outcome.sql or "").lower()
    missing = [kw for kw in case.expected_sql_keywords if kw.lower() not in sql]
    assert not missing, (
        f"{case.id}: generated SQL missing expected keywords {missing}. "
        f"SQL (first 300 chars): {sql[:300]!r}"
    )


@pytest.mark.parametrize("case", _CASES, ids=[c.id for c in _CASES])
def test_nl_answer_contains_expected_values(
    case: SQLCase, sql_outcomes: dict[str, SqlOutcome]
) -> None:
    """NL answer must contain an expected value string (format-normalized)."""
    if not case.expected_answer_contains:
        pytest.skip(f"{case.id}: no expected_answer_contains — execution-only case")
    outcome = sql_outcomes[case.id]
    if outcome.error is not None:
        pytest.fail(f"{case.id}: pipeline raised {outcome.error}; no answer to check.")
    assert _answer_matches(outcome.answer or "", case.expected_answer_contains), (
        f"{case.id}: answer did not contain any of {case.expected_answer_contains}. "
        f"Answer (first 400 chars): {(outcome.answer or '')[:400]!r}"
    )


@pytest.mark.parametrize("case", _CASES, ids=[c.id for c in _CASES])
def test_sql_row_count(case: SQLCase, sql_outcomes: dict[str, SqlOutcome]) -> None:
    """When expected_row_count is set, the SQL result must return exactly that many rows."""
    if case.expected_row_count is None:
        pytest.skip(f"{case.id}: no expected_row_count — skipping row count assertion")
    outcome = sql_outcomes[case.id]
    if outcome.error is not None:
        pytest.fail(f"{case.id}: pipeline raised {outcome.error}; no result to count.")
    assert outcome.row_count == case.expected_row_count, (
        f"{case.id}: expected {case.expected_row_count} rows, got {outcome.row_count}"
    )


# ---------------------------------------------------------------------------
# Suite-level aggregation gates — read the same cache (no extra Groq calls)
# ---------------------------------------------------------------------------

def test_sql_execution_rate(sql_outcomes: dict[str, SqlOutcome]) -> None:
    """Execution success rate across ALL cases must meet SQL_EXECUTION_THRESHOLD."""
    successes = sum(1 for o in sql_outcomes.values() if o.error is None)
    rate = successes / len(_CASES)
    assert rate >= SQL_EXECUTION_THRESHOLD, (
        f"SQL execution rate {rate:.2%} < threshold {SQL_EXECUTION_THRESHOLD:.2%}. "
        f"{len(_CASES) - successes}/{len(_CASES)} cases failed to execute."
    )


def test_sql_nl_match_rate(sql_outcomes: dict[str, SqlOutcome]) -> None:
    """NL keyword match rate across checkable cases must meet SQL_RESULT_MATCH_THRESHOLD."""
    checkable = [c for c in _CASES if c.expected_answer_contains]
    if not checkable:
        pytest.skip("No cases with expected_answer_contains — skipping rate gate")

    matches = sum(
        1
        for c in checkable
        if sql_outcomes[c.id].error is None
        and _answer_matches(sql_outcomes[c.id].answer or "", c.expected_answer_contains)
    )
    rate = matches / len(checkable)
    assert rate >= SQL_RESULT_MATCH_THRESHOLD, (
        f"NL match rate {rate:.2%} < threshold {SQL_RESULT_MATCH_THRESHOLD:.2%}. "
        f"{len(checkable) - matches}/{len(checkable)} cases produced wrong answers."
    )
