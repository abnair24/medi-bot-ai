from datetime import date

from src.adapters.llm_adapter import LLMAdapter
from src.adapters.sql_adapter import SQLAdapter
from src.config.access import SQL_RAG_ROLES
from src.models.schemas import SQLResult
from src.utils.exceptions import RBACError
from src.utils.helpers import strip_sql_fences
from src.utils.logger import log_request_event
from src.utils.timing import stopwatch

# PHI columns from the claims table — must never reach an external LLM API.
# frozenset: immutable module-level constant, safe against accidental mutation.
_PHI_COLUMNS: frozenset[str] = frozenset({"patient_name", "patient_id"})

_SQL_GEN_PROMPT = """\
You are a SQL expert. Given the database schema below, write a SQLite query to answer the
user's question.

Today's date is {today} (YYYY-MM-DD). Use THIS literal date for every relative-date
phrase. NEVER use DATE('now'), CURRENT_DATE, or the system clock — the data's dates are
fixed and will NOT match the real clock, so 'now' returns nothing.

Compute the literal 'YYYY' / 'YYYY-MM' yourself from {today} and embed it as a string:
- "this year"  → strftime('%Y', <date_col>) = the 4-digit year of {today}
- "last year"  → strftime('%Y', <date_col>) = that year minus 1
- "this month" → strftime('%Y-%m', <date_col>) = the year-month of {today}
- "last month" → strftime('%Y-%m', <date_col>) = the year-month of the previous month
- a date range → <date_col> >= 'YYYY-MM-DD' AND <date_col> < 'YYYY-MM-DD'
Match the phrase exactly: "this year" filters the whole YEAR, not a month.

Schema:
- claims(claim_id TEXT, patient_id TEXT, patient_name TEXT, department TEXT,
  claim_type TEXT, diagnosis_code TEXT, insurer TEXT, claimed_amount REAL,
  approved_amount REAL, status TEXT, submitted_date TEXT, resolved_date TEXT)
  Statuses: pending, approved, rejected, submitted, escalated
  Departments: nephrology, cardiology, neurology, gynaecology, orthopaedics,
  general_medicine, emergency

- maintenance_tickets(ticket_id TEXT, equipment_name TEXT, equipment_id TEXT,
  category TEXT, campus TEXT, issue_type TEXT, fault_code TEXT, raised_by TEXT,
  raised_date TEXT, resolved_date TEXT, status TEXT, resolution_note TEXT)
  Categories: sterilisation, infusion, radiology, monitoring, surgical, laboratory
  Statuses: open, in_progress, resolved, escalated

Rules:
- Return ONLY the SQL query, no explanation, no markdown fences.
- Use exact column names and value strings from the schema above.
- For percentages, compute as 100.0 * count / total.
- Date columns are TEXT in YYYY-MM-DD format. Filter a month with
  strftime('%Y-%m', <date_col>) = 'YYYY-MM' and a year with
  strftime('%Y', <date_col>) = 'YYYY'. Never compare only SUBSTR of a hardcoded date.
- For claims, "approved/resolved/rejected <when>" refers to resolved_date;
  "submitted/filed <when>" refers to submitted_date.
- When the question asks which group has the most/least/highest/top (a ranking),
  SELECT BOTH the grouping column AND the aggregate value — e.g.
  SELECT department, COUNT(*) AS claim_count FROM claims GROUP BY department
  ORDER BY claim_count DESC LIMIT 1 — so the answer can state the actual number,
  not just the name. Never return only the label of a ranked group.

User question: {question}

SQL:\
"""

_NARRATION_PROMPT = """\
The user asked: {question}

The database returned: {result}

Write a clear, concise natural-language answer based on this result. Include the actual
numbers.

Rules:
- ALL monetary amounts are Indian Rupees. Format every amount with the ₹ symbol and
  Indian digit grouping (e.g. 220500 → ₹2,20,500). NEVER use "$" or "USD" — there are no
  dollar amounts in this database, whether the claim was approved, rejected, or pending.
- Report amounts exactly as returned; do not convert, round, or invent a currency.
- If the result is empty or null, say that no matching data was found.\
"""


class SQLRAGService:
    """Executes the full SQL RAG chain: role check → SQL gen → execute → PHI strip → narrate.

    Two LLM calls per request: SQL generation and result narration.
    PHI columns are stripped between execution and narration — the narration
    LLM never receives patient_name or patient_id.
    """

    def __init__(self, llm: LLMAdapter, sql: SQLAdapter) -> None:
        self._llm = llm
        self._sql = sql

    async def generate_sql(
        self,
        question: str,
        role: str,
        request_id: str,
    ) -> str:
        """Role-check and generate a clean SQL query from the user's question.

        Args:
            question: user's analytical question. Passed to LLM but never logged.
            role: authenticated user role — must be in SQL_RAG_ROLES.
            request_id: trace ID for log correlation.

        Returns:
            Clean SQL string with markdown fences stripped.

        Raises:
            RBACError: if role is not permitted to use SQL RAG.
            SQLGenerationError: if the LLM response cannot be cleaned into valid SQL.
        """
        if role not in SQL_RAG_ROLES:
            raise RBACError(
                f"Role '{role}' is not permitted to use SQL RAG. "
                f"Requires one of: {SQL_RAG_ROLES}"
            )

        with stopwatch() as elapsed_ms:
            raw_sql = await self._llm.generate(
                _SQL_GEN_PROMPT.format(
                    question=question, today=date.today().isoformat()
                ),
                temperature=0.0,
            )
        clean_sql = strip_sql_fences(raw_sql)

        # Log the generated SQL — operational metadata used by SQL correctness eval.
        # Never log: question (may name a patient).
        log_request_event(
            request_id=request_id,
            role=role,
            event="sql_generated",
            latency_ms=elapsed_ms(),
            generated_sql=clean_sql,
        )

        return clean_sql

    async def execute_and_narrate(
        self,
        sql: str,
        question: str,
        role: str,
        request_id: str,
    ) -> tuple[str, SQLResult]:
        """Execute approved SQL and return a narrated natural-language answer.

        Args:
            sql: clean SQL string, already approved (by human or auto).
            question: original user question for narration context. Never logged.
            role: authenticated user role for log correlation.
            request_id: trace ID for log correlation.

        Returns:
            Tuple of (nl_answer, sql_result). sql_result has PHI columns stripped.

        Raises:
            SQLGenerationError: if execution fails or returns no column metadata.
        """
        with stopwatch() as elapsed_ms:
            result = await self._sql.execute(sql)
            safe_result = _strip_phi(result)

            answer = await self._llm.generate(
                _NARRATION_PROMPT.format(
                    question=question,
                    result=safe_result.model_dump_json(),
                ),
                temperature=0.0,
            )

        # Never log: question, answer (both may echo PHI), or result rows.
        phi_present = bool(_PHI_COLUMNS & {col for row in result.rows for col in row})
        log_request_event(
            request_id=request_id,
            role=role,
            latency_ms=elapsed_ms(),
            event="sql_executed",
            row_count=result.row_count,
            phi_columns_stripped=phi_present,
        )

        return answer, safe_result

    async def answer(
        self,
        question: str,
        role: str,
        request_id: str,
    ) -> tuple[str, SQLResult]:
        """Convenience method: generate_sql → execute_and_narrate in one call.

        Used by tests and callers that do not need the HITL propose/approve split.
        The graph nodes call generate_sql and execute_and_narrate separately.
        """
        clean_sql = await self.generate_sql(question, role, request_id)
        return await self.execute_and_narrate(clean_sql, question, role, request_id)


def _strip_phi(result: SQLResult) -> SQLResult:
    """Remove PHI columns from all rows before they reach the narration LLM.

    patient_name and patient_id must never be sent to an external LLM API
    without a signed data-processing agreement (DATA_HANDLING.md §4).
    Stripping here — at the service boundary — ensures no downstream stage
    (narration prompt, API response, frontend) receives raw PHI from the DB.
    """
    safe_rows = [
        {k: v for k, v in row.items() if k not in _PHI_COLUMNS}
        for row in result.rows
    ]
    return SQLResult(query=result.query, rows=safe_rows, row_count=result.row_count)
