"""Read-only SQL validation via sqlglot AST traversal (Phase 12 agentic SQL).

The linear SQL path (Phase 5) relies on SQLite's `PRAGMA query_only=ON` as its safety net.
The agentic tool-loop (Phase 12) lets the LLM run SQL it composes itself over multiple
iterations, so it needs a stricter, provider-independent gate *before* execution:

  1. Exactly ONE statement — rejects stacked `SELECT 1; DROP TABLE claims`.
  2. The statement is a read-only SELECT (or UNION of SELECTs / CTE-led SELECT).
  3. No DML/DDL node anywhere in the tree — rejects tricks hidden in CTEs or subqueries.
  4. (Optional) every referenced table is in an allow-list — a structural RBAC backstop.

Why an AST, not a regex: a regex blocklist for "DROP"/"DELETE" is trivially bypassed
(comments, casing, `/**/`, string literals). Parsing to an AST and checking node *types*
is robust — `DROP` inside a WITH clause is still an `exp.Drop` node. `PRAGMA query_only`
stays on as defence-in-depth; this validator is the first, LLM-facing gate.
"""

import sqlglot
from sqlglot import exp

from src.utils.exceptions import SQLValidationError

# Statement/expression node types that mutate data or schema — forbidden anywhere in the tree.
_FORBIDDEN: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.TruncateTable,
    exp.Command,   # ATTACH, VACUUM, and other un-parsed passthrough commands
    exp.Pragma,    # PRAGMA writes (e.g. query_only=OFF) — never let the LLM touch pragmas
)

# A valid read-only top-level statement is one of these.
_READ_ONLY_ROOTS: tuple[type[exp.Expression], ...] = (exp.Select, exp.Union, exp.Subquery)


def validate_read_only(sql: str, allowed_tables: set[str] | None = None) -> str:
    """Validate that `sql` is a single read-only SELECT; return it unchanged if valid.

    Args:
        sql: the SQL string to validate (already fence-stripped).
        allowed_tables: if given, every referenced table must be in this set (lower-cased).

    Returns:
        The same `sql` string (convenience for `sql = validate_read_only(sql, ...)`).

    Raises:
        SQLValidationError: with a specific reason the agent can act on.
    """
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except Exception as exc:  # sqlglot.errors.ParseError and friends
        raise SQLValidationError(f"SQL did not parse: {exc}") from exc

    statements = [s for s in statements if s is not None]
    if len(statements) == 0:
        raise SQLValidationError("Empty SQL — nothing to execute.")
    if len(statements) > 1:
        raise SQLValidationError(
            "Multiple SQL statements are not allowed — submit a single SELECT."
        )

    stmt = statements[0]
    if not isinstance(stmt, _READ_ONLY_ROOTS):
        raise SQLValidationError(
            f"Only read-only SELECT queries are allowed, got a {type(stmt).__name__} "
            "statement. No INSERT / UPDATE / DELETE / DROP / ALTER / CREATE / PRAGMA."
        )

    # Defence-in-depth: reject a mutating node hidden anywhere (CTE, subquery, union arm).
    for node in stmt.walk():
        if isinstance(node, _FORBIDDEN):
            raise SQLValidationError(
                f"SQL contains a forbidden {type(node).__name__} operation. "
                "Only read-only SELECT is permitted."
            )

    if allowed_tables is not None:
        # CTE aliases (WITH e AS (...)) are referenced like tables but aren't real tables —
        # exclude them so a legitimate CTE query isn't flagged as an out-of-allow-list table.
        cte_names = {c.alias_or_name.lower() for c in stmt.find_all(exp.CTE) if c.alias_or_name}
        referenced = {
            t.name.lower() for t in stmt.find_all(exp.Table) if t.name
        } - cte_names
        forbidden = sorted(referenced - {a.lower() for a in allowed_tables})
        if forbidden:
            raise SQLValidationError(
                f"Query references table(s) outside the allowed set: {forbidden}. "
                f"Allowed: {sorted(allowed_tables)}."
            )

    return sql
