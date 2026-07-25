import aiosqlite

from src.config.settings import settings
from src.models.schemas import SQLResult
from src.utils.exceptions import SQLGenerationError


class SQLAdapter:
    """Read-only async SQLite wrapper for mediassist.db.

    Enforces two hard constraints on every connection:
    - PRAGMA query_only=ON  : rejects any data-modifying statement at the DB level.
    - PRAGMA journal_mode=WAL: allows concurrent readers without blocking.

    Both pragmas are applied per-connection so LLM-generated SQL cannot bypass them
    regardless of what the LLM outputs.
    """

    def __init__(self) -> None:
        self._path = settings.sqlite_path

    async def execute(self, sql: str) -> SQLResult:
        """Execute a read-only SQL query and return structured results.

        Args:
            sql: clean SQL string with markdown fences already stripped.

        Returns:
            SQLResult with rows (PHI columns NOT yet stripped — caller's responsibility)
            and row_count.

        Raises:
            SQLGenerationError: if SQL is invalid, execution fails, or a write
                operation is attempted (blocked by query_only pragma).
        """
        try:
            async with aiosqlite.connect(self._path) as conn:
                await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute("PRAGMA query_only=ON")

                cursor = await conn.execute(sql)

                if cursor.description is None:
                    raise SQLGenerationError(
                        "Query returned no column metadata — only SELECT statements "
                        "are permitted. Received a non-SELECT or empty statement."
                    )

                columns = [desc[0] for desc in cursor.description]
                rows = await cursor.fetchall()

        except SQLGenerationError:
            raise
        except Exception as e:
            raise SQLGenerationError(f"SQL execution failed: {e}") from e

        return SQLResult(
            query=sql,
            rows=[dict(zip(columns, row)) for row in rows],
            row_count=len(rows),
        )
