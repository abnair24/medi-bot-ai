import uuid


def generate_request_id() -> str:
    """Generate a UUID4 string for trace correlation across log records."""
    return str(uuid.uuid4())


def strip_sql_fences(raw: str) -> str:
    """Remove markdown code fences from LLM SQL output.

    The SQL generation LLM sometimes wraps output in ```sql ... ``` blocks
    despite being instructed not to. This strips them defensively.
    """
    import re
    cleaned = re.sub(r"```(?:sql)?\s*\n?", "", raw, flags=re.IGNORECASE)
    return cleaned.strip()
