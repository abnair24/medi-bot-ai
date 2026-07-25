"""Per-request LLM token accounting — PHI-safe cost observability.

Groq's free tier is capped by tokens-per-minute AND tokens-per-day, so knowing how
many tokens a request spent is operationally important (we hit the daily cap during
development). This meter accumulates the `usage.total_tokens` reported by every Groq
call in a request, so the API layer can log/return the total.

Implementation note — why a mutable object in a ContextVar:
Each HTTP request runs in its own asyncio context, so a ContextVar isolates requests
from each other. But LangGraph may execute nodes in CHILD tasks, and a child's
`ContextVar.set()` does NOT propagate back to the parent. So we set the ContextVar ONCE
per request to a mutable `_Counter`, and child tasks inherit the SAME object reference —
mutating it (`.add`) is visible everywhere, including back in the middleware. Only IDs
and counts flow through here; never query/answer text.
"""

import contextvars


class _Counter:
    """Mutable token accumulator shared across a request's tasks."""

    __slots__ = ("tokens", "calls")

    def __init__(self) -> None:
        self.tokens = 0
        self.calls = 0

    def add(self, tokens: int) -> None:
        self.tokens += tokens
        self.calls += 1


_meter: contextvars.ContextVar["_Counter | None"] = contextvars.ContextVar(
    "llm_token_meter", default=None
)


def reset() -> None:
    """Start a fresh count for the current request (call once at request entry)."""
    _meter.set(_Counter())


def add(tokens: int) -> None:
    """Add an LLM call's token usage to the current request's total. No-op if unset."""
    counter = _meter.get()
    if counter is not None and tokens:
        counter.add(tokens)


def total_tokens() -> int:
    """Total tokens spent by the current request (0 if the meter was never started)."""
    counter = _meter.get()
    return counter.tokens if counter is not None else 0


def call_count() -> int:
    """Number of LLM calls made in the current request."""
    counter = _meter.get()
    return counter.calls if counter is not None else 0
