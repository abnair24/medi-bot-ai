import re

from groq import AsyncGroq, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config.settings import settings
from src.utils import token_meter
from src.utils.exceptions import RateLimitedError

# Groq rate-limit messages embed both a retry hint and the limit kind, e.g.:
#   "... on tokens per day (TPD): Limit 100000, Used 99683 ... Please try again in 4m33.024s."
# We parse the retry-after and classify the window (daily / per-minute / hourly) so the
# client can show "try again in X" instead of a generic 500.
_RETRY_AFTER_RE = re.compile(r"try again in ([0-9hms.]+s)", re.IGNORECASE)

# Ordered signatures → human label. Groq spells the window as (TPD)/(TPM)/(RPM)/etc.
# or as "per day"/"per minute"/"per hour". Check the coarse phrases; the (TPx) codes
# contain "day"/"minute"/"hour" only via the phrase, so the phrase check is sufficient.
_LIMIT_KINDS: tuple[tuple[str, str], ...] = (
    ("per day", "daily"),
    ("(TPD)", "daily"),
    ("(RPD)", "daily"),
    ("per hour", "hourly"),
    ("per minute", "per-minute"),
    ("(TPM)", "per-minute"),
    ("(RPM)", "per-minute"),
)


def _parse_rate_limit(message: str) -> tuple[str, str]:
    """Extract (retry_after, limit_kind) from a Groq RateLimitError message.

    retry_after is cleaned to whole seconds ('4m33.024s' -> '4m33s'); '' if absent.
    limit_kind is 'daily' | 'per-minute' | 'hourly' | '' (unknown).
    """
    retry_after = ""
    match = _RETRY_AFTER_RE.search(message)
    if match:
        # Drop fractional seconds for a cleaner "4m33s" instead of "4m33.024s".
        retry_after = re.sub(r"\.\d+s$", "s", match.group(1))

    limit_kind = ""
    lowered = message.lower()
    for needle, label in _LIMIT_KINDS:
        if needle.lower() in lowered:
            limit_kind = label
            break

    return retry_after, limit_kind


def _friendly_rate_limit_message(retry_after: str, limit_kind: str) -> str:
    """Build the user-facing 'you are rate limited, try after X' message."""
    kind = f" ({limit_kind} limit)" if limit_kind else ""
    when = f" Please try again in {retry_after}." if retry_after else " Please try again shortly."
    return f"You are rate limited{kind}.{when}"


class LLMAdapter:
    """Wraps the Groq async client for LLM text generation.

    All calls use temperature=0.0 — medical content requires deterministic output
    so drug doses and protocol numbers are quoted exactly, not paraphrased.
    Retries on RateLimitError (Groq free tier) with exponential backoff, then converts
    a persistent rate limit into a RateLimitedError carrying the provider's retry-after.
    """

    def __init__(self) -> None:
        self._client = AsyncGroq(api_key=settings.groq_api_key)

    async def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Call the Groq LLM and return the response text.

        Retries transient rate limits internally (see _generate_once). If the provider
        is STILL rate-limiting after the retry budget, the raw groq.RateLimitError is
        converted into a RateLimitedError with a parsed retry-after + limit kind, so the
        API layer can return a clean 429 "try again in X" instead of a 500.

        Args:
            prompt: complete prompt string (system + context + question). Never logged.
            temperature: sampling temperature. 0.0 (default) for factual RAG answers;
                0.5 for conversational responses (e.g. refusals).

        Returns:
            Raw text response from the model.

        Raises:
            RateLimitedError: when Groq rate-limits past the retry budget (→ HTTP 429).
            groq.APIError: for non-retryable errors (bad request, server error).
        """
        try:
            return await self._generate_once(prompt, temperature)
        except RateLimitError as exc:
            retry_after, limit_kind = _parse_rate_limit(str(exc))
            raise RateLimitedError(
                _friendly_rate_limit_message(retry_after, limit_kind),
                retry_after=retry_after,
                limit_kind=limit_kind,
            ) from exc

    # Groq's free tier caps tokens-per-minute; a burst of large-context generation
    # calls (RAG answers, SQL narration) can exceed it. The retry waits out a full
    # ~60s window (up to 30s over 6 attempts — the profile groq_judge.py uses). A daily
    # limit won't clear within the budget, so it surfaces as RateLimitedError above.
    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(RateLimitError),
        reraise=True,
    )
    async def _generate_once(self, prompt: str, temperature: float) -> str:
        """Single Groq call with retry-on-rate-limit. Reraises RateLimitError if exhausted."""
        response = await self._client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        # Record token usage for per-request cost observability (PHI-safe — a count only).
        usage = getattr(response, "usage", None)
        if usage is not None:
            token_meter.add(getattr(usage, "total_tokens", 0) or 0)
        return response.choices[0].message.content
