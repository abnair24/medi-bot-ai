class MediBotError(Exception):
    """Base exception for all MediBot application errors.

    Catch this in tests to assert any MediBot-specific failure.
    Catch a subclass to handle a specific failure mode.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class RetrievalError(MediBotError):
    """Raised when vector search fails (Qdrant unreachable, timeout, etc.)."""


class RBACError(MediBotError):
    """Raised when a role attempts to access a collection it cannot see."""


class SQLGenerationError(MediBotError):
    """Raised when SQL generation produces invalid SQL or execution fails."""


class RoutingError(MediBotError):
    """Raised when the semantic router returns an unrecognised route type."""


class IngestionError(MediBotError):
    """Raised when document parsing or chunk storage fails during ingestion.

    Not mapped to an HTTP response — ingestion runs as a standalone script,
    never inside the FastAPI request path.
    """


class AuthenticationError(MediBotError):
    """Raised when login credentials are invalid or a JWT fails validation.

    Mapped to HTTP 401 in the FastAPI exception handler.
    """


class HITLError(MediBotError):
    """Raised when a HITL confirmation token is invalid, expired, or already used.

    Mapped to HTTP 409 (Conflict) in the FastAPI exception handler —
    the resource state (graph checkpoint) doesn't match the request.
    """


class RateLimitedError(MediBotError):
    """Raised when the LLM provider (Groq) rate-limits a request after retries.

    Mapped to HTTP 429 in the FastAPI exception handler. Carries the provider's
    retry-after hint and the limit kind (daily / per-minute / hourly) parsed from
    the Groq error, so the client can show an informative "try again in X" message
    instead of a generic 500.

    ``message`` is already user-facing (built by the adapter) — safe to surface.
    """

    def __init__(
        self, message: str, retry_after: str = "", limit_kind: str = ""
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.limit_kind = limit_kind
