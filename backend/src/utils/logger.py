import logging
import sys
from typing import Any

from pythonjsonlogger import jsonlogger

from src.config.settings import settings


def _build_logger() -> logging.Logger:
    """Configure and return the application-wide JSON logger."""
    logger = logging.getLogger("medibot")
    logger.setLevel(settings.log_level.upper())

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevent log records from bubbling to the root logger (avoids duplicate output).
    logger.propagate = False
    return logger


logger = _build_logger()


def log_request_event(
    *,
    request_id: str,
    role: str,
    event: str,
    **operational_fields: Any,
) -> None:
    """Log an operational event for a request. Never pass query text or LLM responses.

    Args:
        request_id: trace ID for correlating all log records in one request.
        role: authenticated user role (not PHI).
        event: short label for the pipeline stage e.g. "route_decision", "retrieval".
        **operational_fields: safe fields only — chunk_ids, scores, latency_ms,
            rbac_blocked, route. Never pass query, answer, or prompt.
    """
    logger.info(
        event,
        extra={
            "request_id": request_id,
            "role": role,
            **operational_fields,
        },
    )
