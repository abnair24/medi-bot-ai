"""Eval router — admin-only endpoints to trigger eval runs and read reports.

Mounted only when settings.evals_enabled is True (see app.py). Every endpoint is
gated to the admin role via the require_admin dependency — a non-admin JWT gets a
403 (RBACError), mirroring how SQL RAG is gated to billing_executive/admin.

Only synthetic golden-set queries are ever executed here (data-handling.md).
"""

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from src.models.auth import TokenPayload
from src.security.jwt import get_current_user
from src.services import eval_service
from src.utils.exceptions import RBACError
from src.utils.logger import logger

router = APIRouter(prefix="/evals", tags=["evals"])


def require_admin(user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
    """Dependency: allow only the admin role. Raises RBACError (→403) otherwise."""
    if user.role != "admin":
        raise RBACError(
            f"Evaluation dashboard is admin-only. Your role '{user.role}' "
            "does not have access."
        )
    return user


class EvalRunRequest(BaseModel):
    """POST /evals/run body — which tracks to execute."""

    tracks: list[str] = Field(..., min_length=1)


@router.get("/status", summary="Current eval run status (admin)")
async def eval_status(user: TokenPayload = Depends(require_admin)) -> dict:
    """Return whether a run is in progress and timing metadata."""
    return eval_service.get_status()


@router.get("/results", summary="Latest eval results (admin)")
async def eval_results(user: TokenPayload = Depends(require_admin)) -> dict:
    """Return the parsed results of the most recent runs, per track."""
    return eval_service.get_results()


@router.post(
    "/run",
    summary="Trigger an eval run (admin)",
    status_code=status.HTTP_202_ACCEPTED,
)
async def eval_run(
    body: EvalRunRequest,
    user: TokenPayload = Depends(require_admin),
) -> dict:
    """Launch the selected eval tracks in the background.

    Raises:
        RBACError: if the caller is not admin (via require_admin).
        ValueError: if a run is already in progress or a track key is unknown
                    → surfaced as 400 by FastAPI's default validation handler.
    """
    logger.info("eval_run_requested", extra={"tracks": body.tracks})
    try:
        return eval_service.start_run(body.tracks)
    except ValueError as exc:
        # Map to a 409 conflict (run in progress) / 400 (bad track) equivalently.
        from fastapi import HTTPException
        raise HTTPException(status_code=409, detail=str(exc)) from exc
