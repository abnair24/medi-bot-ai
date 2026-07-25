"""Chat router — /chat, /chat/resume, /collections, /health.

All endpoints except /health require JWT authentication via Depends(get_current_user).
The graph is accessed via request.app.state.graph (compiled once at startup).
"""

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Request, status
from langgraph.types import Command

from src.config.access import get_collections_for_role
from src.config.settings import settings
from src.models.auth import TokenPayload
from src.models.schemas import (
    ChatRequest,
    ChatResponse,
    ConfirmRequest,
    ConfirmResponse,
    SQLProposal,
)
from src.security.jwt import get_current_user
from src.utils.exceptions import HITLError
from src.utils.logger import log_request_event

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    summary="Send a question to MediBot",
    status_code=status.HTTP_200_OK,
)
async def chat(
    body: ChatRequest,
    request: Request,
    user: TokenPayload = Depends(get_current_user),
) -> ChatResponse | SQLProposal:
    """Invoke the MediBot graph for a single question.

    Role is ALWAYS taken from the JWT — body.role is overridden.
    This prevents privilege escalation via forged request bodies.

    Returns:
        ChatResponse for completed answers (hybrid or SQL RAG).
        SQLProposal when HITL is enabled and SQL needs user approval.
    """
    # Reuse the request_id assigned by the request-context middleware (same ID appears
    # in the X-Request-ID response header). Fall back to a fresh one if unset (e.g. tests).
    request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())

    # Override client-supplied role with authenticated JWT role.
    # Defense-in-depth: even if body.role says "admin", we use the JWT's role.
    role = user.role

    log_request_event(
        request_id=request_id,
        role=role,
        event="chat_request_received",
        session_id=body.session_id,
    )

    # LangGraph config — thread_id enables multi-turn memory via checkpointer.
    # If no session_id provided, each request is a standalone conversation.
    config = {"configurable": {"thread_id": body.session_id or request_id}}

    # --- Build history from checkpointed state (prior turns) ---
    # LangGraph's default merge for non-Annotated list fields is REPLACEMENT.
    # Passing history=[] would overwrite whatever the checkpointer restored.
    # Instead: read prior state, extract previous Q&A, accumulate into history.
    graph = request.app.state.graph
    history: list[dict] = []

    if body.session_id:
        # Multi-turn: checkpoint may hold prior turns for this session.
        prior_state = await graph.aget_state(config)
        if prior_state and prior_state.values:
            prev = prior_state.values
            # Carry forward existing history from the prior turn.
            existing_history = prev.get("history", [])
            # Append the prior turn's Q&A pair (if the graph completed).
            # Use ORIGINAL_QUESTION, not `question`: rewrite_node overwrites `question`
            # with the rewritten form, and storing that lets prior-turn context snowball
            # into every later rewrite (a fever turn contaminating all follow-ups).
            prev_question = prev.get("original_question") or prev.get("question")
            prev_answer = prev.get("answer")
            if prev_question and prev_answer:
                history = existing_history + [
                    {"role": "user", "content": prev_question},
                    {"role": "assistant", "content": prev_answer},
                ]
            else:
                # Prior turn didn't complete (e.g., HITL pending) — carry history as-is.
                history = existing_history

    # Sliding window: keep only the last N turns (2 messages each) so a long session
    # doesn't accumulate stale topics or unbounded context/cost.
    window = 2 * settings.conversation_window_turns
    history = history[-window:]

    # Build the graph input — matches MediBotState keys.
    graph_input = {
        "question": body.question,
        "original_question": body.question,  # preserved for SQL path; rewrite_node overwrites question
        "role": role,
        "request_id": request_id,
        "history": history,
        "session_id": body.session_id,
    }

    # --- Invoke the compiled graph ---
    # ainvoke() runs all nodes until END or interrupt().
    result = await graph.ainvoke(graph_input, config=config)

    # --- Check for HITL interrupt (SQL approval pending) ---
    # When interrupt() fires in sql_propose_node, the graph suspends.
    # The result contains proposed_sql but no answer yet.
    graph_state = await graph.aget_state(config)
    if graph_state.next:
        # Graph is paused — there are nodes left to execute.
        # This means HITL interrupted: return the SQL proposal.
        proposed_sql = result.get("proposed_sql", "")
        token = body.session_id or request_id

        log_request_event(
            request_id=request_id,
            role=role,
            event="hitl_proposal_sent",
            proposed_sql_length=len(proposed_sql),
        )

        return SQLProposal(
            proposed_sql=proposed_sql,
            confirmation_token=token,
            expires_at=datetime.now(timezone.utc)
            + timedelta(seconds=settings.hitl_proposal_ttl_seconds),
        )

    # --- Normal completion: build ChatResponse ---
    # sources is already set in state by generate_node (hybrid_rag) or
    # sql_execute_node (sql_rag, returns []). Read directly — don't rebuild.
    sources = result.get("sources", [])

    log_request_event(
        request_id=request_id,
        role=role,
        event="chat_response_sent",
        route=result.get("route"),
        source_count=len(sources),
    )

    return ChatResponse(
        answer=result.get("answer", ""),
        sources=sources,
        retrieval_type=result.get("route", "hybrid_rag"),
        role=role,
    )


@router.post(
    "/chat/resume",
    response_model=ConfirmResponse,
    summary="Approve or reject a proposed SQL query (HITL)",
    status_code=status.HTTP_200_OK,
)
async def resume_chat(
    body: ConfirmRequest,
    request: Request,
    user: TokenPayload = Depends(get_current_user),
) -> ConfirmResponse:
    """Resume a suspended graph after HITL SQL approval.

    The confirmation_token is the session_id / thread_id used to look up
    the checkpointed graph state. The graph resumes from the interrupt()
    point with the user's approve/reject decision.

    Raises:
        HITLError: if the token doesn't match any suspended graph state,
                   or if the proposal has expired (past TTL).
    """
    config = {"configurable": {"thread_id": body.confirmation_token}}

    # Verify the graph is actually suspended for this token.
    graph = request.app.state.graph
    graph_state = await graph.aget_state(config)

    if not graph_state or not graph_state.next:
        raise HITLError(
            "No pending SQL proposal for this token — it may have expired or "
            "already been processed."
        )

    # --- TTL enforcement ---
    # The checkpoint's created_at timestamp tells us when the proposal was made.
    # If the checkpoint is older than hitl_proposal_ttl_seconds, reject it.
    # Without this check, expires_at on SQLProposal is purely decorative.
    checkpoint_ts = graph_state.created_at
    if checkpoint_ts:
        # LangGraph checkpointer stores created_at as an ISO-format string.
        try:
            created = datetime.fromisoformat(checkpoint_ts)
            # Ensure timezone-aware comparison.
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            elapsed = (datetime.now(timezone.utc) - created).total_seconds()
            if elapsed > settings.hitl_proposal_ttl_seconds:
                raise HITLError(
                    f"SQL proposal has expired ({int(elapsed)}s elapsed, "
                    f"TTL is {settings.hitl_proposal_ttl_seconds}s). "
                    "Please submit a new query."
                )
        except (ValueError, TypeError):
            # If timestamp parsing fails, log and proceed — don't block on
            # metadata format changes. TTL is a safety net, not a gate.
            pass

    # Resume the graph: Command(resume=value) feeds the value back to interrupt().
    # In sql_propose_node, `decision = interrupt(...)` will receive this dict.
    resume_value = {"approved": body.approved}
    result = await graph.ainvoke(Command(resume=resume_value), config=config)

    return ConfirmResponse(
        answer=result.get("answer"),
        executed_sql=result.get("proposed_sql"),
        role=user.role,
        rejected=not body.approved,
    )


@router.get(
    "/collections",
    summary="List collections accessible to the authenticated user",
    status_code=status.HTTP_200_OK,
)
async def get_collections(
    user: TokenPayload = Depends(get_current_user),
) -> dict[str, str | list[str]]:
    """Return the document collections this user's role can access.

    Used by the frontend to show the role badge and accessible collections sidebar.
    """
    collections = get_collections_for_role(user.role)
    return {"role": user.role, "collections": collections}


@router.get(
    "/health",
    summary="Health check",
    status_code=status.HTTP_200_OK,
)
async def health() -> dict[str, str]:
    """Simple liveness probe. No auth required.

    Returns 200 if the API process is running. Does NOT check downstream
    services (Qdrant, Groq) — that would be a readiness probe, which is
    a Phase 10 concern.
    """
    return {"status": "ok"}
