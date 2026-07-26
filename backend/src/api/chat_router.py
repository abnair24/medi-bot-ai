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
    ClarifyProposal,
    ClarifyResolveRequest,
    ConfirmRequest,
    ConfirmResponse,
    SQLProposal,
)
from src.security.jwt import get_current_user
from src.utils.exceptions import HITLError, RBACError
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
) -> ChatResponse | SQLProposal | ClarifyProposal:
    """Invoke the MediBot graph for a single question.

    Role is ALWAYS taken from the JWT — body.role is overridden.
    This prevents privilege escalation via forged request bodies.

    Returns:
        ChatResponse for completed answers (hybrid or SQL RAG).
        SQLProposal when HITL is enabled and SQL needs user approval.
        ClarifyProposal when a follow-up is ambiguous and needs disambiguation.
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

    # --- Hard turn limit ---
    # Count completed turns from the full (pre-window) history. Past the cap, refuse and
    # tell the user to start a new chat. Distinct from the rewriter window below.
    prior_turns = len(history) // 2
    if prior_turns >= settings.max_turns_per_session:
        return ChatResponse(
            answer=(
                f"You've reached the {settings.max_turns_per_session}-turn limit for this "
                "chat. Please start a new chat to continue."
            ),
            sources=[],
            retrieval_type="hybrid_rag",
            role=role,
        )

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

    # Register/refresh this session under the user so it appears in their chat list.
    if body.session_id:
        await request.app.state.chat_sessions.touch(body.session_id, user.sub)

    token = body.session_id or request_id
    return await _finalize_turn(graph, config, result, request_id, role, token)


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


@router.post(
    "/chat/clarify",
    summary="Answer an ambiguity clarification and resume the graph",
    status_code=status.HTTP_200_OK,
)
async def clarify_chat(
    body: ClarifyResolveRequest,
    request: Request,
    user: TokenPayload = Depends(get_current_user),
) -> ChatResponse | SQLProposal | ClarifyProposal:
    """Resume a graph suspended at the intent gate with the user's chosen topic.

    The confirmation_token is the session_id / thread_id. The user's clarification string
    is fed back to interrupt() in resolve_intent_node, becomes the resolved retrieval query,
    and the graph runs on to produce an answer — which may itself pause again for SQL HITL.

    Raises:
        HITLError: if no clarification is pending for this token.
    """
    config = {"configurable": {"thread_id": body.confirmation_token}}
    graph = request.app.state.graph
    graph_state = await graph.aget_state(config)

    if not graph_state or not graph_state.next or "resolve_intent" not in graph_state.next:
        raise HITLError(
            "No pending clarification for this token — it may have expired or "
            "already been answered."
        )

    request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())
    # Resume: the clarification string is returned by interrupt() in resolve_intent_node.
    result = await graph.ainvoke(Command(resume=body.clarification), config=config)
    return await _finalize_turn(
        graph, config, result, request_id, user.role, body.confirmation_token
    )


def _first_interrupt_value(result: object) -> dict:
    """Extract the payload dict passed to interrupt() from an ainvoke result.

    When a node calls interrupt(payload), ainvoke returns the accumulated state with a
    "__interrupt__" entry holding Interrupt objects whose .value is that payload. Returns
    {} when no interrupt is present or the payload isn't a dict.
    """
    if not isinstance(result, dict):
        return {}
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return {}
    value = getattr(interrupts[0], "value", None)
    return value if isinstance(value, dict) else {}


async def _finalize_turn(
    graph: object,
    config: dict,
    result: dict,
    request_id: str,
    role: str,
    token: str,
) -> ChatResponse | SQLProposal | ClarifyProposal:
    """Turn an ainvoke result into a response: clarify pause, SQL pause, or completion.

    Shared by /chat and /chat/clarify so a resumed turn that pauses again (e.g. clarify
    then SQL HITL) is handled identically to a first-pass pause.
    """
    graph_state = await graph.aget_state(config)
    if graph_state.next:
        pending = graph_state.next
        interrupt_value = _first_interrupt_value(result)
        expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=settings.hitl_proposal_ttl_seconds
        )

        # Intent gate paused for disambiguation → return a ClarifyProposal.
        if "resolve_intent" in pending:
            candidates = interrupt_value.get("candidates", [])
            log_request_event(
                request_id=request_id,
                role=role,
                event="clarify_proposal_sent",
                candidate_count=len(candidates),
            )
            return ClarifyProposal(
                prompt=interrupt_value.get(
                    "clarify_prompt", "Could you clarify which topic you mean?"
                ),
                candidates=candidates,
                confirmation_token=token,
                expires_at=expires_at,
            )

        # Otherwise SQL HITL paused → return the SQL proposal.
        proposed_sql = interrupt_value.get("proposed_sql") or result.get("proposed_sql", "")
        log_request_event(
            request_id=request_id,
            role=role,
            event="hitl_proposal_sent",
            proposed_sql_length=len(proposed_sql),
        )
        return SQLProposal(
            proposed_sql=proposed_sql,
            confirmation_token=token,
            expires_at=expires_at,
        )

    # --- Normal completion: build ChatResponse ---
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


@router.get(
    "/chats",
    summary="List the authenticated user's past chat sessions",
    status_code=status.HTTP_200_OK,
)
async def list_chats(
    request: Request,
    user: TokenPayload = Depends(get_current_user),
) -> dict[str, list[dict]]:
    """Return this user's chat sessions (newest first) — metadata only, no query text."""
    sessions = await request.app.state.chat_sessions.list_for_user(user.sub)
    return {"sessions": sessions}


@router.get(
    "/chats/{session_id}",
    summary="Load a past chat's messages (owner only)",
    status_code=status.HTTP_200_OK,
)
async def load_chat(
    session_id: str,
    request: Request,
    user: TokenPayload = Depends(get_current_user),
) -> dict[str, object]:
    """Reconstruct a past conversation from the checkpointer for the owning user.

    Raises RBACError (→403) if the session belongs to another user. Messages are text-only
    (role + content) — historical retrieval badges/sources aren't persisted per message.
    """
    owner = await request.app.state.chat_sessions.owner_of(session_id)
    if owner is not None and owner != user.sub:
        raise RBACError("This chat belongs to another user.")

    graph = request.app.state.graph
    state = await graph.aget_state({"configurable": {"thread_id": session_id}})
    values = state.values if state else {}
    messages: list[dict] = list(values.get("history", []))
    last_q = values.get("original_question")
    last_a = values.get("answer")
    if last_q and last_a:
        messages += [
            {"role": "user", "content": last_q},
            {"role": "assistant", "content": last_a},
        ]
    return {"session_id": session_id, "messages": messages}


@router.delete(
    "/chats/{session_id}",
    summary="Delete a past chat (owner only) — index row + checkpointer content",
    status_code=status.HTTP_200_OK,
)
async def delete_chat(
    session_id: str,
    request: Request,
    user: TokenPayload = Depends(get_current_user),
) -> dict[str, str]:
    """Permanently delete a chat: its session-index row AND its conversation content.

    Owner-gated (→403 for another user's chat). Purges both stores so nothing is
    orphaned: the ChatSessionStore index row and the LangGraph checkpointer thread
    (all persisted turns for this session_id). Idempotent — deleting an already-gone
    chat returns ok.

    Raises:
        RBACError: if the session belongs to another user (→403).
    """
    store = request.app.state.chat_sessions
    owner = await store.owner_of(session_id)
    if owner is not None and owner != user.sub:
        raise RBACError("This chat belongs to another user.")

    # Purge conversation content from the checkpointer (guarded: older checkpointers
    # may lack adelete_thread — the index delete below still removes it from the UI).
    graph = request.app.state.graph
    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is not None and hasattr(checkpointer, "adelete_thread"):
        await checkpointer.adelete_thread(session_id)

    await store.delete(session_id)

    log_request_event(
        request_id=getattr(request.state, "request_id", None) or session_id,
        role=user.role,
        event="chat_deleted",
    )
    return {"status": "deleted", "session_id": session_id}


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
