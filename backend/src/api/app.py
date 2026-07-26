"""FastAPI application entrypoint — lifespan, exception handlers, router mounting.

This is the composition root: all heavy resources (graph, checkpointer) are
constructed once at startup and stored on app.state. Exception handlers map
custom exceptions to HTTP responses. Routers are mounted last.

Start with: uvicorn src.api.app:app --reload
"""

import os
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.api.auth_router import router as auth_router
from src.api.chat_router import router as chat_router
from src.config.settings import settings
from src.graph.builder import build_graph
from src.services.chat_session_store import ChatSessionStore
from src.utils import token_meter
from src.utils.exceptions import (
    AuthenticationError,
    HITLError,
    MediBotError,
    RateLimitedError,
    RBACError,
    RetrievalError,
    SQLGenerationError,
)
from src.utils.helpers import generate_request_id
from src.utils.logger import logger
from src.utils.timing import stopwatch


# --- Lifespan: startup + shutdown in one place ---
# Python equivalent of Spring's @PostConstruct / @PreDestroy.
# Everything before `yield` runs at startup; everything after runs at shutdown.

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle: construct graph, open checkpointer, tear down."""

    # --- JWT secret guard ---
    # Crashes in non-dev environments if the default secret is still set.
    # Locally, ENVIRONMENT defaults to "dev" so this won't block development.
    env = os.getenv("ENVIRONMENT", "dev")
    if settings.jwt_secret == "change-this-in-production":
        if env != "dev":
            raise RuntimeError(
                "JWT_SECRET must be changed in non-dev environments. "
                "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
            )
        logger.warning("Using default JWT secret — dev only. Do NOT deploy this.")

    # --- Open async checkpointer (SQLite for LangGraph multi-turn memory) ---
    # async with ensures the connection is closed on shutdown.
    async with AsyncSqliteSaver.from_conn_string(
        settings.checkpoint_db_path
    ) as checkpointer:
        # --- Compile the graph once ---
        # build_graph() is the composition root for the RAG pipeline:
        # adapters → services → node factories → graph assembly → compile.
        compiled_graph = build_graph(checkpointer)
        app.state.graph = compiled_graph

        # Per-user chat-session index (list past chats after re-login).
        app.state.chat_sessions = ChatSessionStore()

        logger.info(
            "app_startup_complete",
            extra={
                "environment": env,
                "checkpoint_db": settings.checkpoint_db_path,
                "hitl_enabled": settings.sql_hitl_enabled,
            },
        )

        yield  # --- app is running ---

    # After yield: shutdown. AsyncSqliteSaver closes its connection via async with.
    logger.info("app_shutdown_complete")


# --- FastAPI app ---

app = FastAPI(
    title="MediBot API",
    description="Internal AI assistant for the Health Network. "
    "Hybrid RAG + SQL RAG with RBAC enforcement.",
    version="0.1.0",
    lifespan=lifespan,
)


# --- CORS middleware ---
# Required because Streamlit (port 8700) and FastAPI (port 8000) are different origins.
# In production, restrict origins to the actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # permissive for dev — restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request-context middleware: request_id, latency, token cost (all PHI-safe) ---
@app.middleware("http")
async def request_context(request: Request, call_next: Callable) -> Response:
    """Assign a request_id, time the request, and expose token cost.

    - request_id: taken from an inbound X-Request-ID (trace propagation) or generated.
      Stashed on request.state so handlers reuse the same ID instead of minting their own.
    - token_meter reset per request; the total is read AFTER the handler runs.
    - Response headers: X-Request-ID (so a user can quote it when reporting an issue) and
      X-LLM-Tokens (per-request Groq token spend — cheap cost visibility).
    Only IDs, a duration, and counts are logged — never query or answer text.
    """
    request_id = request.headers.get("X-Request-ID") or generate_request_id()
    request.state.request_id = request_id
    token_meter.reset()

    with stopwatch() as ms:
        response = await call_next(request)

    tokens = token_meter.total_tokens()
    response.headers["X-Request-ID"] = request_id
    response.headers["X-LLM-Tokens"] = str(tokens)
    logger.info(
        "http_request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": ms(),
            "llm_tokens": tokens,
            "llm_calls": token_meter.call_count(),
        },
    )
    return response


# --- Exception handlers ---
# Map custom exceptions to HTTP responses. Equivalent of Spring's @ControllerAdvice.
# Order doesn't matter — FastAPI matches the most specific exception type.

@app.exception_handler(AuthenticationError)
async def authentication_handler(request: Request, exc: AuthenticationError) -> JSONResponse:
    """401 — invalid credentials or expired/tampered JWT."""
    return JSONResponse(
        status_code=401,
        content={"detail": str(exc)},
    )


@app.exception_handler(RBACError)
async def rbac_handler(request: Request, exc: RBACError) -> JSONResponse:
    """403 — role doesn't have access to the requested resource."""
    return JSONResponse(
        status_code=403,
        content={"detail": str(exc)},
    )


@app.exception_handler(HITLError)
async def hitl_handler(request: Request, exc: HITLError) -> JSONResponse:
    """409 — HITL token invalid, expired, or already used."""
    return JSONResponse(
        status_code=409,
        content={"detail": str(exc)},
    )


@app.exception_handler(RetrievalError)
async def retrieval_handler(request: Request, exc: RetrievalError) -> JSONResponse:
    """502 — upstream vector search failed (Qdrant unreachable, etc.)."""
    # Don't expose internal error details to the client.
    logger.error("retrieval_error", extra={"detail": str(exc)})
    return JSONResponse(
        status_code=502,
        content={"detail": "Search service is currently unavailable."},
    )


@app.exception_handler(SQLGenerationError)
async def sql_generation_handler(request: Request, exc: SQLGenerationError) -> JSONResponse:
    """500 — SQL generation or execution failed."""
    logger.error("sql_generation_error", extra={"detail": str(exc)})
    return JSONResponse(
        status_code=500,
        content={"detail": "SQL query processing failed."},
    )


@app.exception_handler(RateLimitedError)
async def rate_limited_handler(request: Request, exc: RateLimitedError) -> JSONResponse:
    """429 — LLM provider rate-limited the request past the retry budget.

    The message is already user-facing ("You are rate limited (daily limit). Please try
    again in 4m33s."). retry_after is also echoed in the standard Retry-After header so a
    well-behaved client can honour it.
    """
    logger.warning(
        "llm_rate_limited",
        extra={"limit_kind": exc.limit_kind, "retry_after": exc.retry_after},
    )
    headers = {"Retry-After": exc.retry_after} if exc.retry_after else {}
    return JSONResponse(
        status_code=429,
        content={
            "detail": exc.message,
            "retry_after": exc.retry_after,
            "limit_kind": exc.limit_kind,
        },
        headers=headers,
    )


@app.exception_handler(MediBotError)
async def medibot_error_handler(request: Request, exc: MediBotError) -> JSONResponse:
    """500 — catch-all for any MediBotError not handled by a more specific handler."""
    logger.error("unhandled_medibot_error", extra={"detail": str(exc)})
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred."},
    )


# --- Mount routers ---
# No prefix — endpoints sit at the API root (/login, /chat, /health, etc.)
app.include_router(auth_router)
app.include_router(chat_router)

# Eval dashboard router is admin-only AND feature-flagged: only mounted when
# evals_enabled is True (dev/demo). In prod it doesn't exist at all → /evals/* 404.
if settings.evals_enabled:
    from src.api.eval_router import router as eval_router

    app.include_router(eval_router)
    logger.info("eval_router_mounted")
