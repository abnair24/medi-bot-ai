from typing import TypedDict

from src.models.chunk import ChunkResult
from src.models.schemas import Source


class MediBotState(TypedDict, total=False):
    """LangGraph state shared across all graph nodes.

    total=False — all keys are optional at the TypedDict level.
    In practice, question/role/request_id are always set at graph entry.
    All other fields are populated progressively by nodes.

    Nodes return a partial dict — only the keys they update. LangGraph
    merges the partial dict into this state; unmentioned keys are unchanged.
    """

    # --- Set at graph entry (always present) ---
    question: str           # user question; overwritten by rewrite_node when history exists
    original_question: str  # unmodified user question — used by SQL path to avoid
                            # rewrite-injected comparison context polluting narration
    role: str               # authenticated user role from JWT
    request_id: str         # trace ID for log correlation across all nodes
    history: list[dict]     # prior turns: [{"role": "user"|"assistant", "content": str}]
    session_id: str | None  # LangGraph thread_id — used for multi-turn checkpointing

    # --- Set by router_node (two-axis router) ---
    route: str | None        # "hybrid_rag" or "sql_rag"
    specificity: str | None  # "point_lookup" | "aggregate" | "exploratory" — tunes rerank depth

    # --- Set by retrieve_node and rerank_node (hybrid_rag path) ---
    chunks: list[ChunkResult]

    # --- Set by generate_node (hybrid_rag path) ---
    sources: list[Source]

    # --- Set by sql_propose_node (sql_rag path) ---
    proposed_sql: str | None   # LLM-generated clean SQL; None if rejected

    # --- Set by generate_node or sql_execute_node (final answer, all paths) ---
    answer: str | None
