from langgraph.graph import END

from src.graph.state import MediBotState


def should_rewrite(state: MediBotState) -> str:
    """Skip rewrite node on the first turn — no history means no ambiguity to resolve.

    Returns:
        "rewrite" if prior conversation history exists.
        "router" to skip directly to routing on the first turn.
    """
    if state.get("history"):
        return "rewrite"
    return "router"


def route_after_router(state: MediBotState) -> str:
    """Route to the SQL RAG or hybrid RAG path based on the router's decision.

    Returns:
        "sql_rag" or "hybrid_rag" — matches the node names registered in builder.py.
    """
    return state["route"]


def route_after_propose(state: MediBotState) -> str:
    """After sql_propose_node: route to execution or end if HITL rejected or RBAC denied.

    Checks proposed_sql (not answer) because answer may hold a stale value from a
    prior turn in the same multi-turn session — it is only reset by generate_node or
    sql_execute_node, never by sql_propose_node on the normal path.

    sql_propose_node sets proposed_sql=None on HITL rejection or RBAC denial (and sets
    answer to the explanation). On the normal path it sets proposed_sql to the SQL string.

    Returns:
        "sql_execute" if proposed_sql is set (normal path, or HITL approved).
        END if proposed_sql is None (HITL rejected, RBAC denied).
    """
    if state.get("proposed_sql"):
        return "sql_execute"
    return END
