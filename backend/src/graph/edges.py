from langgraph.graph import END

from src.graph.state import MediBotState


def route_after_intent(state: MediBotState) -> str:
    """Route out of the intent gate: proceed to the router, or END on a refusal.

    resolve_intent_node sets intent_state and, on a miss, already wrote the refusal
    answer into state. A resolved intent proceeds to routing; an "empty" verdict
    (contentless, or ambiguous-with-clarify-disabled) ends immediately without retrieval.

    Returns:
        "router" when intent_state == "resolved".
        END when intent_state == "empty" (answer already set — no retrieval).
    """
    if state.get("intent_state") == "resolved":
        return "router"
    return END


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
