from langgraph.types import interrupt

from src.config.settings import settings
from src.graph.state import MediBotState
from src.services.denial_service import verbalize_denial
from src.services.generation_service import GenerationService
from src.services.intent_service import IntentService
from src.services.rerank_service import RerankService
from src.services.retrieval_service import RetrievalService
from src.services.router_service import RouterService
from src.services.sql_rag_service import SQLRAGService
from src.utils.exceptions import RBACError

# Shown when a follow-up can't be pinned to one prior topic and HITL clarify is disabled.
_AMBIGUOUS_REFUSAL = (
    "Your question refers back to an earlier topic, but more than one could apply. "
    "Please ask again naming the specific topic you mean."
)

# Shown when the message has no content and no prior topic to inherit (state=empty).
_EMPTY_REFUSAL = (
    "I couldn't identify a question to answer. Please ask a full question — for example, "
    "name the protocol, drug, code, or policy you need."
)


# Two-axis router: query specificity → how many reranked chunks feed generation.
# Values live in settings (no hardcoding) — this maps the specificity label to the
# corresponding setting. aggregate uses the default reranker_final_k.
def _final_k_for_specificity(specificity: str | None) -> int:
    """Resolve the reranker cutoff from the router's specificity classification."""
    if specificity == "point_lookup":
        return settings.reranker_final_k_point_lookup
    if specificity == "exploratory":
        return settings.reranker_final_k_exploratory
    return settings.reranker_final_k  # aggregate / unknown → default


def make_resolve_intent_node(svc: IntentService):
    """Return the intent-gate node — admission control in front of the router.

    Three outcomes (see IntentService / route_after_intent):
      * resolved  → overwrite `question` with the self-contained query, continue to router.
      * empty     → set the refusal answer and intent_state="empty"; the edge routes to END
                    (never retrieves).
      * ambiguous → if HITL clarify is enabled, interrupt() and ask the user which topic they
                    meant; the resume value becomes the resolved query and we continue. If
                    disabled, refuse deterministically (state="empty") — we never guess.
    """

    async def resolve_intent_node(state: MediBotState) -> dict:
        verdict = await svc.resolve(
            question=state["question"],
            history=state.get("history", []),
            request_id=state["request_id"],
            role=state["role"],
        )

        if verdict.state == "resolved":
            # Overwrite question in place — all downstream nodes read the same key.
            return {
                "question": verdict.resolved_query or state["question"],
                "intent_state": "resolved",
            }

        if verdict.state == "ambiguous":
            if not settings.intent_clarify_enabled:
                return {"answer": _AMBIGUOUS_REFUSAL, "sources": [], "intent_state": "empty"}
            # Pause and ask the user which prior topic they meant. On resume, interrupt()
            # returns the clarification string, which becomes the resolved retrieval query.
            clarification = interrupt(
                {"clarify_prompt": _build_clarify_prompt(verdict.candidates),
                 "candidates": verdict.candidates}
            )
            resolved = (clarification or "").strip() or state["question"]
            return {"question": resolved, "intent_state": "resolved",
                    "clarify_candidates": verdict.candidates}

        # empty
        return {"answer": _EMPTY_REFUSAL, "sources": [], "intent_state": "empty"}

    return resolve_intent_node


def _build_clarify_prompt(candidates: list[str]) -> str:
    """Build the user-facing 'did you mean X or Y?' question from candidate topics."""
    if len(candidates) >= 2:
        options = ", ".join(candidates[:-1]) + f", or {candidates[-1]}"
        return f"Your question could refer to a few topics. Did you mean: {options}?"
    return "Could you clarify which topic your question refers to?"


def make_router_node(svc: RouterService):
    """Return a node that classifies the question on two axes: route × specificity.

    route        → hybrid_rag vs sql_rag (which path to take).
    specificity  → point_lookup / aggregate / exploratory (how deep to rerank).
    Both are written to state; the rerank node reads specificity to size its output.
    """

    async def router_node(state: MediBotState) -> dict:
        decision = await svc.classify_detailed(
            question=state["question"],
            request_id=state["request_id"],
            role=state["role"],
        )
        return {"route": decision.route, "specificity": decision.specificity}

    return router_node


def make_retrieve_node(svc: RetrievalService):
    """Return a node that runs hybrid retrieval with RBAC pre-filter."""

    async def retrieve_node(state: MediBotState) -> dict:
        chunks = await svc.search(
            query=state["question"],
            role=state["role"],
            request_id=state["request_id"],
        )
        # Access-miss probe: which collections OUTSIDE this role match the query? Lets the
        # generate node show an RBAC denial (not a generic "no relevant protocol") when the
        # answer lives in a collection the user can't see. Never surfaces restricted content.
        restricted = await svc.restricted_matches(
            query=state["question"],
            role=state["role"],
            request_id=state["request_id"],
            user_chunks=chunks,
        )
        return {"chunks": chunks, "restricted_matches": restricted}

    return retrieve_node


def make_rerank_node(svc: RerankService):
    """Return a node that reranks retrieved chunks with the cross-encoder."""

    async def rerank_node(state: MediBotState) -> dict:
        # Two-axis router: size the reranked context by query specificity.
        final_k = _final_k_for_specificity(state.get("specificity"))
        reranked = await svc.rerank(
            query=state["question"],
            chunks=state["chunks"],
            request_id=state["request_id"],
            role=state["role"],
            final_k=final_k,
        )
        # Replace chunks with top-k reranked subset.
        return {"chunks": reranked}

    return rerank_node


def make_generate_node(svc: GenerationService):
    """Return a node that generates a grounded answer from reranked chunks."""

    async def generate_node(state: MediBotState) -> dict:
        answer, sources = await svc.generate(
            question=state["question"],
            chunks=state["chunks"],
            role=state["role"],
            request_id=state["request_id"],
            restricted_matches=state.get("restricted_matches", []),
        )
        return {"answer": answer, "sources": sources}

    return generate_node


def make_sql_propose_node(svc: SQLRAGService):
    """Return a node that generates SQL and optionally pauses for human approval.

    HITL flow (sql_hitl_enabled=True):
      1. Generate SQL, store in state["proposed_sql"].
      2. Call interrupt() — graph suspends, checkpoints state, returns proposal to API.
      3. On resume: interrupt() returns the user's decision dict.
      4. If approved: return proposed_sql so sql_execute_node can run it.
      5. If rejected: set answer to rejection message, proposed_sql=None.
         route_after_propose edge sends this directly to END.

    Non-HITL flow (sql_hitl_enabled=False):
      Steps 2-5 are skipped. proposed_sql is set and sql_execute_node runs immediately.
    """

    async def sql_propose_node(state: MediBotState) -> dict:
        role = state["role"]
        # Use original_question for SQL generation, not the rewrite_node output.
        # Rewrite augments the question with prior-turn context ("given 0 escalated")
        # which causes narration to produce comparisons the user didn't ask for.
        sql_question = state.get("original_question") or state["question"]
        try:
            clean_sql = await svc.generate_sql(
                question=sql_question,
                role=role,
                request_id=state["request_id"],
            )
        except RBACError:
            # Role is not authorized for SQL RAG. Return a conversational denial
            # rather than propagating an HTTP 403 — the user is authenticated;
            # they just lack the feature permission. route_after_propose sees the
            # answer field and routes directly to END without executing SQL.
            # Use the deterministic denial verbalizer (single source of truth). Do NOT
            # enumerate which roles CAN run SQL — naming privileged roles is an
            # unnecessary disclosure (tells the user who to target) and violates the
            # guardrail that a refusal states only the user's own role + access.
            denial = verbalize_denial(role)
            sql_note = " Analytical database queries are not available at your access level."
            return {
                "answer": denial.message + sql_note,
                "proposed_sql": None,
                "sources": [],
            }

        if settings.sql_hitl_enabled:
            # Pause graph and return SQL proposal to the human.
            # On resume, interrupt() returns the decision dict from ConfirmRequest.
            decision = interrupt({"proposed_sql": clean_sql})

            if not decision.get("approved", False):
                return {
                    "answer": "SQL execution was rejected. No data was queried.",
                    "proposed_sql": None,
                    "sources": [],
                }

        return {"proposed_sql": clean_sql}

    return sql_propose_node


def make_sql_execute_node(svc: SQLRAGService):
    """Return a node that executes approved SQL and narrates the result."""

    async def sql_execute_node(state: MediBotState) -> dict:
        sql = state["proposed_sql"]
        sql_question = state.get("original_question") or state["question"]
        answer, _result = await svc.execute_and_narrate(
            sql=sql,
            question=sql_question,
            role=state["role"],
            request_id=state["request_id"],
        )
        # sources is empty for SQL RAG — answers cite row data, not document chunks.
        return {"answer": answer, "sources": []}

    return sql_execute_node
