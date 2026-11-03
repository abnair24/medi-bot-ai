from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

# --- Type aliases ---
# Defined here once. Import from this module — never use bare strings in business logic.
Role = Literal["doctor", "nurse", "billing_executive", "technician", "admin"]
Collection = Literal["general", "clinical", "nursing", "billing", "equipment"]
RouteType = Literal["hybrid_rag", "sql_rag"]
ChunkType = Literal["text", "table", "heading", "code"]

# Two-axis router second dimension: HOW specific the query is.
# point_lookup  — single value requested ("what is the dose of X?")
# aggregate     — computed over many records ("how many claims were rejected?")
# exploratory   — broad topic ("tell me about the NSTEMI protocol")
QuerySpecificity = Literal["point_lookup", "aggregate", "exploratory"]

# Intent-gate verdict (admission control, runs BEFORE the two-axis router).
# resolved  — a self-contained information need exists → proceed to retrieval/SQL.
# ambiguous — a back-reference the history can't pin to ONE topic → ask the user (HITL).
# empty     — no content and no recoverable prior intent → refuse, never retrieve.
IntentState = Literal["resolved", "ambiguous", "empty"]


# --- Request models ---

class ChatRequest(BaseModel):
    """Incoming /chat request payload."""

    question: str = Field(..., min_length=1, max_length=2000)
    role: Role
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for multi-turn conversation memory.",
    )


# --- Response sub-models ---

class Source(BaseModel):
    """A single source citation included in a chat response."""

    model_config = ConfigDict(frozen=True)

    source_document: str
    section_title: str
    collection: Collection


# --- Response models ---

class ChatResponse(BaseModel):
    """Outgoing /chat response payload."""

    model_config = ConfigDict(frozen=True)

    answer: str
    sources: list[Source]
    retrieval_type: RouteType
    role: Role


class RouteDecision(BaseModel):
    """Structured output from the semantic router LLM call."""

    model_config = ConfigDict(frozen=True)

    route: RouteType
    reasoning: str = Field(
        default="",
        description="Router chain-of-thought — logged internally, never returned to user",
    )


class TwoAxisRouteDecision(BaseModel):
    """Two-axis routing decision: route type × query specificity.

    route        — where the answer lives (SQL database vs document collection).
    specificity  — how the query is shaped, used to tune downstream behaviour:
                   point_lookup  → aggressive reranking (top-1 suffices)
                   aggregate     → SQL execution or broad document scan
                   exploratory   → larger retrieval window, more context in prompt
    """

    model_config = ConfigDict(frozen=True)

    route: RouteType
    specificity: QuerySpecificity
    reasoning: str = Field(
        default="",
        description="Two-axis chain-of-thought — logged internally, never returned to user",
    )


class IntentResolution(BaseModel):
    """Verdict from the intent gate — admission control in front of the router.

    state           — resolved / ambiguous / empty (see IntentState).
    resolved_query  — the self-contained retrieval query (set only when state=resolved).
    candidates      — competing prior topics to disambiguate between (state=ambiguous).
                      Short topic labels, NEVER raw patient text — surfaced to the user
                      as clarify options, never logged.
    reason          — one-line rationale; logged as operational metadata only.
    """

    model_config = ConfigDict(frozen=True)

    state: IntentState
    resolved_query: str | None = None
    candidates: list[str] = Field(default_factory=list)
    reason: str = ""


class SQLResult(BaseModel):
    """Validated output from SQL execution before passing to narration LLM.

    patient_name and patient_id must be stripped from rows before this model
    is constructed — enforced in sql_rag_service, not here.
    """

    model_config = ConfigDict(frozen=True)

    query: str = Field(description="The SQL string that was executed")
    rows: list[dict] = Field(description="Raw rows returned — PHI stripped upstream")
    row_count: int


# --- HITL SQL models (human-in-the-loop approval flow) ---

class SQLProposal(BaseModel):
    """Returned when HITL is enabled: proposed SQL awaiting user approval."""

    model_config = ConfigDict(frozen=True)

    proposed_sql: str
    confirmation_token: str
    needs_confirmation: Literal[True] = True
    expires_at: datetime


class ConfirmRequest(BaseModel):
    """User's approval or rejection of a proposed SQL query."""

    confirmation_token: str
    approved: bool


# --- HITL clarify models (ambiguous-intent disambiguation flow) ---

class ClarifyProposal(BaseModel):
    """Returned when the intent gate can't pin a follow-up to one prior topic.

    The graph is suspended at an interrupt() in the resolve_intent node; the user
    picks a candidate (or types a clarification) and the graph resumes.
    """

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(description="User-facing question, e.g. 'Did you mean X or Y?'")
    candidates: list[str] = Field(description="Competing prior topics to choose from")
    confirmation_token: str
    needs_clarification: Literal[True] = True
    expires_at: datetime


class ClarifyResolveRequest(BaseModel):
    """User's clarification for an ambiguous follow-up (resumes the suspended graph)."""

    confirmation_token: str
    clarification: str = Field(..., min_length=1, max_length=2000)


class ConfirmResponse(BaseModel):
    """Result after user approves or rejects a proposed SQL query."""

    model_config = ConfigDict(frozen=True)

    answer: str | None = Field(default=None, description="None if rejected")
    executed_sql: str | None = None
    retrieval_type: Literal["sql_rag"] = "sql_rag"
    role: str
    rejected: bool = False
