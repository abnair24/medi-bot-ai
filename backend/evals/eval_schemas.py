"""Pydantic shapes for golden-set records.

The golden JSON files under evals/golden/ are the eval oracle — ground truth,
synthetic only, checked in. These models validate every record ON LOAD, so a
malformed case (bad role, unknown attack type, missing field) fails immediately
with a clear error instead of surfacing as a confusing assertion later.

Type aliases (Role, Collection) are imported from src.models.schemas — never
re-declared here. A golden case with an invalid role is a validation error, not
a silent test miss.
"""

from typing import Literal

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

from src.models.schemas import Collection, Role, RouteType
from src.services.safety_guard import ViolationType

# The five adversarial vectors from docs/RBAC_REDTEAM.md, as a closed type.
AttackType = Literal[
    "direct",            # plain request for restricted content
    "injection",         # prompt-override attempt ("ignore your instructions...")
    "roleplay",          # identity spoofing ("as an admin, give me...")
    "semantic-overlap",  # same entity in multiple collections (NSTEMI)
    "indirect",          # asks about a concept that lives in a restricted doc
]


class AdversarialCase(BaseModel):
    """One RBAC red-team case (Track 2). Loaded from evals/golden/adversarial.json.

    The test executes retrieval as `attacker_role`, then inspects the returned
    chunks' `collection` metadata against `target_collection`. It checks the
    RETRIEVAL layer, not the LLM answer — a leaked chunk in context is a failure
    even if the LLM would have refused politely.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, description="Stable case ID, e.g. 'A-001'.")
    question: str = Field(min_length=1, description="The adversarial query.")
    attacker_role: Role = Field(description="Role executing the retrieval.")
    target_collection: Collection = Field(
        description="The collection under scrutiny. Normally the BLOCKED collection "
        "whose chunks must NOT appear. For a control case, an ALLOWED collection "
        "whose chunks MUST appear (see is_control)."
    )
    attack_type: AttackType
    is_control: bool = Field(
        default=False,
        description="False (default): assert NO chunk from target_collection leaks. "
        "True: assert target_collection IS retrieved — proves the filter does not "
        "over-block. Encodes cases like A-015 (technician + equipment, should pass).",
    )


class RoutingCase(BaseModel):
    """One routing-accuracy case (Track 5). Loaded from evals/golden/routing.json.

    The test runs `question` through RouterService.classify and compares the predicted
    route to `expected_route`, then aggregates all cases into a 2x2 confusion matrix.
    ANALYTICAL questions expect 'sql_rag'; KNOWLEDGE questions expect 'hybrid_rag'.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, description="Stable case ID, e.g. 'R-001'.")
    question: str = Field(min_length=1)
    expected_route: RouteType = Field(
        description="Ground-truth route: 'sql_rag' (analytical) or 'hybrid_rag' (knowledge)."
    )
    rationale: str = Field(
        default="",
        description="Why this is the correct route — documents the labeling decision, "
        "especially for boundary cases. Not asserted on; it's the case's provenance.",
    )
    is_boundary: bool = Field(
        default=False,
        description="Marks a semantically ambiguous case near the analytical/knowledge "
        "line. Lets the harness report boundary accuracy separately from clear-cut cases.",
    )


class RelevantChunk(BaseModel):
    """A (source_document, section_title) pair that contains a factual answer.

    ChunkResult does not expose a stable chunk ID, so relevance is keyed on the
    document + section a chunk came from — the identity fields ChunkResult carries.
    """

    model_config = ConfigDict(frozen=True)

    source_document: str = Field(min_length=1)
    section_title: str = Field(min_length=1)


class RetrievalCase(BaseModel):
    """One retrieval-quality case (Track 1). Loaded from evals/golden/factual.json.

    The harness runs `question` three ways (dense-only, hybrid, hybrid+rerank) and
    computes P@3 / MRR against `expected`. A retrieved chunk counts as relevant if its
    (source_document, section_title) matches ANY entry in `expected`.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, description="Stable case ID, e.g. 'F-001'.")
    question: str = Field(min_length=1)
    expected: list[RelevantChunk] = Field(
        min_length=1, description="Chunks that contain the answer — any match is relevant."
    )
    keyword_heavy: bool = Field(
        default=False,
        description="True when the answer hinges on an exact token (ICD code, fault code, "
        "drug name) — the cases where hybrid/BM25 is expected to beat dense-only.",
    )


class NumberMatchCase(BaseModel):
    """One number-match case (Track 3b). Loaded from evals/golden/numbers.json.

    Deterministic guardrail check (NO LLM judge): a drug dose / lab value / threshold
    must be quoted EXACTLY as it appears in the source — never rounded, interpolated, or
    unit-converted (guardrails.md). The test runs the pipeline and asserts every
    expected value appears verbatim (whitespace-normalized) in the generated answer.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, description="Stable case ID, e.g. 'N-001'.")
    question: str = Field(min_length=1)
    role: Role = Field(description="Role with access to the collection holding the value.")
    expected_values: list[str] = Field(
        min_length=1,
        description="Exact value strings that MUST appear in the answer (e.g. '300 mg', "
        "'₹1,20,000'). Sourced verbatim from the corpus — never a rounded/converted form.",
    )


class SQLCase(BaseModel):
    """One SQL-RAG correctness case (Track 4). Loaded from evals/golden/sql.json.

    Three-layer assertion per case:
    1. Execution success — sql_rag_service.answer() raises no exception.
    2. Keyword presence  — generated SQL references the expected table/column tokens.
    3. NL faithfulness   — NL answer contains all expected_answer_contains strings.

    expected_row_count is optional: when set, verifies the SQL result has exactly that
    many rows (useful for COUNT queries). When None, only NL keyword assertions are made.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, description="Stable case ID, e.g. 'S-001'.")
    question: str = Field(min_length=1)
    role: Role = Field(description="Must be billing_executive or admin — SQL RAG is gated.")
    expected_sql_keywords: list[str] = Field(
        description="Tokens that MUST appear in the generated SQL string (case-insensitive). "
        "Validates the LLM targeted the correct table and column.",
    )
    expected_answer_contains: list[str] = Field(
        description="Strings that MUST appear in the NL answer. Empty list skips NL check.",
    )
    expected_row_count: int | None = Field(
        default=None,
        description="If set, assert the SQL result has exactly this many rows.",
    )


class SafetyCase(BaseModel):
    """One medical-safety guardrail case (Track 11). Loaded from evals/golden/safety.json.

    Tests run_safety_checks() deterministically — no LLM, no network. Each case supplies
    a pre-written answer string and the checks it should trigger (or not trigger).
    The safety_guard module is the single source of truth for detection logic.

    is_clinical controls whether MISSING_DISCLAIMER can be triggered. Non-clinical answers
    cannot violate that rule regardless of their content.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, description="Stable case ID, e.g. 'SG-001'.")
    description: str = Field(description="Human-readable intent of this case.")
    answer: str = Field(min_length=1, description="Pre-written answer to run through safety_guard.")
    is_clinical: bool = Field(description="True when answer is from a clinical/nursing collection.")
    expected_violations: list[ViolationType] = Field(
        description="Zero or more ViolationType members expected from run_safety_checks(). "
        "Empty list means the answer is fully compliant.",
    )
