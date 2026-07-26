from src.adapters.llm_adapter import LLMAdapter
from src.config.settings import settings
from src.models.chunk import ChunkResult
from src.models.schemas import Source
from src.services.denial_service import verbalize_denial
from src.services.safety_guard import run_safety_checks
from src.utils.logger import log_request_event
from src.utils.timing import stopwatch

# ---------------------------------------------------------------------------
# Prompts — loaded as module-level constants per PROMPTS.md guidance.
# Never inline multi-line prompts in business logic.
# ---------------------------------------------------------------------------

_RAG_PROMPT = """\
You are MediBot, a retrieval assistant for the Health Network. You SURFACE
relevant protocol text. You do NOT give medical advice, diagnoses, or treatment decisions.

Rules:
- Answer ONLY from the context below. Never guess, fabricate, or use outside knowledge.
- If NONE of the provided context is relevant to the question, respond with EXACTLY this
  token and nothing else: NO_RELEVANT_PROTOCOL
- Quote clinical values (doses, temperatures, thresholds, codes, prices) EXACTLY as they
  appear in the context. NEVER convert units, round, interpolate, or compute a value —
  not even a value from the user's question.
- Do NOT tell the user what to do. Never state a course of action, prescribe, recommend,
  or advise. Surface the relevant protocol and let the qualified clinician decide.
- Present the relevant protocol text as verbatim Markdown blockquotes, quoting clinical
  values exactly. Do NOT append source names, filenames, or section titles to the
  excerpts, and do NOT add a "Sources" list — the citations are shown to the user
  separately in a dedicated panel, so repeating them here is redundant.
- Reply with the relevant protocol content only. Do not add commentary, patient-specific
  reasoning, or clinical interpretation.

Context:
{context}

User question: {question}

Relevant protocol excerpts:\
"""

# Sentinel the LLM emits when no retrieved chunk is relevant to the question.
# Gating sources on the LLM's relevance judgment is far more reliable than a raw
# reranker-score floor: the score distributions of relevant vs irrelevant answers
# overlap (a correct answer can score -5.9 while an irrelevant one scores -2.7), so no
# single threshold separates them. The LLM reads the full context and decides.
_NO_RELEVANT_SENTINEL = "NO_RELEVANT_PROTOCOL"

_NO_RELEVANT_MESSAGE = (
    "No relevant protocol was found for this question. Please rephrase, or "
    "consult the appropriate department or a qualified clinician for guidance."
)

# ---------------------------------------------------------------------------
# Disclaimers — appended deterministically by the service, never by the LLM.
# The LLM does not decide whether a disclaimer is needed (GUARDRAILS.md §4).
# ---------------------------------------------------------------------------

_CLINICAL_DISCLAIMER = (
    "\n\n*This information is from internal protocols. Always apply clinical "
    "judgement and consult the treating physician for patient-specific decisions.*"
)

_DRUG_DISCLAIMER = (
    "\n\n*Verify dose and contraindications against the patient's specific condition "
    "before prescribing. Refer to the full formulary entry.*"
)

# Collections whose chunks trigger the clinical disclaimer.
_CLINICAL_COLLECTIONS = {"clinical", "nursing"}

# Distinctive substring of _CLINICAL_DISCLAIMER — injected into the safety guard so it can
# verify the deterministic disclaimer append actually happened (defense in depth).
_DISCLAIMER_MARKERS = ("Always apply clinical judgement",)

# Shown to the user when the runtime guard blocks a non-compliant answer (fail-closed).
# Never contains the blocked answer text. Sources are still returned so the clinician can
# consult the protocol directly.
_SAFE_FALLBACK = (
    "MediBot can only surface protocol excerpts and cannot provide a recommendation or "
    "course of action for this query. Please consult the cited protocol directly, or a "
    "qualified clinician, for a patient-specific decision."
)

# source_document substrings that indicate drug content — triggers drug disclaimer
# on top of the clinical disclaimer.
_DRUG_SOURCE_KEYWORDS = {"drug_formulary", "formulary"}


class GenerationService:
    """Builds grounded answers from reranked chunks using the Groq LLM.

    Handles two paths:
    - chunks present: RAG generation with deterministic disclaimers.
    - no chunks (RBAC blocked everything): polite refusal via LLM.
    """

    def __init__(self, llm: LLMAdapter) -> None:
        self._llm = llm

    async def generate(
        self,
        question: str,
        chunks: list[ChunkResult],
        role: str,
        request_id: str,
        restricted_matches: list[str] | None = None,
    ) -> tuple[str, list[Source]]:
        """Generate a grounded answer from reranked chunks.

        On a MISS (no chunks, or nothing relevant), the response depends on WHY:
        - the query's answer lives in a collection the role can't access
          (restricted_matches non-empty) → RBAC denial ("not available at your access
          level"), so a nurse asking about billing gets a clear access message.
        - the topic simply isn't in the corpus → "no relevant protocol" message.

        Args:
            question: user's natural-language question. Passed to LLM but never logged.
            chunks: top-k ChunkResult objects from RerankService (may be empty).
            role: authenticated user role, for refusal phrasing and log correlation.
            request_id: trace ID for correlating log records across pipeline stages.
            restricted_matches: collections OUTSIDE the role that the query matched
                (from RetrievalService.restricted_matches). Names only — never content.

        Returns:
            Tuple of (answer_text, sources). Sources is empty on any miss path.
        """
        restricted = restricted_matches or []

        if not chunks:
            # Empty results = a hard access miss (RBAC filtered everything). Always a denial.
            denial = verbalize_denial(role)
            log_request_event(
                request_id=request_id,
                role=role,
                event="generation_rbac_refusal",
                chunk_count=0,
                accessible_collections=denial.accessible_collections,
            )
            return denial.message, []

        # Relevance floor — deterministic backstop UNDER the LLM sentinel (see
        # _keep_above_floor). Drops keyword-collision noise before it reaches the prompt.
        kept = _keep_above_floor(chunks)
        if not kept:
            # Everything was noise. Same fork as the sentinel path: access miss → RBAC
            # denial (answer lives behind RBAC), else genuinely absent → no-content.
            if restricted:
                denial = verbalize_denial(role)
                log_request_event(
                    request_id=request_id,
                    role=role,
                    event="generation_rbac_refusal",
                    chunk_count=len(chunks),
                    reason="below_relevance_floor",
                    accessible_collections=denial.accessible_collections,
                )
                return denial.message, []
            log_request_event(
                request_id=request_id,
                role=role,
                event="generation_no_relevant_content",
                chunk_count=len(chunks),
                reason="below_relevance_floor",
            )
            return _NO_RELEVANT_MESSAGE, []
        chunks = kept

        context = _build_context(chunks)
        prompt = _RAG_PROMPT.format(context=context, question=question)

        # temperature=0.0 (default) — drug doses and protocol numbers must be
        # quoted exactly. Any sampling variance risks paraphrasing critical values.
        with stopwatch() as gen_ms:
            answer = await self._llm.generate(prompt)

        # No-relevant-content gate. The LLM may emit the sentinel alone (nothing relevant)
        # or — imperfectly — alongside real protocol text. Strip the token and decide by
        # what REMAINS: if nothing substantive is left, it's a genuine no-content answer →
        # clean message with NO sources (citations must agree with the answer). Otherwise
        # keep the real content, dropping the stray sentinel so it never leaks to the user.
        cleaned = answer.replace(_NO_RELEVANT_SENTINEL, "").strip()
        if not cleaned:
            # Nothing relevant in the role's accessible collections. If the query's answer
            # lives in a collection the role CAN'T see, this is an access miss → RBAC denial
            # (e.g. nurse asks about billing). Otherwise it's genuinely absent → no-content.
            if restricted:
                denial = verbalize_denial(role)
                log_request_event(
                    request_id=request_id,
                    role=role,
                    event="generation_rbac_refusal",
                    chunk_count=len(chunks),
                    reason="access_scoped_miss",
                    accessible_collections=denial.accessible_collections,
                )
                return denial.message, []
            log_request_event(
                request_id=request_id,
                role=role,
                event="generation_no_relevant_content",
                chunk_count=len(chunks),
            )
            return _NO_RELEVANT_MESSAGE, []

        answer = _append_disclaimers(cleaned, chunks)

        sources = [
            Source(
                source_document=c.source_document,
                section_title=c.section_title,
                collection=c.collection,
            )
            for c in chunks
        ]

        # Runtime safety guard (fail-closed) — deterministic, no LLM. Withhold any answer
        # that converts units, prescribes, or drops the disclaimer, before it reaches the
        # user. Same check logic runs in the Track 11 eval (single source of truth).
        collections_used = {c.collection for c in chunks}
        violations = run_safety_checks(
            answer,
            is_clinical=_clinical_is_central(chunks),
            disclaimer_markers=_DISCLAIMER_MARKERS,
        )
        if violations:
            # Log violation TYPES only — never the answer text (PHI risk).
            log_request_event(
                request_id=request_id,
                role=role,
                event="generation_safety_block",
                chunk_count=len(chunks),
                violation_types=[v.type.value for v in violations],
            )
            return _SAFE_FALLBACK, sources

        # Log operational metadata only — never log question, context, or answer (PHI risk).
        log_request_event(
            request_id=request_id,
            role=role,
            event="generation",
            latency_ms=gen_ms(),
            chunk_count=len(chunks),
            source_documents=[c.source_document for c in chunks],
            collections_used=list(collections_used),
        )

        return answer, sources

# ---------------------------------------------------------------------------
# Module-level helpers — pure functions, no self dependency
# ---------------------------------------------------------------------------

def _build_context(chunks: list[ChunkResult]) -> str:
    """Concatenate chunks into the context block for the RAG prompt.

    Each chunk is prefixed with its source citation so the LLM can reference
    it in the answer. Format matches PROMPTS.md §2 spec.
    """
    parts = [
        f"[Source: {c.source_document} — {c.section_title}]\n{c.text}"
        for c in chunks
    ]
    return "\n\n".join(parts)


def _append_disclaimers(answer: str, chunks: list[ChunkResult]) -> str:
    """Append mandatory disclaimers, keyed on whether clinical content is CENTRAL.

    Deterministic — the LLM does not decide whether a disclaimer is needed.
    Disclaimer logic from GUARDRAILS.md §3-4:
    - clinical/nursing is the answer's dominant grounding → clinical disclaimer
    - drug_formulary source → additional prescribing disclaimer

    The clinical disclaimer keys on whether clinical/nursing content is CENTRAL — not
    merely present. Keying on "any clinical chunk retrieved" wrongly stamped the clinical
    disclaimer onto equipment/HR answers whenever a single stray nursing chunk (e.g. hand
    hygiene) survived into the context. We treat clinical as central when it is the dominant
    collection by count OR grounds the top-scored chunk; a lone low-ranked stray does not
    trigger it. This tightens a false-positive without under-protecting genuine clinical
    answers (both the count-majority and top-chunk cases still fire), and the runtime safety
    guard remains the fail-closed backstop.
    """
    sources_used = {c.source_document.lower() for c in chunks}

    if _clinical_is_central(chunks):
        answer += _CLINICAL_DISCLAIMER

    if any(kw in src for kw in _DRUG_SOURCE_KEYWORDS for src in sources_used):
        answer += _DRUG_DISCLAIMER

    return answer


def _keep_above_floor(chunks: list[ChunkResult]) -> list[ChunkResult]:
    """Drop chunks whose reranker score is below the relevance floor.

    Deterministic backstop UNDER the LLM no-relevant sentinel. Cross-encoder logits below
    settings.rerank_relevance_floor are keyword-collision noise (e.g. "why no response"
    hitting a billing "24 hrs no response" SLA row). Removing them before the prompt stops
    generation from stitching a confident answer out of junk. Admin — which has no RBAC
    narrowing, so its candidate pool spans all five collections — benefits most.
    """
    return [c for c in chunks if c.score >= settings.rerank_relevance_floor]


def _clinical_is_central(chunks: list[ChunkResult]) -> bool:
    """True when clinical/nursing content dominates the answer, not just appears in it.

    Central if EITHER the top-scored chunk is clinical/nursing (primary grounding) OR
    clinical/nursing chunks are a plurality of the context. `chunks` arrives sorted by
    reranker score descending (rerank_service), so chunks[0] is the top-scored chunk.
    """
    if not chunks:
        return False
    clinical_count = sum(1 for c in chunks if c.collection in _CLINICAL_COLLECTIONS)
    top_is_clinical = chunks[0].collection in _CLINICAL_COLLECTIONS
    return top_is_clinical or clinical_count * 2 > len(chunks)
