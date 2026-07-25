"""HyDE — Hypothetical Document Embedding for dense retrieval quality improvement.

The vocabulary gap problem: a user query ("first-line drug for T2DM?") and its answer
document ("Metformin 500 mg BD is initiated...") live in different parts of the embedding
space. The question embedding is closest to other questions, not to answer documents.

HyDE (Gao et al. 2022) bridges this by asking the LLM to generate a SHORT hypothetical
answer, then embedding that synthetic answer instead of the raw query. The hypothetical
uses answer-space vocabulary, so it lands geometrically closer to real answer documents.

Design constraints:
  - HyDE improves DENSE retrieval only. BM25 keyword matching is already literal — the
    original query tokens are the right input. Never apply HyDE to the sparse path.
  - Hypothetical answers must NOT hallucinate specific numbers. The LLM is instructed
    to use placeholders ("[X mg]", "[N days]") where it lacks ground-truth. A hallucinated
    dose in the hypothetical could bias retrieval toward the wrong protocol chunk.
  - HyDE adds one LLM call per request (~200-400ms Groq latency). Gate behind
    settings.hyde_enabled (default False) so it only fires when explicitly opted in.
  - Failures in HyDE generation fall back to the original query — the retrieval path
    must never hard-fail because of a HyDE generation error.
"""

from src.adapters.llm_adapter import LLMAdapter
from src.utils.logger import log_request_event

_HYDE_PROMPT = """\
You are a document generator for a hospital internal assistant.

Given a user's question, write a SHORT hypothetical excerpt from the hospital's internal
protocol documentation that would answer the question.

Rules:
- Write in the style of a clinical protocol, policy document, or technical manual.
- If you do not know the exact number (dose, threshold, code), use a placeholder like
  "[X mg]", "[N hours]", "[CODE]" — never invent specific clinical values.
- Keep the excerpt to 2-3 sentences.
- Do NOT include a heading, source label, or disclaimer — just the excerpt text.

Question: {question}

Hypothetical protocol excerpt:\
"""


class HyDEService:
    """Expands a user query into a hypothetical answer document for dense retrieval.

    Injected into RetrievalService when settings.hyde_enabled is True.
    Failures fall back gracefully to the original query — never hard-fail retrieval.
    """

    def __init__(self, llm: LLMAdapter) -> None:
        self._llm = llm

    async def expand(self, query: str, request_id: str) -> str:
        """Generate a hypothetical answer document from the user's query.

        Args:
            query: the user's natural-language question. Never logged (PHI risk).
            request_id: trace ID for log correlation.

        Returns:
            The hypothetical document text. Falls back to the original query on any
            LLM error so retrieval can continue without the HyDE expansion.
        """
        try:
            hypothetical = await self._llm.generate(
                _HYDE_PROMPT.format(question=query),
                temperature=0.3,  # slight variance lets the LLM explore phrasings
            )
            log_request_event(
                request_id=request_id,
                event="hyde_expansion",
                expanded_length=len(hypothetical),
            )
            return hypothetical.strip()
        except Exception:
            # HyDE is a retrieval quality enhancement, not a correctness requirement.
            # Log the failure as a warning and continue with the original query.
            log_request_event(
                request_id=request_id,
                event="hyde_fallback",
            )
            return query
