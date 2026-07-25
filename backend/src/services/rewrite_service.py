from src.adapters.llm_adapter import LLMAdapter
from src.utils.logger import log_request_event

_REWRITE_PROMPT = """\
You are a query rewriter for a medical knowledge assistant. Your ONLY job is to make a
follow-up message self-contained by resolving references to earlier turns. You do NOT
merge topics and you do NOT add context the latest message does not actually depend on.

Decide in two steps:

1. Is the latest message ALREADY self-contained — does it name its own subject and make
   sense on its own, without the history? This INCLUDES any new clinical scenario,
   symptom, patient, drug, or code (e.g. "patient fractured leg", "patient has diarrhoea",
   "treatment for fever with 106F", "what is the dose of Metformin?"). If so, return it
   UNCHANGED, verbatim. Do NOT add anything from prior turns, even when the topic differs.

2. ONLY if the message cannot be understood on its own — it has a pronoun or ellipsis that
   points back ("what's the max dose?", "any contraindications?", "what about children?") —
   rewrite it by substituting the specific entity from the history the reference points to.

Hard rules:
- NEVER combine two different clinical scenarios into one question.
- NEVER carry a previous symptom or diagnosis into a message that introduces a new one.
- When in doubt, return the message UNCHANGED.
- Return ONLY the resulting question, nothing else.

Examples (history topic → latest message → output):
- fever/dengue → "patient fractured leg" → patient fractured leg
- fever/dengue → "patient has diarrhoea" → patient has diarrhoea
- "first-line drug for Type 2 Diabetes?" then "Metformin ..." → "what's the max dose?"
  → what is the maximum daily dose of Metformin?

Conversation history:
{history}

Latest message: {question}

Rewritten question:\
"""


class RewriteService:
    """Rewrites ambiguous follow-up questions into self-contained retrieval queries.

    Runs before the router node. On the first turn (empty history), skips the LLM
    call and returns the question unchanged — no ambiguity to resolve.
    """

    def __init__(self, llm: LLMAdapter) -> None:
        self._llm = llm

    async def rewrite(
        self,
        question: str,
        history: list[dict],
        request_id: str,
        role: str,
    ) -> str:
        """Return a self-contained rewrite of the question, or the original if unambiguous.

        Args:
            question: user's latest message. Never logged.
            history: prior conversation turns from LangGraph state.
                     Each dict must have "role" and "content" keys.
            request_id: trace ID for log correlation.
            role: authenticated user role for log correlation.

        Returns:
            Rewritten question string. Semantically equivalent to the input —
            no new medical content is generated.
        """
        if not history:
            log_request_event(
                request_id=request_id,
                role=role,
                event="rewrite_skipped",
                reason="first_turn",
            )
            return question

        formatted = _format_history(history)
        rewritten = await self._llm.generate(
            _REWRITE_PROMPT.format(history=formatted, question=question),
            temperature=0.0,
        )
        rewritten = rewritten.strip()

        # Log the skip/rewrite event — never the question or rewritten text (PHI risk).
        log_request_event(
            request_id=request_id,
            role=role,
            event="rewrite_applied",
            history_turns=len(history),
        )

        return rewritten


def _format_history(history: list[dict]) -> str:
    """Flatten LangGraph conversation state into a prompt-friendly string.

    Each dict is expected to have "role" ("user" or "assistant") and "content" keys.
    Produces alternating labelled turns:
        User: <message>
        Assistant: <message>
    """
    return "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}"
        for turn in history
    )
