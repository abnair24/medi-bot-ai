"""Intent gate — admission control in front of the router.

The two-axis router answers WHERE the answer lives and HOW the query is shaped, but it
has no "none of the above" verdict — every question is forced into a retrieval bucket and
fails open to the widest search. That is why contentless follow-ups ("EXPLAIN", "ONE MORE",
"WHEN") and out-of-context nudges get confident, wrong answers.

This service adds the missing question — WHETHER we have earned the right to retrieve at
all — and returns a typed IntentResolution with three outcomes:

  * resolved  → we have a self-contained query  → proceed to the router.
  * ambiguous → a back-reference the history can't pin to ONE topic → ask the user (HITL).
  * empty     → no content and no recoverable prior intent → refuse, never retrieve.

Cost shape: turn 1 is decided deterministically (no LLM). The LLM is spent only when there
IS history to disambiguate against — exactly where judgement adds value.
"""

import json
import re

from src.adapters.llm_adapter import LLMAdapter
from src.models.schemas import IntentResolution
from src.utils.logger import log_request_event

# Meta / imperative / filler tokens that carry no domain subject of their own. A message
# made up ONLY of these (plus stopwords) is a continuation — it leans entirely on history.
_CONTINUATION_TOKENS: frozenset[str] = frozenset({
    "explain", "elaborate", "expand", "clarify", "detail", "details", "describe",
    "why", "how", "when", "where", "what", "which", "who",
    "more", "again", "continue", "go", "next", "one", "another", "same",
    "tell", "show", "give", "list", "ok", "okay", "yes", "no", "sure", "please",
    "this", "that", "these", "those", "it", "them", "there",
})

# Common English stopwords — combined with continuation tokens to judge "contentless".
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "to", "for", "in", "on", "at",
    "is", "are", "was", "were", "be", "been", "do", "does", "did", "me", "us",
    "my", "our", "your", "about", "with", "as", "so", "then", "up", "out",
})

_INTENT_PROMPT = """\
You are the intent gate for a hospital knowledge assistant. Using the conversation history,
classify the user's LATEST message into exactly one of three states and return JSON only.

States:
- "resolved": the message is self-contained OR it is a follow-up that clearly points to ONE
  topic in the history. Set "resolved_query" to a standalone, retrieval-ready question. If
  the message already names its own subject, copy it verbatim. If it is a follow-up
  ("explain", "why", "any contraindications?", "one more"), rewrite it by substituting the
  single topic it refers to (e.g. history about Metformin + "max dose?" -> "maximum daily
  dose of Metformin").
- "ambiguous": the message is a follow-up with NO subject of its own, and the history
  contains TWO OR MORE distinct topics it could plausibly point to. Do NOT guess. Set
  "candidates" to the 2-4 competing topics as short labels (e.g. "ceftriaxone dose",
  "metronidazole dose").
- "empty": the message has no usable content AND the history offers no topic to inherit
  (e.g. a bare "explain" with unrelated or no prior context, or a meta-question about the
  assistant itself). Nothing to retrieve.

Hard rules:
- NEVER merge two different clinical scenarios into one query.
- NEVER carry a previous symptom/diagnosis into a message that introduces a NEW one — that
  new message is "resolved" with its own subject.
- When a follow-up genuinely fits ONE topic, prefer "resolved" over "ambiguous".
- Return ONLY this JSON, no prose:
{{"state": "resolved"|"ambiguous"|"empty", "resolved_query": string|null,
  "candidates": [string], "reason": "one short clause"}}

Conversation history:
{history}

Latest message: {question}

JSON:\
"""


class IntentService:
    """Resolves each turn to an IntentResolution verdict before retrieval runs."""

    def __init__(self, llm: LLMAdapter) -> None:
        self._llm = llm

    async def resolve(
        self,
        question: str,
        history: list[dict],
        request_id: str,
        role: str,
    ) -> IntentResolution:
        """Classify the turn into resolved / ambiguous / empty.

        Args:
            question: user's latest message. Never logged (PHI risk).
            history: prior turns [{"role", "content"}], already windowed by the caller.
            request_id: trace ID for log correlation.
            role: authenticated user role for log correlation.

        Returns:
            IntentResolution. On any LLM/parse failure the verdict degrades to
            resolved(question) — fail toward answering the literal query rather than
            hard-refusing, so a classifier hiccup never blocks a valid question.
        """
        contentless = _is_contentless(question)

        # --- Turn 1: decide deterministically, no LLM call. ---
        if not history:
            if contentless:
                return self._log_and_return(
                    request_id, role,
                    IntentResolution(state="empty", reason="contentless_first_turn"),
                )
            return self._log_and_return(
                request_id, role,
                IntentResolution(state="resolved", resolved_query=question,
                                 reason="self_contained_first_turn"),
            )

        # --- History present: the LLM adjudicates resolve vs ambiguous vs empty. ---
        formatted = _format_history(history)
        raw = await self._llm.generate(
            _INTENT_PROMPT.format(history=formatted, question=question),
            temperature=0.0,
        )
        verdict = _parse_intent(raw, fallback_query=question)
        return self._log_and_return(request_id, role, verdict)

    def _log_and_return(
        self, request_id: str, role: str, verdict: IntentResolution
    ) -> IntentResolution:
        # Log the DECISION only — never the question, resolved_query, or candidate text.
        log_request_event(
            request_id=request_id,
            role=role,
            event="intent_resolved",
            intent_state=verdict.state,
            candidate_count=len(verdict.candidates),
            reason=verdict.reason,
        )
        return verdict


def _is_contentless(question: str) -> bool:
    """True if the message is short and made up ONLY of meta/imperative/stop tokens.

    "explain" / "one more" / "go on" -> True (needs history or must be refused).
    "why no response" -> False ("response" is a content token) — handled downstream by
    the relevance floor, not here (we can't know it's off-domain without retrieving).
    """
    tokens = re.findall(r"[a-z0-9]+", question.lower())
    if not tokens:
        return True
    meta = _CONTINUATION_TOKENS | _STOPWORDS
    return len(tokens) <= 4 and all(t in meta for t in tokens)


def _parse_intent(raw: str, fallback_query: str) -> IntentResolution:
    """Parse the intent-gate JSON; fail toward resolved(fallback_query) on any error."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return IntentResolution(state="resolved", resolved_query=fallback_query,
                                reason="parse_fallback_no_json")
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return IntentResolution(state="resolved", resolved_query=fallback_query,
                                reason="parse_fallback_bad_json")

    state = data.get("state")
    if state == "ambiguous":
        candidates = [str(c) for c in data.get("candidates", []) if str(c).strip()]
        # An "ambiguous" verdict with no candidates can't be disambiguated — treat as empty.
        if not candidates:
            return IntentResolution(state="empty", reason="ambiguous_without_candidates")
        return IntentResolution(state="ambiguous", candidates=candidates,
                                reason=str(data.get("reason", "")))
    if state == "empty":
        return IntentResolution(state="empty", reason=str(data.get("reason", "")))

    # Default / "resolved": use the model's query, else fall back to the raw question.
    resolved = data.get("resolved_query") or fallback_query
    return IntentResolution(state="resolved", resolved_query=str(resolved),
                            reason=str(data.get("reason", "")))


def _format_history(history: list[dict]) -> str:
    """Flatten conversation state into labelled User:/Assistant: lines for the prompt."""
    return "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}" for turn in history
    )
