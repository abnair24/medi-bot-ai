import json
import re

from src.adapters.llm_adapter import LLMAdapter
from src.models.schemas import RouteDecision, TwoAxisRouteDecision
from src.utils.exceptions import RoutingError
from src.utils.logger import log_request_event
from src.utils.timing import stopwatch

_ROUTER_PROMPT = """\
You are a query classifier for a hospital internal assistant.

Given the user's question, classify it as one of:
- ANALYTICAL: the answer must be COMPUTED over records in a database of claims or
  maintenance tickets — counting, summing, averaging, comparing, or trending across rows.
- KNOWLEDGE: the answer is STATED in a document — protocols, procedures, policies, drug
  information, equipment manuals, billing codes and rates, or general knowledge. A single
  documented figure (a package rate, fee, standard/expected value, code meaning, or stated
  SLA) is KNOWLEDGE even though it is a number.

Decide by where the answer LIVES, not by the wording. Words like "rate", "cost", or
"how long" do not by themselves mean ANALYTICAL — choose ANALYTICAL only when the answer
requires aggregating over many records, and KNOWLEDGE when it is a value stated in a
document.

Respond with JSON only:
{{"route": "sql_rag" or "hybrid_rag", "reasoning": "one sentence why"}}

User question: {question}\
"""

_TWO_AXIS_ROUTER_PROMPT = """\
You are a query classifier for a hospital internal assistant.

Classify the user's question on TWO axes:

Axis 1 — WHERE the answer lives:
- "sql_rag":    answer must be COMPUTED over records (counts, sums, averages, comparisons
                across claims or maintenance ticket rows in a database).
- "hybrid_rag": answer is STATED in a document (protocols, drug doses, codes, policies,
                package rates, equipment manuals, or any single documented value).

Axis 2 — HOW the query is shaped:
- "point_lookup":  user wants ONE specific value (a drug dose, a code, a package rate,
                   an SLA, the meaning of one fault code). A single document sentence
                   fully answers it.
- "aggregate":     user wants a COUNT, SUM, AVERAGE, COMPARISON, or TREND across many
                   records or document entries. Requires scanning multiple rows or sections.
- "exploratory":   user wants a broad overview, explanation, or 'tell me about' — the
                   answer draws on multiple sections, protocols, or policies together.

Rules:
- sql_rag is ALWAYS paired with "aggregate" (SQL queries are by definition aggregations).
- hybrid_rag can be any specificity.
- Choose by what the answer REQUIRES, not by wording alone.

Respond with JSON only:
{{
  "route": "sql_rag" or "hybrid_rag",
  "specificity": "point_lookup" or "aggregate" or "exploratory",
  "reasoning": "one sentence why"
}}

User question: {question}\
"""


class RouterService:
    """Classifies a user question as sql_rag (ANALYTICAL) or hybrid_rag (KNOWLEDGE).

    Two classification methods:
    - classify()          → RouteDecision (original 1-axis, backward-compatible)
    - classify_detailed() → TwoAxisRouteDecision (route × specificity)

    Both use the Groq LLM with a structured JSON prompt. Both fall back to
    hybrid_rag on parse failure rather than crashing the pipeline.
    """

    def __init__(self, llm: LLMAdapter) -> None:
        self._llm = llm

    async def classify(
        self,
        question: str,
        request_id: str,
        role: str,
    ) -> RouteDecision:
        """Classify the question and return a validated RouteDecision (1-axis).

        Args:
            question: user's natural-language question. Passed to LLM but never logged.
            request_id: trace ID for log correlation.
            role: authenticated user role for log correlation.

        Returns:
            RouteDecision with route ("sql_rag" or "hybrid_rag") and reasoning.

        Raises:
            RoutingError: if the LLM response cannot be parsed into a valid RouteDecision
                after defensive JSON extraction.
        """
        prompt = _ROUTER_PROMPT.format(question=question)
        response = await self._llm.generate(prompt, temperature=0.0)

        decision = _parse_route_decision(response)

        log_request_event(
            request_id=request_id,
            role=role,
            event="route_decision",
            route=decision.route,
            reasoning=decision.reasoning,
        )

        return decision

    async def classify_detailed(
        self,
        question: str,
        request_id: str,
        role: str,
    ) -> TwoAxisRouteDecision:
        """Classify on two axes: route type × query specificity.

        Returns a TwoAxisRouteDecision carrying both dimensions. Used by nodes
        that need specificity to tune retrieval depth or reranking aggressiveness.
        Falls back to hybrid_rag + exploratory on parse failure.

        Args:
            question: user's natural-language question. Never logged.
            request_id: trace ID for log correlation.
            role: authenticated user role for log correlation.

        Returns:
            TwoAxisRouteDecision with route, specificity, and reasoning.
        """
        prompt = _TWO_AXIS_ROUTER_PROMPT.format(question=question)
        with stopwatch() as elapsed_ms:
            response = await self._llm.generate(prompt, temperature=0.0)

        decision = _parse_two_axis_decision(response)

        log_request_event(
            request_id=request_id,
            role=role,
            event="route_decision_detailed",
            latency_ms=elapsed_ms(),
            route=decision.route,
            specificity=decision.specificity,
            reasoning=decision.reasoning,
        )

        return decision


def _extract_json(response: str) -> dict:
    """Extract the first JSON object from an LLM response string.

    The LLM is instructed to return JSON only, but may wrap it in markdown
    fences or prose. Tries clean parse first; falls back to regex extraction.

    Raises:
        RoutingError: if no valid JSON object is found.
    """
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*?\}", response, re.DOTALL)
    if not match:
        raise RoutingError(f"Router returned no JSON object. Raw response: {response!r}")

    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        raise RoutingError(
            f"Router JSON failed validation. Extracted: {match.group()!r}"
        ) from e


def _parse_route_decision(response: str) -> RouteDecision:
    """Parse and validate a 1-axis RouteDecision from the LLM response.

    Raises:
        RoutingError: if JSON extraction fails or the schema does not validate.
    """
    data = _extract_json(response)
    try:
        return RouteDecision(**data)
    except Exception as e:
        raise RoutingError(f"RouteDecision validation failed: {data!r}") from e


def _parse_two_axis_decision(response: str) -> TwoAxisRouteDecision:
    """Parse and validate a TwoAxisRouteDecision from the LLM response.

    Falls back to hybrid_rag + exploratory on any parse or validation error so
    the pipeline degrades gracefully — a failed specificity classification is
    better than a hard crash, and exploratory is the safest default (widest
    retrieval, no aggressive reranking cutoff).
    """
    try:
        data = _extract_json(response)
        return TwoAxisRouteDecision(**data)
    except Exception:
        return TwoAxisRouteDecision(
            route="hybrid_rag",
            specificity="exploratory",
            reasoning="parse_fallback",
        )
