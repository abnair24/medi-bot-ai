"""Eval thresholds — the single source of truth for every eval test file.

Never hardcode a threshold inside a test. Import it from here so a change to what
"good enough" means happens in one place (code-quality.md: no hardcoded values).
Values sourced from EVAL_PLAN.md / phase8-plan.md.
"""

# --- Track 6: chunk quality ---
CHUNK_HEADING_THRESHOLD = 0.95
CHUNK_TABLE_THRESHOLD = 0.90
CITATION_COMPLETENESS = 1.0  # source_document + section_title non-empty — fail-closed

# --- Track 5: routing accuracy ---
ROUTING_ACCURACY_THRESHOLD = 0.90  # predicted route == expected route, over all cases

# --- Track 12: intent-gate accuracy (multiturn admission control) ---
# predicted intent_state == expected_state, over all cases. Below routing's bar because
# the ambiguous/resolved boundary is genuinely harder than route classification, and a
# miss degrades gracefully (over-clarify or refuse) rather than leaking or crashing.
INTENT_ACCURACY_THRESHOLD = 0.85
# First-turn (empty-history) cases are decided deterministically — they must be PERFECT.
INTENT_FIRST_TURN_THRESHOLD = 1.0

# --- Track 13: disclaimer keying (deterministic, no LLM) ---
# _clinical_is_central must match the golden label on every case — fail-closed on a
# safety-relevant guardrail, so the bar is 1.0.
DISCLAIMER_ACCURACY_THRESHOLD = 1.0

# --- Track 1: retrieval quality (dense vs hybrid vs +rerank) ---
RETRIEVAL_P3_THRESHOLD = 0.70  # hybrid+rerank P@3 gate. Dense-only is baseline (no gate).

# --- Track 3: faithfulness (DeepEval LLM-as-judge) ---
FAITHFULNESS_THRESHOLD = 0.80  # answer claims supported by retrieved context
NUMBER_MATCH_THRESHOLD = 0.90  # exact-string match of doses/values (deterministic, not judged)

# --- Track 4: SQL-RAG correctness ---
SQL_EXECUTION_THRESHOLD = 0.90    # fraction of cases where SQL executes without error
SQL_RESULT_MATCH_THRESHOLD = 0.85  # fraction of cases where NL answer contains expected values

# Judge model — MUST differ from the generator (llama-3.3-70b-versatile) to avoid
# self-preference bias. gpt-oss-20b is a plain LLM (not the agentic groq/compound),
# giving DeepEval a clean generate() interface. Its reliability is validated in Track 7.
JUDGE_MODEL = "openai/gpt-oss-20b"
