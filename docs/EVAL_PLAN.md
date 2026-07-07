# MediBot — Eval Plan

> Every eval track: what it measures, how, the tool, the threshold, and what a failure
> means. This is the document that turns the build into a portfolio piece.

---

## Principle

Logging tells you what happened. Eval tells you whether it was correct. The pipeline
already logs router decisions, retrieval scores, reranker scores, and SQL queries. The
eval harness takes those same outputs and scores them against ground truth.

Deterministic evals (retrieval, RBAC, SQL, routing, chunk quality) use plain pytest +
numpy. Judge-based evals (faithfulness, answer correctness) use DeepEval. Do not use a
framework where an assertion suffices.

---

## 1. Retrieval quality (dense vs hybrid vs +rerank)

**What:** did the correct chunks come back, and did hybrid/rerank improve over dense-only?

**Metric:** Precision@3, MRR (Mean Reciprocal Rank).

**How:**
- Golden set: each factual question tagged with `expected_chunk_ids` (the chunk IDs that
  contain the answer).
- Run retrieval three ways: dense-only, hybrid (dense+BM25), hybrid+rerank.
- For each, check if expected chunks appear in top-3.
- Compute P@3 and MRR per method. Report delta with bootstrap 95% CI.

**Tool:** plain Python (numpy). No framework needed — it's set math.

**Threshold:** hybrid+rerank P@3 ≥ 0.70. Dense-only P@3 is the baseline (no gate, just
reported for comparison).

**Test file:** `evals/test_retrieval.py`

**Key test queries** (where keyword search matters):
- `I21.4` (ICD code — pure keyword)
- `E-12` (fault code)
- `24G` (cannula size)
- `EXCL-03` (exclusion code)
- `Piperacillin-Tazobactam` (drug name)

---

## 2. RBAC red-team (leak rate)

**What:** can a restricted chunk leak to an unauthorized role?

**Metric:** leak rate = (leaked chunks / total adversarial queries). Must be zero.

**How:**
- Run every case in `evals/golden/adversarial.json` (see RBAC_REDTEAM.md for the full
  matrix).
- For each: execute the retrieval with the attacker role's filter.
- Assert: no chunk from a restricted collection appears in the retrieved context.
- Check the **retrieved context**, not the final answer — the LLM might refuse politely
  even if a restricted chunk leaked into the prompt.

**Tool:** plain pytest assertions.

**Threshold:** leak count = 0. Any leak → build fails (fail-closed).

**Test file:** `evals/test_rbac.py`

**Critical case:** NSTEMI / I21.4 appears in clinical (treatment protocol), billing
(package rate ₹1,20,000), and the claim guide (worked example). A billing_executive
querying "NSTEMI" must get the billing data and never the clinical treatment. This tests
RBAC under semantic overlap — the hard case.

---

## 3. Numeric faithfulness

**What:** when the answer contains a dose, code, price, or threshold, is it correct and
sourced from the retrieved context?

**Metric:** DeepEval FaithfulnessMetric score.

**How:**
- Golden set: questions whose answers contain specific numbers (drug doses, ICD codes,
  equipment fault codes, package rates, lab values).
- Run the full pipeline (retrieve + generate).
- DeepEval checks: does the answer's factual content follow from the retrieved chunks?
- Additionally: a custom assertion checks the specific number matches the golden answer
  (e.g., "Metformin first-line dose" must contain "500 mg BD").

**Tool:** DeepEval (faithfulness metric) + custom pytest assertion for the number.

**Threshold:** faithfulness ≥ 0.80. Number-match accuracy ≥ 0.90.

**Test file:** `evals/test_faithfulness.py`

**Key cases:**
- Metformin first-line = 500 mg BD (not 1000, not TDS)
- Paracetamol max = 4 g/day
- Paediatric cannula <5kg = 24G
- NSTEMI package = ₹1,20,000
- Emergency pre-auth deadline = 6 hours
- Braden scale threshold = ≤18
- Critical haemoglobin = <7 g/dL

---

## 4. SQL-RAG correctness

**What:** did the generated SQL execute, return the right data, and was the NL answer
faithful to the result?

**Metric:** execution success rate, result-match rate, NL faithfulness.

**How:**
- Golden set: each SQL question has `expected_sql_result` (the correct data).
- Run SQL RAG. Check: (1) SQL executes without error, (2) result matches expected,
  (3) NL answer is consistent with the result.

**Tool:** plain pytest for SQL execution + result match. DeepEval for NL faithfulness.

**Threshold:** execution success ≥ 0.90. Result match ≥ 0.85.

**Test file:** `evals/test_sql.py`

**Golden answers (verified from the actual DB):**

| Question | Expected answer |
|---|---|
| How many claims are escalated? | 8 |
| Which department has the most claims? | cardiology (20) |
| Which equipment category has most open/in-progress tickets? | radiology and monitoring, tied at 8 |
| What is the total approved claim amount? | ₹24,42,700 |
| What is the claim rejection rate? | 14.1% |
| How many claims does Star Health have? | (verify at build time) |
| How many maintenance tickets are resolved? | 42 |

**Trap case:** "ticket TKT-2024-2001 has fault F-05 — what does that code mean?" requires
DB lookup + equipment-manual lookup. The DB's issue_type and the manual's fault-code
description may not match perfectly. Tests cross-source handling.

---

## 5. Routing accuracy

**What:** does the semantic router correctly classify analytical vs knowledge queries?

**Metric:** accuracy, confusion matrix.

**How:**
- Golden set: ~40 questions labeled ANALYTICAL or KNOWLEDGE.
- Run each through the router. Compare predicted vs actual.
- Report confusion matrix and per-class accuracy.

**Tool:** plain Python. `sklearn.metrics.confusion_matrix` if you want.

**Threshold:** routing accuracy ≥ 0.90.

**Test file:** `evals/test_routing.py`

**Edge cases to include:**
- "how many claims were escalated" → ANALYTICAL (clear)
- "first-line drug for diabetes" → KNOWLEDGE (clear)
- "what's our claim volume looking like" → ANALYTICAL (no keyword trigger)
- "tell me about billing codes for cardiac" → KNOWLEDGE (mentions numbers but is a doc Q)

---

## 6. Chunk quality

**What:** does each chunk carry its parent heading? Are tables intact?

**Metric:** heading-present rate, table-integrity rate.

**How:**
- After ingestion, sample chunks from table-heavy docs (drug_formulary, billing_codes,
  equipment_manual).
- Assert: chunk text includes the section heading (e.g., a dose table chunk contains the
  drug name, not just "500 mg BD").
- Assert: table rows are not split across chunks.

**Tool:** plain pytest. String assertions on chunk content.

**Threshold:** heading-present ≥ 0.95. Table-intact ≥ 0.90.

**Test file:** `evals/test_chunk_quality.py`

---

## 7. LLM-as-judge calibration (reported, not gated)

**What:** does the DeepEval judge agree with your human labels?

**How:**
- Label ~30 answers yourself (correct / partially correct / wrong).
- Run DeepEval on the same 30.
- Compute agreement rate (Cohen's kappa or simple %).
- Use a **different model** as judge than as generator (avoid self-preference bias).

**Tool:** plain Python comparison.

**Threshold:** none (this is a calibration report, not a gate). But if agreement < 0.70,
the judge is unreliable and you should adjust the rubric or model.

---

## CI gate configuration (.github/workflows/eval.yml)

```yaml
# Pseudocode — adapt to your runner
on: [push, pull_request]
jobs:
  eval:
    steps:
      - run: pytest evals/test_rbac.py -x          # fail-fast, zero tolerance
      - run: pytest evals/test_chunk_quality.py
      - run: pytest evals/test_sql.py
      - run: pytest evals/test_routing.py
      - run: pytest evals/test_retrieval.py
      - run: pytest evals/test_faithfulness.py      # needs LLM API key
```

RBAC runs first and fails fast (`-x`). If any RBAC leak exists, nothing else runs.
Faithfulness runs last because it costs LLM tokens.
