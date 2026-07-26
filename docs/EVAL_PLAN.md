# MediBot — Eval Plan
> Every eval track: what it measures, the metric, the tool, the gate, and where it lives.
> Logging tells *what happened*; eval tells  *whether it was correct*. The pipeline
> already logs router decisions, retrieval/reranker scores, and SQL; the harness scores
> those same outputs against ground truth.

**Rule of thumb:** deterministic checks (RBAC, retrieval, SQL, routing, chunk quality,
numbers, safety, disclaimer keying + relevance floor) use plain **pytest + numpy** — no
framework where an assertion suffices. The intent gate is deterministic on first-turn cases
and LLM-adjudicated only on the ambiguous multi-turn middle.

Only judgment-based checks (faithfulness, correctness) use **DeepEval**, and the judge model
(`openai/gpt-oss-20b`) is **different from the generator** (`llama-3.3-70b`) to avoid
self-preference bias. Thresholds live in `backend/evals/eval_config.py` — never hardcoded in tests.

---

## Track summary

| # | Track | What it measures | Metric | Tool | Gate (CI) | Status |
|---|---|---|---|---|---|---|
| 1 | **RBAC red-team** | Can a restricted chunk leak to an unauthorized role? | Leak rate = leaked ÷ adversarial | pytest | **leak = 0** — fail-closed, runs **first** | ✅ built |
| 2 | **Retrieval quality** | Did the right chunks come back? Hybrid/rerank lift over dense | P@3, MRR (bootstrap 95% CI) | numpy | hybrid+rerank **P@3 ≥ 0.70** (dense = ungated baseline) | ✅ built |
| 3 | **Faithfulness** | Are answer claims grounded in the retrieved context? | DeepEval `FaithfulnessMetric` | DeepEval (judge) | **≥ 0.80** | ✅ built |
| 3b | **Number match** | Doses/codes/prices/thresholds quoted **exactly** (no round/convert) | Exact-string match (whitespace-normalized) | pytest | **≥ 0.90** | ✅ built |
| 4 | **SQL-RAG correctness** | Valid SQL → right data → faithful NL answer | Exec-success rate, result-match rate | pytest | **exec ≥ 0.90, match ≥ 0.85** | ✅ built |
| 5 | **Routing accuracy** | Analytical vs knowledge classification (+ specificity) | Accuracy, confusion matrix | numpy / sklearn | **≥ 0.90** | ✅ built |
| 6 | **Chunk quality** | Every chunk carries its parent heading; citations complete | Heading-present, table-intact rates | pytest | **heading ≥ 0.95, table ≥ 0.90** | ✅ built |
| 7 | **Medical safety** | Disclaimer present, no unit conversion, no prescriptive voice | Violation count (deterministic guard) | pytest | **zero violations** — fail-closed | ✅ built |
| 8 | **Judge calibration** | Does the LLM-judge agree with human labels? | Cohen's κ vs human labels | numpy | **none** — reported; investigate if κ < 0.70 | ✅ built |
| 9 | Multi-turn retrieval | Query-rewrite → P@3 lift over the raw follow-up | P@3 delta | numpy | rewrite **P@3 ≥ baseline** (+0.10 on ambiguous) | 🔲 planned |
| 10 | Rewrite faithfulness | Rewrite preserves user intent (no hallucinated topic) | DeepEval semantic similarity | DeepEval | **≥ 0.90** | 🔲 planned |
| 11 | HITL SQL safety | Token lifecycle: approve / reject / expire / tamper / replay | Pass/fail assertions | pytest | **all pass** — fail-closed | 🔲 planned |
| 12 | **Intent gate** | Admission control: is a turn resolved / ambiguous / empty *before* retrieval? | Accuracy, 3×3 confusion matrix | pytest (+ LLM for multiturn) | first-turn (deterministic) **= 1.0**; overall **≥ 0.85** | ✅ built |
| 13 | **Disclaimer keying + floor** | Clinical disclaimer keyed on *central* clinical content, not any stray chunk; relevance floor drops noise | Label match (deterministic) | pytest (no LLM) | **= 1.0** — fail-closed | ✅ built |

---

## Files per track

| # | Track | Test file | Golden set |
|---|---|---|---|
| 1 | RBAC red-team | `backend/evals/test_rbac.py` | `golden/adversarial.json` (see `RBAC_REDTEAM.md`) |
| 2 | Retrieval quality | `backend/evals/test_retrieval.py` | `golden/factual.json` |
| 3 | Faithfulness | `backend/evals/test_faithfulness.py` | `golden/factual.json` |
| 3b | Number match | `backend/evals/test_numbers.py` | `golden/numbers.json` |
| 4 | SQL-RAG correctness | `backend/evals/test_sql.py` | `golden/sql.json` |
| 5 | Routing accuracy | `backend/evals/test_routing.py` | `golden/routing.json` |
| 6 | Chunk quality | `backend/evals/test_chunk_quality.py` | (scans ingested Qdrant payloads) |
| 7 | Medical safety | `backend/evals/test_safety.py` | `golden/safety.json` |
| 8 | Judge calibration | `backend/evals/test_calibration.py` | `golden/calibration_labels.json` |
| 9–11 | Multi-turn / HITL | `test_multiturn.py`, `test_sql_hitl.py` *(planned)* | `golden/multiturn.json`, `golden/sql_hitl.json` *(planned)* |
| 12 | Intent gate | `backend/evals/test_intent.py` | `golden/intent.json` |
| 13 | Disclaimer keying + floor | `backend/evals/test_disclaimer.py` | `golden/disclaimer.json` |

Shared infra: `eval_config.py` (thresholds), `eval_schemas.py` (Pydantic golden shapes),
`conftest.py` (real-adapter fixtures), `groq_judge.py` (DeepEval judge over Groq).

---


## Golden answers — SQL (verified against the actual DB)

| Question | Expected |
|---|---|
| How many claims are escalated? | **8** |
| Which department has the most claims? | **cardiology (20)** |
| Which equipment category has most open/in-progress tickets? | **radiology & monitoring, tied at 8** |
| Total approved claim amount | **₹24,42,700** |
| Claim rejection rate | **≈14%** |
| Maintenance tickets resolved | **42** |

---
