# MediBot — Data Handling & Privacy

> Rules for handling Protected Health Information (PHI), logging, storage, and data flow.
> These apply to every line of code in the repo.

---

## 1. What counts as PHI in this system

MediBot is an internal tool for hospital staff — the "users" are employees, not patients.
But user queries can contain patient information:

- A doctor might ask: "What's the protocol for patient Ramesh, 62M, diabetic with eGFR 38?"
- A nurse might ask: "IV cannula size for baby Priya, 3.8 kg?"
- A billing exec might ask: "Claim CLM-2024-1042 for Kavya Pillai — what's the status?"

The query itself becomes PHI the moment it contains a patient name, ID, diagnosis, or
clinical detail. The DB already contains patient names and claim IDs.

---

## 2. Logging rules (critical)

### What you MUST log (for eval and debugging)
- Request ID (UUID, generated per request)
- User role (from JWT — this is employee info, not PHI)
- Route decision (sql_rag / hybrid_rag)
- Retrieved chunk IDs and scores (these are document metadata, not PHI)
- Reranker scores
- Retrieval type and latency
- Whether RBAC blocked the query (boolean)

### What you must NEVER log
- The user's raw query text (may contain patient details)
- The LLM's full response text (may echo patient details from the query)
- Patient names, IDs, or diagnosis codes from queries
- The full prompt sent to the LLM (contains the query)
- SQL query results that contain patient_name or patient_id

### How to handle eval logging (the tension)

The eval harness needs to compare answers to golden cases, which requires seeing the
response. Resolution:

- **Eval runs only on golden-set queries** (synthetic, no real patient data).
- Eval logs go to `backend/evals/results/` (gitignored, local only, never committed).
- Production logging (the FastAPI app) follows the rules above — no query text.
- Tracing tools (Phoenix/LangSmith) are dev-only and must not run in production with
  real user traffic.

```python
# WRONG — logs PHI
logger.info(f"Query: {question}, Response: {answer}")

# RIGHT — logs metadata only
logger.info(f"request_id={req_id} role={role} route={route} "
            f"chunks_retrieved={len(chunks)} latency_ms={latency}")
```

---

## 3. Storage rules

| Data | Where | PHI risk | Rule |
|---|---|---|---|
| Source PDFs | `backend/ingestion/data/docs/` (git) | No — internal policy docs, not patient records | Safe to commit |
| mediassist.db | `backend/ingestion/data/` (git) | **Yes** — contains patient_name, patient_id | This is demo/synthetic data; if it were real, it would not be in git |
| Qdrant vectors | Docker volume | No — document chunks only, no patient data | Safe |
| JWT tokens | In-memory only | No — employee role, not PHI | Never persist |
| User queries | **Not stored** | Yes | Queries are processed and discarded; never written to disk or DB |
| LLM responses | **Not stored** | Yes | Responses are returned to the user and discarded |
| Eval results | `backend/evals/results/` (gitignored) | No — synthetic golden-set only | Never commit; never run on real queries |

---

## 4. API transmission rules

- User queries are sent to the Groq API for LLM inference. This is a data-transmission
  risk if queries contain PHI.
- For a production deployment: use the LLM provider's data-processing agreement (DPA),
  or self-host via vLLM (the adapter pattern makes this a one-file swap).
- For this portfolio project: queries are synthetic/demo. Note the risk in the README
  and state that a production deployment would require a DPA or self-hosted inference.

---

## 5. SQL RAG — special handling

The `claims` table contains `patient_name` and `patient_id`. The SQL RAG chain must:

- Never include `patient_name` or `patient_id` in the LLM's narration prompt unless
  the query specifically asks for it AND the user has the appropriate role.
- Prefer aggregate queries (counts, sums, averages) over queries that return individual
  patient rows.
- If a query returns individual patient records, the narration should summarize
  ("3 claims in cardiology are escalated") rather than listing names.

```python
# In sql_rag_service.py — strip PHI columns from result before narration
PHI_COLUMNS = {"patient_name", "patient_id"}

def sanitize_result(result, columns):
    safe_cols = [c for c in columns if c not in PHI_COLUMNS]
    # return only safe columns to the narration LLM
```

---

## 6. Demo data acknowledgement

The mediassist.db contains synthetic data (fictional patient names and IDs). State this
in the README:

> **Data notice:** All patient names, IDs, and claim details in this project are
> synthetic and do not represent real individuals. In a production deployment, this
> database would be behind additional access controls and would not be committed to
> version control.

---

## 7. For Claude Code — persistent rules

The rules above are also encoded in `/.claude/rules/` so they apply to every file
Claude Code generates. See the rules files for the enforced versions.
