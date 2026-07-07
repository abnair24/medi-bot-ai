# MediBot — Instructions & Build Spec

> **What this is:** the single source of truth for building MediBot. It merges the original
> Codebasics assignment requirements with the eval-first extensions we designed. The
> assignment is the substrate; the eval layer is the project.
>
> **Who builds from this:** Claude Code (or you manually). Every other `.md` file in this
> repo is referenced from here and scoped to one concern to keep context windows small.

---

## 1. What MediBot is

An internal AI assistant for MediAssist Health Network (12 hospitals, 40+ clinics, India).
Two jobs:

1. **Intelligent retrieval** — staff ask natural-language questions, get cited answers from
   the right documents.
2. **Role-Based Access Control (RBAC)** — every query is scoped to what the user's role is
   allowed to see, enforced at the vector-store level, not the UI.

Plus our extensions:
3. **Measurable quality** — every pipeline stage is evaluated against ground truth and gated
   in CI.
4. **Deployable UI** — Streamlit app with a demo tab and an eval-dashboard tab.

---

## 2. User roles & access matrix

Five roles. Access enforced via Qdrant metadata filter on every query — restricted chunks
must never be returned, not filtered after the fact.

| Role | Collections accessible |
|---|---|
| `doctor` | clinical, nursing, general |
| `nurse` | nursing, general |
| `billing_executive` | billing, general |
| `technician` | equipment, general |
| `admin` | **all** |

SQL RAG is gated to `billing_executive` and `admin` only.

---

## 3. Data sources

### 3.1 Document collections (PDFs + markdown)

| Collection | Documents | Format |
|---|---|---|
| `general` | staff_handbook, leave_policy, code_of_conduct, general_faqs | PDF |
| `clinical` | treatment_protocols, drug_formulary, diagnostic_reference | PDF (table-heavy) |
| `nursing` | icu_nursing_procedures, infection_control | PDF |
| `billing` | billing_codes, claim_submission_guide | PDF + MD |
| `equipment` | equipment_manual | PDF (table-heavy) |

All source files are in `data/docs/`.

### 3.2 Relational database

`data/mediassist.db` — SQLite with two tables:

- `claims` (85 rows) — columns: claim_id, patient_id, patient_name, department, claim_type,
  diagnosis_code, insurer, claimed_amount, approved_amount, status, submitted_date,
  resolved_date. Statuses: pending, approved, rejected, submitted, escalated. Departments:
  nephrology, cardiology, neurology, gynaecology, orthopaedics, general_medicine, emergency.
- `maintenance_tickets` (78 rows) — columns: ticket_id, equipment_name, equipment_id,
  category, campus, issue_type, fault_code, raised_by, raised_date, resolved_date, status,
  resolution_note. Categories: sterilisation, infusion, radiology, monitoring, surgical,
  laboratory. Statuses: open, in_progress, resolved, escalated.

### 3.3 Chunk metadata schema

Every chunk stored in Qdrant must carry:

| Field | Type | Description |
|---|---|---|
| `source_document` | str | Original filename |
| `collection` | Literal enum | One of general/clinical/nursing/billing/equipment |
| `access_roles` | list[str] | Roles permitted to see this chunk |
| `section_title` | str | Parent heading for context |
| `chunk_type` | Literal enum | text / table / heading / code |

Validated via Pydantic at ingest time. See `ARCHITECTURE.md` §Pydantic boundaries.

---

## 4. Pipeline components

### Component 1: Document ingestion (Docling + HierarchicalChunker)

- Parse all PDFs/MD with **Docling** — structural awareness for headings, tables, code.
- Chunk with **HierarchicalChunker** — splits along document structure first (section →
  subsection → paragraph/table), then applies token limit as second pass.
- Each chunk's embedded text must carry its parent section heading as context.
- Each chunk must include the metadata schema above, validated via Pydantic before storage.
- All ingestion runs as a standalone script (`scripts/ingest.py`), not at app startup.

### Component 2: Hybrid RAG (dense + BM25)

- Dense embeddings: BAAI/bge-small-en-v1.5 (or similar).
- Sparse embeddings: FastEmbed (native Qdrant integration).
- Both stored at index time; queried together in a single Qdrant hybrid query.
- Not two separate queries merged in application code.
- RBAC metadata filter (`access_roles`) applied on the Qdrant query itself (pre-filter).

### Component 3: Reranking (cross-encoder)

- After hybrid retrieval returns top-10 candidates, a cross-encoder scores each
  query+chunk pair jointly.
- Only top-3 reranked chunks are passed to the LLM.
- Log reranker scores for every query (feeds the eval harness).

### Component 4: SQL RAG

- Triggered when the semantic router classifies a query as analytical.
- Gated to `billing_executive` and `admin` roles.
- Three explicit steps: (1) LLM generates SQL, (2) clean/extract the SQL statement,
  (3) execute against SQLite, (4) LLM produces natural-language answer from the result.
- Implemented as a plain Python function, not a LangChain agent.

### Component 5: Semantic router

- LLM-based classification: ANALYTICAL → SQL RAG, KNOWLEDGE → hybrid RAG.
- Uses Pydantic structured output for the route decision.
- Logged for every query; evaluated via confusion matrix (see EVAL_PLAN.md).

### Component 6: Backend (FastAPI)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/login` | Username + password → role-tagged JWT |
| POST | `/chat` | Question + role (from JWT) → answer + sources + retrieval_type |
| GET | `/collections/{role}` | Collections accessible to this role |
| GET | `/health` | Health check |

`/chat` response schema:
```json
{
  "answer": "...",
  "sources": [{"source_document": "...", "section_title": "...", "collection": "..."}],
  "retrieval_type": "hybrid_rag | sql_rag",
  "role": "..."
}
```

### Component 7: Frontend (Streamlit)

Two tabs:

**Tab 1 — Demo:**
- Login with 5 demo accounts (one per role).
- Chat interface with answer + source citations + retrieval-type badge.
- Role badge and accessible-collections sidebar.
- Clear RBAC refusal message (not a generic error): "As a nurse, you don't have access to
  billing documents."

**Tab 2 — Eval Dashboard:**
- Retrieval A/B chart (dense vs hybrid vs +rerank, P@3/MRR).
- RBAC leak-rate matrix (role × collection × attack type).
- Faithfulness scores.
- SQL correctness table.
- Routing confusion matrix.

---

## 5. Eval layer (our extension — the actual portfolio piece)

Full details in `EVAL_PLAN.md`. Summary of eval tracks:

| Track | What it measures | Tool | CI-gated? |
|---|---|---|---|
| Retrieval A/B | P@3, MRR — dense vs hybrid vs +rerank | plain Python (numpy) | Yes — threshold |
| RBAC red-team | leak rate across role × collection × attack | plain pytest | Yes — **leak=0 or build fails** |
| Numeric faithfulness | dose/code correct and from context | DeepEval | Yes — threshold |
| SQL correctness | valid SQL + correct result + faithful NL | plain pytest | Yes — pass/fail |
| Routing accuracy | confusion matrix (analytical vs knowledge) | plain Python | Yes — threshold |
| Chunk quality | parent heading present, tables intact | plain pytest | Yes — pass/fail |
| LLM-as-judge calibration | judge agreement with human labels | plain Python | No — reported |

Golden dataset: `evals/golden/` — see `GOLDEN_SET.md` for schema and starter cases.

---

## 6. Build sequence

Build in this order. Each phase is independently runnable and testable.

| Phase | What | Depends on | Deliverable |
|---|---|---|---|
| **0** | Project scaffold, deps, env | nothing | repo structure, `.env.example`, docker-compose |
| **1** | Ingestion | Phase 0 | `scripts/ingest.py` runs, chunks in Qdrant with correct metadata |
| **2** | Retrieval + RBAC | Phase 1 | hybrid search returns results, RBAC filter blocks restricted chunks |
| **3** | Reranker | Phase 2 | top-10 → top-3 with logged scores |
| **4** | LLM generation | Phase 3 | answers with citations from reranked chunks |
| **5** | SQL RAG + router | Phase 4 | analytical queries route to SQL, knowledge to hybrid |
| **6** | FastAPI backend | Phases 4+5 | all endpoints working with JWT auth |
| **7** | Streamlit frontend | Phase 6 | Tab 1 (demo) functional |
| **8** | Eval harness | Phases 1-6 | golden set, all eval tracks, CI green |
| **9** | Eval dashboard | Phase 8 | Tab 2 (dashboard) showing real scores |
| **10** | Deploy | Phase 9 | docker-compose up, or Streamlit Cloud + hosted Qdrant |

---

## 7. Tech stack

| Component | Tool | Why |
|---|---|---|
| PDF parsing | Docling | structural awareness for tables/headings |
| Chunking | HierarchicalChunker (Docling) | splits on document structure, not character count |
| Dense embedding | BAAI/bge-small-en-v1.5 | good quality, small, free |
| Sparse embedding | FastEmbed | native Qdrant hybrid integration |
| Vector DB | Qdrant (local) | hybrid search + metadata filter in one query |
| Reranker | cross-encoder (e.g. ms-marco-MiniLM) | lightweight, effective |
| LLM | Groq (llama-3 or similar) | fast, cheap inference |
| Backend | FastAPI | native Pydantic, async |
| Frontend | Streamlit | fast to build, native charts for eval dashboard |
| Auth | JWT (PyJWT) | simple, stateless |
| SQL DB | SQLite | provided with assignment |
| Eval framework | DeepEval (judge metrics) + pytest + numpy | pytest-native CI gating |
| Tracing | Arize Phoenix (optional) | open-source, complements DeepEval |
| Validation | Pydantic | every boundary — see ARCHITECTURE.md |
| CI | GitHub Actions | eval gate on every push |

---

## 8. Reference files in this repo

| File | Purpose | When to read |
|---|---|---|
| `INSTRUCTIONS.md` | This file — master spec | First |
| `ARCHITECTURE.md` | Component diagram, RAG pipeline, adapters, DI, Pydantic boundaries | Before coding |
| `EVAL_PLAN.md` | All eval tracks with metrics, tools, thresholds | Before Phase 8 |
| `GOLDEN_SET.md` | Golden dataset schema + starter cases | Before Phase 8 |
| `RBAC_REDTEAM.md` | Full adversarial prompt matrix | Before Phase 8 |
| `DATA_DICTIONARY.md` | DB schema, document inventory, metadata | During Phase 1 |
| `PROMPTS.md` | All LLM prompts (router, generator, SQL, judge) | During Phases 4-5, 8 |
| `GUARDRAILS.md` | Medical AI safety boundaries — what MediBot must never do | Before coding any generation logic |
| `DATA_HANDLING.md` | PHI rules, logging sanitization, storage rules | Before coding any logging or data flow |
| `CODING_STANDARDS.md` | Python engineering standards, config, DI, error handling, naming | Before coding — read once, follow always |
| `DEPLOY.md` | Docker, compose, cloud deploy instructions | Phase 10 |

---

## 9. Claude Code rules (`/.claude/rules/`)

Three rule files that Claude Code reads on every invocation. These encode the
non-negotiable constraints so they apply to every file without eating conversation
context:

| Rule file | What it enforces |
|---|---|
| `guardrails.md` | Medical safety boundaries, RBAC enforcement, response format |
| `data-handling.md` | PHI logging prohibitions, SQL result sanitization, gitignore |
| `code-quality.md` | Pydantic everywhere, adapter pattern, eval rules, type hints |

These are not documentation — they are machine-readable instructions that Claude Code
follows while generating code. If a rule here conflicts with a user request, the rule
wins.

---

## 10. Key design decisions (non-negotiable)

1. **RBAC is pre-filter, not post-filter.** The metadata filter is part of the Qdrant query.
   Restricted chunks are never returned to the application.
2. **Pydantic at every boundary.** API in/out, chunk metadata, LLM structured output, eval
   golden-set records.
3. **Logging feeds eval, but never logs PHI.** Every router decision, retrieval result with
   scores, reranker scores, and SQL query is logged. User query text and LLM responses are
   never logged. See DATA_HANDLING.md.
4. **MediBot retrieves, never advises.** It quotes protocols with citations. It never
   prescribes, diagnoses, or interprets patient-specific data. See GUARDRAILS.md.
5. **Deterministic evals use pytest, not frameworks.** RBAC, SQL, routing, retrieval metrics
   are plain Python. DeepEval only for judge-based metrics (faithfulness, correctness).
6. **Fail-closed CI gate.** Any RBAC leak = build fails. Retrieval/faithfulness below
   threshold = build fails.
7. **Adapters + dependency injection.** Each external dep (Qdrant, LLM, embedder) is behind
   an interface. Enables isolated testing and model swapping.
