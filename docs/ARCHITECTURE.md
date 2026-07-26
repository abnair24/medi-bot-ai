# MediBotAI — Architecture

> Component layout, LangGraph state machine, adapter pattern, Pydantic boundaries,
> and folder structure. Read this before writing code.

---

## 1. Query flow (text diagram)

```
User → Login (JWT) → /chat
                        │
                   extract role from JWT
                        │
               LANGGRAPH STATE GRAPH
               ┌──────────────────────────────────────────────────────┐
               │  State: {question, role, history, intent_state,      │
               │   clarify_candidates, route, chunks, answer,         │
               │   proposed_sql}                                      │
               ├──────────────────────────────────────────────────────┤
               │  RESOLVE_INTENT NODE  (admission control)            │
               │   ┌──────────┬───────────────┬───────────┐          │
               │ resolved   ambiguous        empty                   │
               │   │          │                │                     │
               │   │   interrupt() ←─ HITL   refuse → END            │
               │   │   clarify (which topic?) (no retrieval)         │
               │   ▼          │(resume)                              │
               │  ROUTER NODE (LLM)  ◄────────┘                      │
               │  ┌────┴────┐                                        │
               │ ANALYTICAL  KNOWLEDGE                               │
               │  │            │                                     │
               │  │   RETRIEVE → RERANK → GENERATE                   │
               │  │            (relevance floor · safety guard ·     │
               │  │             content-keyed disclaimers)           │
               │ SQL_PROPOSE                                         │
               │  │                                                  │
               │ interrupt() ←─ HITL pause (approve/reject)          │
               │  │                                                  │
               │ SQL_EXECUTE → NARRATE                               │
               └───────────┴──────────────────────────────────────────┘
                        │
               response: answer + sources + retrieval_type
               (state persisted via SqliteSaver for multi-turn)
```

**Intent gate (admission control):** the first node decides *whether* a turn is answerable
before any retrieval runs. `resolved` → proceed to the router; `ambiguous` (a follow-up that
matches ≥2 prior topics) → `interrupt()` for a clarify prompt, resuming into the router with
the chosen topic; `empty` (contentless, no recoverable prior intent) → deterministic refusal
with no retrieval. First-turn cases are decided deterministically (no LLM); the LLM is spent
only on the resolved-vs-ambiguous middle.

---

## 2. RAG pipeline — models, stores, and data flow

Two paths through the system: **ingestion** (offline, run once) and **query** (per request).
Different models serve different jobs — none of them is "the LLM."

### 2.1 Ingestion path (backend/ingestion/ingest.py → Qdrant)

```
PDF / MD files (backend/ingestion/data/docs/)
     │
     ▼
DOCLING  (document parser — not a model, a parser library)
     │   reads PDF structure: headings, tables, code, paragraphs
     │   outputs a document tree, not flat text
     │
     ▼
HIERARCHICAL CHUNKER  (Docling's chunker)
     │   splits on document structure first (section → subsection → table/para)
     │   then enforces a token-size cap as second pass
     │   attaches parent section heading to each chunk
     │
     ▼
PYDANTIC VALIDATION  (ChunkMetadata model)
     │   every chunk validated: source_document, collection, access_roles,
     │   section_title, chunk_type — reject if anything is missing
     │
     ├──────────────────────────────┐
     ▼                              ▼
DENSE EMBEDDING MODEL          SPARSE EMBEDDING MODEL
(BAAI/bge-small-en-v1.5)      (FastEmbed / Qdrant native)
     │                              │
     │  384-dim dense vector         │  sparse token-weight vector
     │                              │
     └──────────┬───────────────────┘
                ▼
          QDRANT  (vector store)
                │
                │  stores per chunk:
                │    - dense vector
                │    - sparse vector
                │    - payload (all metadata fields)
                │
                │  one collection: "medibot"
                │  metadata indexed for filtering: access_roles, collection
                ▼
          ready for queries
```

**What runs where:**
- Docling + chunker: CPU, local, pure Python.
- Dense embedding model: downloaded once (~130MB), runs on CPU (or GPU if available).
  This is **not** the LLM — it's a small encoder that converts text to a vector. No
  generation, no tokens, no API cost.
- Sparse embedding: FastEmbed, also local, also not the LLM. Produces BM25-style
  term-weight vectors.
- Qdrant: local Docker container or Qdrant Cloud.

No LLM is called during ingestion. No API cost. Run once.

### 2.2 Query path (/chat → response)

```
user question + role (from JWT)
     │
     ▼
INTENT GATE ─────────────────────────────────── LLM call #0 (only if history)
     │   admission control BEFORE retrieval. Turn-1 & contentless cases are
     │   decided deterministically (no LLM). Output: IntentResolution
     │   {state: resolved|ambiguous|empty, resolved_query, candidates}
     │     - empty      → deterministic refusal, END (no retrieval)
     │     - ambiguous  → interrupt() clarify → resume with chosen topic
     │     - resolved   → overwrite question with resolved_query, continue ▼
     │
     ▼
SEMANTIC ROUTER ─────────────────────────────── LLM call #1
     │   model: Groq (Llama-3 / similar)              (cheap, small prompt)
     │   input: question
     │   output: RouteDecision {"route": "sql_rag"|"hybrid_rag"}
     │
     ├── sql_rag ──────────────────────────────── see SQL RAG path below
     │
     ▼ hybrid_rag
     │
GENERATE QUERY EMBEDDINGS
     │   same two models as ingestion:
     │   dense: bge-small-en-v1.5  (local, no API)
     │   sparse: FastEmbed          (local, no API)
     │
     ▼
QDRANT HYBRID QUERY ─────────────────────────── no LLM, vector math
     │   one API call to Qdrant that does:
     │     1. apply metadata filter: access_roles contains user's role
     │     2. dense similarity search (cosine)
     │     3. sparse keyword search (BM25)
     │     4. fuse results (Reciprocal Rank Fusion)
     │   returns: top-10 candidate chunks with scores
     │
     │   note: RBAC enforcement happens HERE — restricted chunks are
     │   never returned, the LLM never sees them
     │
     ▼
CROSS-ENCODER RERANKER ──────────────────────── not the LLM
     │   model: ms-marco-MiniLM-L-6-v2 (or similar)
     │   a small classifier (~80MB), runs locally
     │   scores each (query, chunk) pair jointly
     │   input: 10 candidates
     │   output: top-3 by relevance score
     │   logged: all 10 scores before/after (feeds eval)
     │
     ▼
RELEVANCE FLOOR ─────────────────────────────── no LLM, deterministic
     │   drop reranked chunks below settings.rerank_relevance_floor —
     │   a backstop UNDER the LLM no-relevant sentinel that removes
     │   keyword-collision noise. If none survive → RBAC denial or
     │   "no relevant protocol" refusal (no fabricated answer).
     │
     ▼
LLM ANSWER GENERATION ──────────────────────── LLM call #2
     │   model: Groq API (Llama-3 / similar)           (the main LLM call)
     │   input: question + surviving chunks as context
     │   prompt: see PROMPTS.md §2
     │   output: grounded answer + source citations
     │   constraint: answer ONLY from context, never fabricate
     │   disclaimers: content-keyed — clinical disclaimer appended only
     │     when clinical/nursing is CENTRAL (top chunk or plurality),
     │     not merely present (deterministic, in-code)
     │
     ▼
RESPONSE
     │   answer + sources + retrieval_type="hybrid_rag" + role
```

### 2.3 SQL RAG path

```
question (classified as ANALYTICAL by router)
     │
     ▼
ROLE CHECK ──────────────────────────────────── no model, just logic
     │   only billing_executive and admin may use SQL RAG
     │   others get a refusal message
     │
     ▼
SQL GENERATION ──────────────────────────────── LLM call #2 (alt path)
     │   model: Groq API
     │   input: question + DB schema (from PROMPTS.md §4)
     │   output: raw SQL string
     │
     ▼
SQL CLEANUP + EXECUTION
     │   strip markdown fences, extract pure SQL
     │   execute against SQLite (mediassist.db)
     │   no model — deterministic
     │
     ▼
RESULT NARRATION ────────────────────────────── LLM call #3
     │   model: Groq API
     │   input: question + SQL result
     │   output: natural-language answer with numbers
     │
     ▼
RESPONSE
     │   answer + retrieval_type="sql_rag" + role
```

### 2.4 Model inventory — what is what

| Model | Type | Where it runs | API cost | Purpose |
|---|---|---|---|---|
| **bge-small-en-v1.5** | embedding encoder (384-dim) | local (CPU) | free | convert text to dense vectors |
| **FastEmbed sparse** | sparse encoder (BM25-style) | local (CPU) | free | convert text to keyword-weight vectors |
| **ms-marco-MiniLM** | cross-encoder classifier | local (CPU) | free | rerank query-chunk pairs by relevance |
| **Llama-3 via Groq** | generative LLM | Groq cloud API | per-token (cheap) | router, answer generation, SQL generation, narration |

Key distinction: the first three are **local models you download once**. They are not
generative LLMs — they don't produce text, they produce vectors or scores. The only
generative LLM is the Groq-hosted model, called 2–3 times per request (router + generation,
or router + SQL-gen + narration).

---

## 3. Folder structure

```
medibot/
├── CLAUDE.md
├── docker-compose.yml
├── pytest.ini                        # pythonpath = backend; testpaths = backend/evals
├── requirements.txt                 
├── docs/                             # ARCHITECTURE.md, EVAL_PLAN.md, PROMPTS.md, …
├── frontend/
│   └── app.py                        # Streamlit — Tab 1 (demo) + Tab 2 (eval)
└── backend/                          # everything importable lives here (PYTHONPATH=backend)
    ├── src/                          # FastAPI app — imported as `from src...`
    │   ├── graph/                    # LangGraph orchestration layer
    │   │   ├── state.py              # MediBotState TypedDict (+ intent_state, clarify_candidates)
    │   │   ├── nodes.py              # node functions (resolve_intent, router, retrieve, …)
    │   │   ├── edges.py              # conditional edges (route_after_intent, …)
    │   │   └── builder.py            # build_graph() — assemble + compile
    │   ├── api/
    │   │   ├── app.py                # FastAPI app, mount routers, compile graph
    │   │   ├── chat_router.py        # /chat, /chat/resume, /chat/clarify, /chats (list/load/delete), /collections, /health
    │   │   ├── eval_router.py        # /evals/run, /evals/status, /evals/results (admin)
    │   │   └── auth_router.py        # /login, JWT
    │   ├── services/                 # retrieval, rerank, generation, sql_rag, router,
    │   │   │                         #   intent, hyde, denial, safety_guard, sql_validator,
    │   │   │                         #   chat_session_store, eval
    │   │   └── …
    │   ├── adapters/                 # qdrant, embedding, sparse, reranker, llm, sql
    │   ├── models/                   # schemas.py, chunk.py, auth.py
    │   ├── security/jwt.py           # JWT encode/decode, role extraction
    │   ├── config/                   # settings.py (pydantic-settings), access.py
    │   └── utils/                    # logger.py, helpers.py, exceptions.py
    ├── ingestion/                    # imported as `from ingestion...`
    │   ├── ingest.py                 # standalone ingestion — run once
    │   ├── ensure_ingested.py        # idempotent ingest (compose `ingest` service entrypoint)
    │   ├── run_evals.py              # headless full-suite eval runner (manual / CI)
    │   └── data/
    │       ├── docs/                 # source PDFs + MD (git-tracked)
    │       ├── mediassist.db         # SQLite — app data (git-tracked, small)
    │       ├── checkpoints.db        # SQLite — LangGraph conversation state (gitignored)
    │       └── sessions.db           # SQLite — per-user chat index for history/delete (gitignored)
    └── evals/                        # imported as `from evals...`
        ├── golden/                   # factual, adversarial, sql, numbers, safety, routing …
        ├── results/                  # JUnit XML + logs (gitignored)
        ├── conftest.py               # shared fixtures, adapter injection
        └── test_*.py                 # rbac, safety, chunk_quality, routing, sql, retrieval,
                                      #   numbers, faithfulness, calibration, intent, disclaimer
```

---

## 3. Adapter pattern

Each external dependency is behind an interface. The app code imports the interface,
never the vendor SDK directly.

```
service layer          adapter layer            external
─────────────          ─────────────            ────────
retrieval_service  →   qdrant_adapter       →   Qdrant
                       embedding_adapter    →   sentence-transformers
                       sparse_adapter       →   FastEmbed
generation_service →   llm_adapter          →   Groq API
rerank_service     →   reranker_adapter     →   cross-encoder model
sql_rag_service    →   sql_adapter          →   SQLite
                       llm_adapter          →   Groq API
```

---

## 4. Pydantic boundaries (where validation lives)

```
API boundary (in):    ChatRequest, ClarifyResolveRequest, ConfirmRequest → schemas.py
API boundary (out):   ChatResponse, Source, SQLProposal, ClarifyProposal → schemas.py
Ingestion:            ChunkMetadata                     → chunk.py
Intent gate:          IntentResolution                  → schemas.py
Router decision:      RouteDecision, TwoAxisRouteDecision → schemas.py
SQL output:           SQLResult                         → schemas.py
Auth:                 User, TokenPayload                → auth.py
Golden set:           RoutingCase, AdversarialCase, IntentCase, DisclaimerCase → backend/evals/eval_schemas.py
Config:               Settings (pydantic-settings)       → settings.py
```

Every boundary validates shape and type before the next stage trusts the data. A chunk
missing `access_roles` fails at ingest, not at query time. A route decision that isn't
`hybrid_rag` or `sql_rag` fails at the router, not downstream.

---

## 5. Logging structure

Every pipeline stage logs a structured record. The eval harness reads these.

| Stage | What is logged | Used by eval |
|---|---|---|
| Intent gate | intent_state + candidate_count + reason (never the query) | intent-gate confusion matrix |
| Router | route decision + reasoning | routing confusion matrix |
| Retrieval | top-K chunk IDs + scores + role filter applied | retrieval P@3, RBAC leak check |
| Reranker | before/after scores + final top-3 | rerank lift measurement |
| Generation | prompt + response + sources used | faithfulness scoring |
| SQL RAG | generated SQL + execution result + NL answer | SQL correctness |

Use Python `logging` with JSON formatter. Each log record includes a `request_id` for
trace correlation.

---

## 6. Auth model (simple, demo-grade)

Hardcoded user table (fine for portfolio):

| Username | Password | Role |
|---|---|---|
| dr.mehta | doctor123 | doctor |
| nurse.priya | nurse123 | nurse |
| billing.ravi | billing123 | billing_executive |
| tech.anand | tech123 | technician |
| admin.sys | admin123 | admin |

JWT payload: `{"sub": "dr.mehta", "role": "doctor", "exp": ...}`.
Token lifetime: 24 hours. No refresh — this is a demo.


