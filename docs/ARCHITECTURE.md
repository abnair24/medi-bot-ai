# MediBot — Architecture

> Component layout, LangGraph state machine, adapter pattern, Pydantic boundaries,
> and folder structure. Read this before writing code.
>
> **Companion:** [`ARCHITECTURE_REVIEW.md`](ARCHITECTURE_REVIEW.md) — the forward-looking
> design review (improvement backlog, portfolio-vs-prod judgment, deferred prod design,
> findings log). This doc = *what is built*; the review = *what to build next and why*.

---

## 1. Query flow (text diagram)

```
User → Login (JWT) → /chat
                        │
                   extract role from JWT
                        │
               LANGGRAPH STATE GRAPH
               ┌────────────────────────────────────────────┐
               │  State: {question, role, history, route,  │
               │         chunks, answer, proposed_sql}     │
               ├────────────────────────────────────────────┤
               │  REWRITE NODE (if follow-up question)     │
               │       │                                   │
               │  ROUTER NODE (LLM)                        │
               │  ┌────┴────┐                               │
               │ ANALYTICAL  KNOWLEDGE                     │
               │  │            │                            │
               │ SQL_PROPOSE RETRIEVE → RERANK → GENERATE  │
               │  │                                        │
               │ interrupt() ←─ HITL pause                 │
               │  │ (user approves/rejects)                │
               │ SQL_EXECUTE → NARRATE                     │
               │       │                                   │
               │  RESPONSE NODE                             │
               └───────────┴────────────────────────────────┘
                        │
               response: answer + sources + retrieval_type
               (state persisted via SqliteSaver for multi-turn)
```

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
LLM ANSWER GENERATION ──────────────────────── LLM call #2
     │   model: Groq API (Llama-3 / similar)           (the main LLM call)
     │   input: question + top-3 chunks as context
     │   prompt: see PROMPTS.md §2
     │   output: grounded answer + source citations
     │   constraint: answer ONLY from context, never fabricate
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

### 2.5 What vLLM and MCP are (and why they're not in this architecture)

**vLLM** is an inference server for self-hosting LLMs on your own GPU. It replaces the Groq
API — instead of calling Groq, you'd run Llama-3 on your machine via vLLM. You'd use it
if you needed to self-host for cost, privacy, or latency. For this project, Groq is faster
and cheaper at this scale. If asked: "I'd swap the llm_adapter to point at a vLLM endpoint
instead of Groq — the adapter pattern makes that a one-file change."

**MCP** (Model Context Protocol) is a protocol for connecting external tools/data-sources
to an LLM agent — like how Claude connects to Google Drive or Slack. It's an
interoperability standard, not an inference layer. Your adapters serve the same purpose at
the application level: they're your app's interface to Qdrant, SQLite, and the LLM. MCP
would matter if you were publishing MediBot as a tool that plugs into someone else's agent
(e.g., exposing MediBot's retrieval as an MCP server that Claude or another agent could
call). That's a later extension, not a foundation.

---

## 3. Folder structure

```
medibot/
├── INSTRUCTIONS.md
├── CLAUDE.md
├── docker-compose.yml
├── pytest.ini                        # pythonpath = backend; testpaths = backend/evals
├── requirements.txt
├── .github/
│   └── workflows/
│       └── eval.yml                  # CI gate — runs evals on push
├── docs/                             # ARCHITECTURE.md, EVAL_PLAN.md, PROMPTS.md, …
├── frontend/
│   └── app.py                        # Streamlit — Tab 1 (demo) + Tab 2 (eval)
└── backend/                          # everything importable lives here (PYTHONPATH=backend)
    ├── src/                          # FastAPI app — imported as `from src...`
    │   ├── graph/                    # LangGraph orchestration layer
    │   │   ├── state.py              # MediBotState TypedDict
    │   │   ├── nodes.py              # node functions (call services)
    │   │   ├── edges.py              # conditional edge functions
    │   │   └── builder.py            # build_graph() — assemble + compile
    │   ├── api/
    │   │   ├── app.py                # FastAPI app, mount routers, compile graph
    │   │   ├── chat_router.py        # /chat, /chat/resume, /collections, /health
    │   │   ├── eval_router.py        # /evals/run, /evals/status, /evals/results (admin)
    │   │   └── auth_router.py        # /login, JWT
    │   ├── services/                 # retrieval, rerank, generation, sql_rag, router,
    │   │   │                         #   rewrite, hyde, denial, safety_guard, eval
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
    │       └── checkpoints.db        # SQLite — LangGraph state (gitignored)
    └── evals/                        # imported as `from evals...`
        ├── golden/                   # factual, adversarial, sql, numbers, safety, routing …
        ├── results/                  # JUnit XML + logs (gitignored)
        ├── conftest.py               # shared fixtures, adapter injection
        └── test_*.py                 # rbac, safety, chunk_quality, routing, sql,
                                      #   retrieval, numbers, faithfulness, calibration
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

**Why this matters for eval:** in tests, you inject a fake/mock adapter. The RBAC test
injects a real Qdrant adapter (to test the actual metadata filter) but a fake LLM adapter
(to isolate the retrieval check). The faithfulness test injects a real LLM but a
deterministic retriever (to test generation in isolation). DI makes stage-by-stage eval
possible.

---

## 4. Pydantic boundaries (where validation lives)

```
API boundary (in):    ChatRequest, LoginRequest         → schemas.py
API boundary (out):   ChatResponse, Source              → schemas.py
Ingestion:            ChunkMetadata                     → chunk.py
Router decision:      RouteDecision                     → schemas.py
SQL output:           SQLResult                         → schemas.py
Auth:                 User, TokenPayload                → auth.py
Golden set:           GoldenCase, AdversarialCase        → backend/evals/conftest.py
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

---

## 7. Production considerations (known limitations)

Four architectural items to address during build. These are acceptable for a demo but
would need resolution for production traffic.

### 7.1 Async pipeline execution

FastAPI is async, but the local models (embedding, sparse, reranker) run synchronously
on CPU. Under concurrent requests, synchronous model calls block the event loop and
serialize all requests.

**Fix:** wrap every local-model call in `asyncio.to_thread()`:

```python
# In retrieval_service.py
dense_vector = await asyncio.to_thread(self.embedder.embed, query)
sparse_vector = await asyncio.to_thread(self.sparse.embed, query)

# In rerank_service.py
scores = await asyncio.to_thread(self.reranker.rerank, query, chunks)
```

This moves CPU-bound work to a thread pool. The event loop stays free for I/O
(Qdrant queries, Groq API calls). Apply this pattern in every service that calls a
local model — retrieval, rerank, and ingestion.

### 7.2 JWT secret default value

The settings default `jwt_secret = "change-this-in-production"` will be forgotten in
deployed environments. Add a startup guard in `app.py`:

```python
# In backend/src/api/app.py — lifespan or startup event
if settings.jwt_secret == "change-this-in-production":
    if os.getenv("ENVIRONMENT", "dev") != "dev":
        raise RuntimeError("JWT_SECRET must be changed in non-dev environments")
    logger.warning("Using default JWT secret — dev only")
```

Generate a real secret during deployment setup:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 7.3 SQLite under concurrency

SQLite's default mode disallows sharing connections across threads. The SQL adapter must
use `check_same_thread=False` and enable WAL mode for concurrent reads:

```python
# In sql_adapter.py
self._conn = sqlite3.connect(path, check_same_thread=False)
self._conn.execute("PRAGMA journal_mode=WAL")  # allows concurrent readers
self._conn.execute("PRAGMA query_only=ON")      # read-only guard for query path
```

For full async compatibility, consider `aiosqlite` as a drop-in wrapper. Since the
SQL-RAG path only runs read queries, a single read-only connection with WAL mode is
sufficient for demo-scale concurrency.

### 7.4 Groq API rate limiting

Groq's free tier has aggressive rate limits (~30 RPM). A demo with multiple concurrent
users will hit 429 errors. Two mitigations:

**1. Retry with exponential backoff in `llm_adapter.py`:**

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type(RateLimitError),
)
async def generate(self, prompt: str) -> str:
    ...
```

**2. Request-level rate limiting on the FastAPI app:**

```python
# In backend/src/api/app.py — use a simple middleware or dependency
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# In chat_router.py
@router.post("/chat")
@limiter.limit("10/minute")
async def chat(request: Request, body: ChatRequest):
    ...
```

This prevents a single user from exhausting the Groq quota for everyone.
