# Medi-Bot-AI — RBAC-Aware Clinical RAG Assistant

Medibot-AI is an internal AI assistant.Staff ask natural-language questions and get **cited answers from the right
documents** — scoped to what their role is allowed to see.

---

## Highlights

| Capability | What it does |
|---|---|
| **Hybrid RAG** | Dense (BGE) + sparse (BM25) retrieval fused in Qdrant, then a cross-encoder reranker. Answers are grounded in retrieved chunks with source citations. |
| **RBAC as a pre-filter** | The role's collection allow-list is applied *inside* the Qdrant query — restricted chunks are never returned to the app or the LLM, not post-filtered after the fact. |
| **SQL RAG** | Natural-language → SQL over the claims/maintenance database, gated to `billing_executive` and `admin`, with PHI columns stripped before narration. |
| **Intent gate (admission control)** | Before any retrieval, a gate classifies each turn as *resolved* (answer it), *ambiguous* (ask which prior topic — HITL clarify), or *empty* (refuse; never retrieve). Stops contentless/keyword-collision inputs from producing confident wrong answers. |
| **LangGraph orchestration** | Intent gate → router (analytical vs knowledge) → path; multi-turn memory (SqliteSaver) and human-in-the-loop `interrupt()` for both SQL approval **and** ambiguity clarification are native. |
| **Deterministic medical safety** | Disclaimers appended in code (not the LLM) and *content-keyed* (only when clinical/nursing is central), a relevance floor that drops low-score noise before generation, plus a guard blocking unit conversions and prescriptive language. |
| **Persistent chat management** | Per-user chat history survives logout (sidebar shows the 5 most recent; the DB keeps all), 5-turn cap per chat, New Chat, and delete (purges both the session index and the checkpointer content). |
| **Eval harness** | 11 built tracks (RBAC red-team, retrieval A/B, SQL correctness, routing, numeric faithfulness, chunk quality, safety, judge calibration, **intent gate**, **disclaimer keying + relevance floor**) with an admin dashboard. |
| **PHI-safe observability** | Structured JSON logs correlated by `request_id` (`X-Request-ID` header), per-stage latency, and per-request LLM token/cost accounting (`X-LLM-Tokens`). Never logs query or answer text. |

---

## Architecture

```mermaid
flowchart TB
    U([Staff user · JWT role]):::client --> UI[Streamlit UI<br/>chat · admin eval dashboard]:::client

    subgraph RT[Request pipeline · FastAPI + LangGraph]
        direction TB
        MW[Request middleware<br/>request_id · latency · token meter]:::infra --> IG{Intent gate<br/>resolved · ambiguous · empty}:::node
        IG -. ambiguous .-> CLR[HITL clarify<br/>did you mean X or Y?]:::node
        IG -. empty .-> REF[refuse<br/>no retrieval]:::node
        IG -->|resolved| RTR{Router<br/>route × specificity}:::node
        subgraph KP[Knowledge path · hybrid RAG]
            direction LR
            RET[retrieve]:::node --> RRK[rerank]:::node --> GEN[generate<br/>relevance floor · safety guard<br/>content-keyed disclaimers]:::node
        end
        subgraph AP[Analytical path · SQL RAG]
            direction LR
            SQP[SQL propose]:::node -. HITL approve .-> SQX[execute → narrate<br/>PHI stripped]:::node
        end
        RTR -->|knowledge| RET
        RTR -->|analytical| SQP
    end
    UI --> MW

    ENC[BGE + BM25 · cross-encoder<br/>local models]:::model
    GROQ[Groq LLM · llama-3.3-70b]:::model
    QD[(Qdrant<br/>🔒 RBAC pre-filter · hybrid)]:::store
    SDB[(SQLite · claims · tickets)]:::store
    RET --> QD & ENC
    RRK --> ENC
    RTR --> GROQ
    GEN --> GROQ
    SQX --> GROQ & SDB

    subgraph CC[Enforced on every request]
        direction LR
        C1[RBAC pre-filter<br/>role → collections]:::cc
        C2[Safety guard<br/>no unit-convert · no advice]:::cc
        C3[PHI handling<br/>strip patient_* · never log text]:::cc
        C4[Denial verbalizer<br/>deterministic refusals]:::cc
    end

    subgraph EO[Evaluation & Observability]
        direction LR
        E1[Eval harness · 11 tracks<br/>rbac · retrieval · sql · routing · safety · numbers<br/>chunk_quality · faithfulness · calibration<br/>intent gate · disclaimer keying + floor]:::eval
        E2[DeepEval<br/>LLM-judge]:::eval
        E3[Admin eval<br/>dashboard]:::eval
        E4[Observability<br/>JSON logs · request_id / X-Request-ID<br/>per-stage latency · token/cost]:::obs
    end

    RT -. applied to every stage .-> CC
    RT -. graded + traced .-> EO

    classDef client fill:#e3f2fd,stroke:#1565c0,color:#0d47a1;
    classDef infra fill:#eceff1,stroke:#546e7a,color:#263238;
    classDef node fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;
    classDef model fill:#fff3e0,stroke:#ef6c00,color:#e65100;
    classDef store fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c;
    classDef cc fill:#fff8e1,stroke:#f9a825,color:#5f4300;
    classDef eval fill:#e8eaf6,stroke:#3f51b5,color:#1a237e;
    classDef obs fill:#e0f7fa,stroke:#00838f,color:#006064;

    style RT fill:#fbfdff,stroke:#90caf9
    style KP fill:#f4fbf5,stroke:#c5e1a5
    style AP fill:#f4fbf5,stroke:#c5e1a5
    style CC fill:#fffdf5,stroke:#f9a825
    style EO fill:#f5f6ff,stroke:#3f51b5
```

**Ingestion (offline, run once):**

```mermaid
flowchart LR
    D[PDFs · Markdown<br/>data/docs]:::store --> P[Docling parse]:::node
    P --> C[Hierarchical chunker<br/>+ parent headings]:::node
    C --> V[Pydantic ChunkMetadata<br/>access_roles · collection]:::node
    V --> E[BGE dense + BM25 sparse]:::model
    E --> Q[(Qdrant · medibot<br/>access_roles indexed for RBAC)]:::store

    classDef node fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;
    classDef model fill:#fff3e0,stroke:#ef6c00,color:#e65100;
    classDef store fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c;
```
**Repository Layout**

<img width="311" height="455" alt="Screenshot 2026-07-25 at 9 18 15 PM" src="https://github.com/user-attachments/assets/4b9b64cc-73b5-402e-8543-943cfe61fd25" />

- **Ingestion (offline):** Docling parses PDFs/MD → HierarchicalChunker → Pydantic-
  validated `ChunkMetadata` → dense + sparse embeddings → Qdrant (one `medibot` collection,
  `access_roles` indexed for the RBAC pre-filter).
- **Query (per request):** intent gate (admission control — resolve / clarify / refuse) →
  router → hybrid retrieval (RBAC pre-filter) → cross-encoder rerank → relevance floor →
  grounded generation with content-keyed disclaimers; **or** the SQL path.
- **Three-layer rule:** graph nodes → services → adapters. Each layer only calls the next;
  vendor SDKs live only in adapters.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full pipeline, model inventory,
and Pydantic boundaries.

---

## Tech stack

| Concern | Choice |
|---|---|
| PDF parsing / chunking | Docling + HierarchicalChunker |
| Dense embedding | `BAAI/bge-small-en-v1.5` (local, CPU) |
| Sparse embedding | FastEmbed / Qdrant BM25 (local) |
| Vector store | Qdrant (hybrid search + metadata filter) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Orchestration | LangGraph (StateGraph, SqliteSaver, `interrupt()`) |
| Backend | FastAPI + Pydantic, JWT auth |
| Frontend | Streamlit (chat tab + admin eval dashboard) |
| Evals | pytest + numpy (deterministic) · DeepEval (LLM-judge) |
| Lint / types | ruff · mypy |

---

## Quick start (Docker — recommended)

**Prerequisites:** Docker, and a free [Groq API key](https://console.groq.com).

```bash
# 1. Configure env
cp .env.example .env

# edit .env:  GROQ_API_KEY=your_key   (QDRANT_PORT=6335 already set for compose)

# 2. Build + start. First run ingests the corpus (downloads embedding models, a few minutes), then the backend + frontend come up with a populated index.
docker compose up -d --build

# 3. Open the app
open http://localhost:8700
```

Startup ordering is automatic (no race): `qdrant → ingest (populate, then exit) →
backend (waits for ingest to complete, then healthcheck) → frontend`.

| Service | URL |
|---|---|
| Frontend (Streamlit) | http://localhost:8700 |
| Backend (FastAPI) | http://localhost:8001 · docs at `/docs` |
| Qdrant dashboard | http://localhost:6335/dashboard |

```bash
docker compose logs -f backend      # follow backend logs

docker compose logs ingest          # ingestion progress / "skip (already populated)"

docker compose down                 # stop
```

---

## Demo accounts & RBAC matrix

One-click login buttons are on the sign-in page (passwords never rendered).

| Role | Username | Document collections | SQL RAG |
|---|---|---|---|
| Doctor | `dr.mehta` | clinical, nursing, general | ✗ |
| Nurse | `nurse.priya` | nursing, general | ✗ |
| Billing | `billing.ravi` | billing, general | ✓ |
| Technician | `tech.anand` | equipment, general | ✗ |
| Admin | `admin.sys` | **all** | ✓ |

**Try it:**
- As **Doctor**: *"first-line drug for Type 2 Diabetes?"* → cited clinical answer.
  <img width="1865" height="760" alt="docter" src="https://github.com/user-attachments/assets/5b4a9317-cc31-40b6-aca8-fd9c051e9ffd" />
  
- As **Billing**: *"how many claims are escalated?"* → SQL RAG answer (`8`).
  <img width="1913" height="636" alt="Screenshot 2026-07-25 at 9 27 37 PM" src="https://github.com/user-attachments/assets/eb5dc748-ae80-4dd8-bb99-4f1288ffb430" />

- RBAC red-team: the same NSTEMI query returns *billing* data to Billing and *clinical*
  protocol to Doctor — never crossing collections.

---

## Local dev (venv, no Docker for the app)

```bash
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt -r requirements-dev.txt

docker compose up -d qdrant  #Qdrant only (host 6335)

# ingest once, then run backend + frontend
PYTHONPATH=backend python backend/ingestion/ingest.py

PYTHONPATH=backend uvicorn src.api.app:app --reload --port 8000

MEDIBOT_API_URL=http://localhost:8000 streamlit run frontend/app.py --server.port 8700
```

---

## Evaluations

The eval harness is the point of the project. Tracks are **run on demand from the admin
Evaluation tab** (`POST /evals/run`), or headless:

```bash
# all tracks (needs Groq for judge tracks) — writes JUnit XML + logs to backend/evals/results/

pytest backend/evals/ -v

# deterministic-only (no Groq tokens), RBAC first / fail-fast

pytest backend/evals/test_rbac.py -x && \
  pytest backend/evals/test_safety.py backend/evals/test_chunk_quality.py
```

`Tracks`: 
- RBAC red-team (zero-leak, fail-closed), 
- retrieval A/B (P@3/MRR), 
- SQL correctness,
- routing accuracy, 
- numeric faithfulness, 
- chunk quality, 
- medical safety, 
- judge calibration,
- intent gate (multi-turn admission control — resolved/ambiguous/empty), 
- disclaimer keying + relevance floor (deterministic, no LLM).

Thresholds live in `backend/evals/eval_config.py`. See [`docs/EVAL_PLAN.md`](docs/EVAL_PLAN.md).

> **Note:** Groq's free tier has per-minute and per-day token caps; the LLM-judge tracks
> (faithfulness, calibration) can hit them. Run the deterministic tracks freely; pace the
> judge tracks. Rate-limit errors surface as a clean "try again in X" message.

---

## Observability

PHI-safe by design — only IDs, counts, scores, and durations are ever logged.

- **Correlation:** every request gets a `request_id`, threaded through all stages and
  returned as the `X-Request-ID` response header (propagated if the client sends one).
- **Latency:** `latency_ms` per stage (router, retrieval, rerank, generation, SQL) plus a
  total `duration_ms` on the `http_request` event.
- **Cost:** per-request Groq token spend in the `X-LLM-Tokens` header and logs.

Visual tracing (Arize Phoenix) is planned — see Roadmap.

---

## Roadmap

- **Agentic SQL tool-loop** — replace the linear SQL path with a ReAct sub-graph
  (multi-step queries, self-correction). *(Phase 12)*
- **Distributed tracing with Arize Phoenix** — per-request span waterfalls + latency/cost
  dashboards on top of the existing PHI-safe logs. *(Phase 12)*
- **Extension eval tracks** — intent-gate and disclaimer-keying tracks are **built**;
  dedicated multi-turn-retrieval-lift and HITL-SQL-token-lifecycle tracks are next.

