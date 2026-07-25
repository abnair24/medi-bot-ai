# MediBot — Complete Repo Manifest

> Every file in the repo. Nothing missing. Three sections: docs (you read), .claude
> (Claude Code reads), and source (you/Claude Code build).

---

## Section A: Documentation (repo root — `docs/`)

These are reference files for humans and Claude Code. All built and ready.
Place them in `docs/` to keep root clean, except INSTRUCTIONS.md which stays at root.

| # | File | Location | Purpose | Status |
|---|---|---|---|---|
| 1 | `INSTRUCTIONS.md` | `/` (repo root) | Master spec — everything starts here | ✅ built |
| 2 | `ARCHITECTURE.md` | `docs/` | RAG pipeline (ingestion + query paths), folder structure, adapters, DI, Pydantic boundaries, model inventory, vLLM/MCP clarification | ✅ built |
| 3 | `CODING_STANDARDS.md` | `docs/` | Python engineering: pydantic-settings, no hardcoding, design principles (SRP/DI/ISP/OCP), error handling, naming, docstrings, testing conventions | ✅ built |
| 4 | `EVAL_PLAN.md` | `docs/` | All 7 eval tracks: retrieval A/B, RBAC red-team, numeric faithfulness, SQL correctness, routing accuracy, chunk quality, judge calibration. Metrics, tools, thresholds, CI config | ✅ built |
| 5 | `GOLDEN_SET.md` | `docs/` | Golden dataset schema (Pydantic) + 43 starter cases: 23 factual, 15 adversarial, 8 SQL. All verified against actual corpus/DB | ✅ built |
| 6 | `RBAC_REDTEAM.md` | `docs/` | Full adversarial matrix: 5 attack types × 3 attacker roles, NSTEMI semantic-overlap case, verification pseudocode, leak-rate reporting format | ✅ built |
| 7 | `DATA_DICTIONARY.md` | `docs/` | Document inventory (12 files, 5 collections), DB schema (claims 85 rows, tickets 78 rows), distinct values, pre-computed golden SQL answers, cross-document overlap map | ✅ built |
| 8 | `PROMPTS.md` | `docs/` | Every LLM prompt: semantic router, RAG generation, RBAC refusal, SQL generation, SQL narration, DeepEval faithfulness rubric, correctness rubric. Versioning guidance | ✅ built |
| 9 | `GUARDRAILS.md` | `docs/` | Medical AI safety: 5 "never do" rules, mandatory disclaimers (clinical/drug/emergency), adversarial handling, positive-list of allowed behaviors | ✅ built |
| 10 | `DATA_HANDLING.md` | `docs/` | PHI rules: what counts as PHI, logging do/don't with code examples, storage rules, SQL result sanitization, eval vs production logging separation, API transmission risk | ✅ built |
| 11 | `DEPLOY.md` | `docs/` | Local dev setup, docker-compose.yml, Dockerfiles, cloud deploy (Streamlit Cloud, Railway, VPS), .env.example, CI eval workflow YAML | ✅ built |
| 12 | `EXTENSIONS.md` | `docs/` | Phase 11 enhancements: conversational memory (multi-turn + query rewrite), HITL SQL RAG (2-phase approval), 3 extension eval tracks with golden datasets | ✅ built |
| 13 | `ARCHITECTURE_REVIEW.md` | `docs/` | Principal-level design review: Route→Decide→Enforce principle, Intent×Target routing gap, 7-item improvement backlog, portfolio-vs-prod decision matrix, deferred prod design (OPA/RLS/audit), findings log. Companion to ARCHITECTURE.md | ✅ built |

---

## Section B: Claude Code rules (`/.claude/rules/`)

Machine-readable rules Claude Code reads on **every invocation**. These are NOT
documentation — they are persistent instructions. Keep them concise (Claude Code
reads all of them every time, so token-efficiency matters).

| # | File | Purpose | Status |
|---|---|---|---|
| 12 | `.claude/rules/guardrails.md` | Medical safety boundaries, RBAC enforcement at query level, response format (disclaimers, citations) | ✅ built |
| 13 | `.claude/rules/data-handling.md` | PHI logging prohibitions, SQL result sanitization, gitignore rules, never log query/response text | ✅ built |
| 14 | `.claude/rules/code-quality.md` | No hardcoding (pydantic-settings), design principles (SRP/DI/ISP/OCP), Pydantic everywhere, adapter pattern, error handling, naming, eval rules, testing conventions | ✅ built |

---

## Section C: Source code and config (built during development)

These are the files you/Claude Code create during each build phase. Listed here so
nothing is missed. None of these exist yet — they're built phase by phase.

### Root config files

| # | File | Purpose | Build phase |
|---|---|---|---|
| 15 | `CLAUDE.md` | Claude Code master instructions — read on every session start, routes to correct docs per phase | ✅ built |
| 16 | `.env.example` | Template for env vars (committed) | Phase 0 |
| 17 | `.env` | Actual env vars (gitignored, never committed) | Phase 0 |
| 18 | `.gitignore` | See gitignore section below | Phase 0 |
| 19 | `requirements.txt` | Python dependencies | Phase 0 |
| 20 | `docker-compose.yml` | Qdrant + backend + frontend services | Phase 0 |
| 21 | `Dockerfile.backend` | Backend container | Phase 10 |
| 22 | `Dockerfile.frontend` | Frontend container | Phase 10 |
| 23 | `pyproject.toml` | (optional) if using poetry/hatch instead of requirements.txt | Phase 0 |
| 24 | `README.md` | Project overview, highlights, architecture, tech stack, repo layout, quick start, demo credentials + RBAC matrix, evals, observability, roadmap | ✅ built |


### Data files

| # | File | Purpose | Build phase |
|---|---|---|---|
| 24 | `backend/ingestion/data/docs/*.pdf` | All 11 source PDFs | Phase 0 (copy in) |
| 25 | `backend/ingestion/data/docs/claim_submission_guide.md` | Billing markdown doc | Phase 0 (copy in) |
| 26 | `backend/ingestion/data/mediassist.db` | SQLite database (claims + tickets) | Phase 0 (copy in) |

### Scripts

| # | File | Purpose | Build phase |
|---|---|---|---|
| 27 | `backend/ingestion/ingest.py` | Parse PDFs → chunk → embed → store in Qdrant. Run once, standalone | Phase 1 |

### Source — config

| # | File | Purpose | Build phase |
|---|---|---|---|
| 28 | `backend/src/__init__.py` | Package marker | Phase 0 |
| 29 | `backend/src/config/__init__.py` | Package marker | Phase 0 |
| 30 | `backend/src/config/settings.py` | pydantic-settings: all env vars, model names, thresholds | Phase 0 |
| 31 | `backend/src/config/access.py` | ROLE_COLLECTIONS map, SQL_RAG_ROLES set, collection enum | Phase 0 |

### Source — graph (LangGraph orchestration)

| # | File | Purpose | Build phase |
|---|---|---|---|
| 36 | `backend/src/graph/__init__.py` | Package marker | Phase 0 |
| 37 | `backend/src/graph/state.py` | MediBotState TypedDict — graph state schema | Phase 5 |
| 38 | `backend/src/graph/nodes.py` | Node functions: rewrite, router, retrieve, rerank, generate, sql_propose, sql_execute | Phase 5 |
| 39 | `backend/src/graph/edges.py` | Conditional edge functions: route_decision, should_rewrite | Phase 5 |
| 40 | `backend/src/graph/builder.py` | `build_graph()` — assembles and compiles the StateGraph with SqliteSaver | Phase 5 |

### Source — models (Pydantic schemas)

| # | File | Purpose | Build phase |
|---|---|---|---|
| 41 | `backend/src/models/__init__.py` | Package marker | Phase 0 |
| 42 | `backend/src/models/schemas.py` | ChatRequest, ChatResponse, Source, RouteDecision, SQLResult | Phase 0 |
| 43 | `backend/src/models/chunk.py` | ChunkMetadata (validated at ingest), ChunkResult (returned from search) | Phase 1 |
| 44 | `backend/src/models/auth.py` | User, TokenPayload, LoginRequest, LoginResponse | Phase 6 |

### Source — adapters

| # | File | Purpose | Build phase |
|---|---|---|---|
| 36 | `backend/src/adapters/__init__.py` | Package marker | Phase 0 |
| 37 | `backend/src/adapters/qdrant_adapter.py` | Qdrant client wrapper: insert, hybrid_search with RBAC filter | Phase 1-2 |
| 38 | `backend/src/adapters/embedding_adapter.py` | Dense embedding (bge-small-en-v1.5) | Phase 1 |
| 39 | `backend/src/adapters/sparse_adapter.py` | Sparse embedding (FastEmbed) | Phase 1 |
| 40 | `backend/src/adapters/reranker_adapter.py` | Cross-encoder rerank | Phase 3 |
| 41 | `backend/src/adapters/llm_adapter.py` | Groq API wrapper: generate, structured output | Phase 4 |
| 42 | `backend/src/adapters/sql_adapter.py` | SQLite wrapper: execute query, return results | Phase 5 |

### Source — services

| # | File | Purpose | Build phase |
|---|---|---|---|
| 51 | `backend/src/services/__init__.py` | Package marker | Phase 0 |
| 52 | `backend/src/services/ingestion_service.py` | Docling parse → HierarchicalChunker → validate → embed → store | Phase 1 |
| 53 | `backend/src/services/retrieval_service.py` | Embed query → RBAC filter → Qdrant hybrid search → return chunks | Phase 2 |
| 54 | `backend/src/services/rerank_service.py` | Take top-K → cross-encoder score → return top-N with scores | Phase 3 |
| 55 | `backend/src/services/generation_service.py` | Build prompt from chunks → LLM call → append disclaimer → return | Phase 4 |
| 56 | `backend/src/services/router_service.py` | LLM classify query → RouteDecision (sql_rag / hybrid_rag) | Phase 5 |
| 57 | `backend/src/services/sql_rag_service.py` | Role check → LLM generate SQL → clean → execute → LLM narrate | Phase 5 |
| 58 | `backend/src/services/rewrite_service.py` | LLM-based query rewrite using conversation history from graph state | Phase 5 |

### Source — API

| # | File | Purpose | Build phase |
|---|---|---|---|
| 59 | `backend/src/api/__init__.py` | Package marker | Phase 6 |
| 60 | `backend/src/api/app.py` | FastAPI app, mount routers, compile graph at startup | Phase 6 |
| 61 | `backend/src/api/chat_router.py` | /chat, /chat/resume, /collections/{role}, /health endpoints | Phase 6 |
| 53 | `backend/src/api/auth_router.py` | /login endpoint, JWT creation | Phase 6 |

### Source — security

| # | File | Purpose | Build phase |
|---|---|---|---|
| 54 | `backend/src/security/__init__.py` | Package marker | Phase 6 |
| 55 | `backend/src/security/jwt.py` | JWT encode/decode, role extraction, token validation | Phase 6 |

### Source — utils

| # | File | Purpose | Build phase |
|---|---|---|---|
| 56 | `backend/src/utils/__init__.py` | Package marker | Phase 0 |
| 57 | `backend/src/utils/logger.py` | Structured JSON logging, PHI-safe (no query/response text) | Phase 0 |
| 58 | `backend/src/utils/helpers.py` | Shared utilities (request ID generation, text cleaning) | Phase 0 |
| 59 | `backend/src/utils/exceptions.py` | MediBotError, RetrievalError, RBACError, SQLGenerationError, RoutingError | Phase 0 |

### Frontend

| # | File | Purpose | Build phase |
|---|---|---|---|
| 60 | `frontend/app.py` | Streamlit main: Tab 1 (demo chat) + Tab 2 (eval dashboard) | Phase 7 + 9 |
| 61 | `frontend/components/__init__.py` | Package marker | Phase 7 |
| 62 | `frontend/components/chat.py` | Chat UI: message display, source citations, RBAC refusal | Phase 7 |
| 63 | `frontend/components/login.py` | Login form, role selection, session state | Phase 7 |
| 64 | `frontend/components/sidebar.py` | Role badge, accessible collections, retrieval type | Phase 7 |
| 65 | `frontend/components/eval_dashboard.py` | Retrieval A/B chart, RBAC matrix, faithfulness scores, routing confusion matrix | Phase 9 |

### Evals

| # | File | Purpose | Build phase |
|---|---|---|---|
| 66 | `backend/evals/__init__.py` | Package marker | Phase 8 |
| 67 | `backend/evals/conftest.py` | Shared fixtures: adapter injection, golden set loading, Pydantic models for test cases | Phase 8 |
| 68 | `backend/evals/golden/factual.json` | 23+ factual Q&A cases with expected answers, sources, critical numbers | Phase 8 |
| 69 | `backend/evals/golden/adversarial.json` | 15+ RBAC attack cases with attacker role, target collection, attack type | Phase 8 |
| 70 | `backend/evals/golden/sql.json` | 8+ SQL-RAG cases with expected results | Phase 8 |
| 71 | `backend/evals/test_retrieval.py` | P@3, MRR: dense vs hybrid vs +rerank | Phase 8 |
| 72 | `backend/evals/test_rbac.py` | Leak-rate matrix, fail-closed assertions | Phase 8 |
| 73 | `backend/evals/test_faithfulness.py` | DeepEval faithfulness + custom number-match | Phase 8 |
| 74 | `backend/evals/test_sql.py` | SQL execution success + result match | Phase 8 |
| 75 | `backend/evals/test_routing.py` | Confusion matrix (analytical vs knowledge) | Phase 8 |
| 76 | `backend/evals/test_chunk_quality.py` | Parent heading present, table integrity | Phase 8 |
| 77 | `backend/evals/results/` | Directory (gitignored) for eval output logs | Phase 8 |
| 78 | `backend/evals/test_multiturn.py` | Multi-turn retrieval quality + rewrite faithfulness | Phase 8 |
| 79 | `backend/evals/test_sql_hitl.py` | HITL SQL safety: interrupt/resume lifecycle tests | Phase 8 |
| 80 | `backend/evals/golden/multiturn.json` | Multi-turn golden conversations (3–5 turns each) | Phase 8 |
| 81 | `backend/evals/golden/sql_hitl.json` | HITL golden cases: approve/reject/resume | Phase 8 |

### Unit tests

| # | File | Purpose | Build phase |
|---|---|---|---|
| 82 | `tests/__init__.py` | Package marker | Phase 2+ |
| 83 | `tests/test_retrieval_service.py` | Unit test: retrieval with mocked Qdrant | Phase 2 |
| 84 | `tests/test_rerank_service.py` | Unit test: reranking with mocked encoder | Phase 3 |
| 85 | `tests/test_generation_service.py` | Unit test: generation with mocked LLM | Phase 4 |
| 86 | `tests/test_router_service.py` | Unit test: routing with mocked LLM | Phase 5 |
| 87 | `tests/test_sql_rag_service.py` | Unit test: SQL chain with mocked LLM + real SQLite | Phase 5 |
| 88 | `tests/test_graph.py` | Unit test: graph nodes + edges with mocked services | Phase 5 |
| 89 | `tests/test_rewrite_service.py` | Unit test: query rewrite with mocked LLM | Phase 5 |
| 90 | `tests/test_auth.py` | Unit test: JWT encode/decode, role extraction | Phase 6 |
| 91 | `tests/conftest.py` | Shared test fixtures, mock adapters | Phase 2 |

### CI

| # | File | Purpose | Build phase |
|---|---|---|---|
| 86 | `.github/workflows/eval.yml` | CI eval gate: Qdrant service, ingest, pytest backend/evals/ | Phase 8 |
| 87 | `.github/workflows/test.yml` | CI unit tests: pytest tests/ | Phase 2 |

---

## Addendum: files added after the original inventory

These were created during Phase 8 and the portfolio-upgrade work; they extend (not
replace) the plan above. Listed here to keep the inventory accurate without renumbering.

| File | Purpose | Phase |
|---|---|---|
| `backend/src/services/safety_guard.py` | Deterministic medical-safety violation checks (disclaimer / unit-conversion / prescriptive), shared by generation runtime guard + Track 11 eval | Phase 4/8 |
| `backend/src/services/denial_service.py` | Deterministic RBAC **denial verbalizer** (`verbalize_denial`) — replaced the LLM-generated refusal (zero token cost, auditable, no hallucination) | Phase 8 |
| `backend/src/services/hyde_service.py` | **HyDE** hypothetical-document expansion for the dense retrieval path (gated by `settings.hyde_enabled`) | Portfolio upgrade |
| `backend/ingestion/ensure_ingested.py` | Idempotent ingest (wait for Qdrant → ingest only if empty). Entrypoint for the compose `ingest` service; the backend waits for it to complete | Phase 10 |
| `backend/ingestion/run_evals.py` | Headless full-suite eval runner (manual / CI) — reuses `ensure_ingested`, honours `EVAL_TRACKS`. Not wired to compose | Phase 8 |
| `pytest.ini` | pytest config: `pythonpath = . backend` so tests resolve `from src...` (backend) + `from evals...` (root) | Phase 8 |
| `backend/evals/eval_config.py` | Single source of truth for all eval thresholds | Phase 8 |
| `backend/evals/eval_schemas.py` | Pydantic shapes for golden-set records (validated on load) | Phase 8 |
| `backend/evals/conftest.py` | Shared eval fixtures (real adapters, lazy-imported) | Phase 8 |
| `backend/evals/groq_judge.py` | Groq-backed DeepEval judge model | Phase 8 |
| `backend/evals/test_safety.py` | Track 11 — medical-safety guardrail (deterministic) | Phase 8 |
| `backend/evals/test_numbers.py` | Track 3b — exact number-match guardrail (deterministic) | Phase 8 |
| `backend/evals/test_calibration.py` | Track 7 — judge calibration (Cohen's κ vs human labels) | Phase 8 |
| `backend/evals/golden/safety.json` | Safety guardrail golden cases | Phase 8 |
| `backend/evals/golden/numbers.json` | Number-match golden cases | Phase 8 |
| `backend/evals/golden/routing.json` | Routing accuracy golden cases | Phase 8 |
| `backend/evals/golden/calibration_labels.json` | Human labels for judge-calibration κ | Phase 8 |

Note: `backend/evals/test_multiturn.py`, `backend/evals/test_sql_hitl.py`, and their golden JSONs (rows
78–81 above) remain **planned** — the multi-turn and HITL *features* are built (Phase 5),
but their dedicated eval tracks are not yet implemented.

Two-axis routing (`RouterService.classify_detailed`) and the HyDE toggle were added to
existing files (`router_service.py`, `retrieval_service.py`, `models/schemas.py`,
`config/settings.py`) rather than as new files.

---

## Section D: .gitignore (complete)

```
# Environment
.env
.venv/
venv/
__pycache__/
*.pyc

# Eval outputs (may contain synthetic but still sensitive data)
backend/evals/results/

# Qdrant data (Docker volume)
qdrant_data/

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db

# Build
dist/
*.egg-info/
build/

# Logs
*.log
```

---

## Summary count

| Category | Files |
|---|---|
| Documentation (docs/) | 12 |
| Claude Code rules (.claude/rules/) | 3 |
| Root config | 10 |
| Data files | 3 |
| Scripts | 1 |
| Source code (backend/src/) | 37 |
| Frontend | 6 |
| Evals | 16 |
| Unit tests | 10 |
| CI workflows | 2 |
| **Total** | **100** |

Of these, **16 are built** (12 docs + 3 rules + CLAUDE.md). The remaining **84 are built
phase by phase** during development (Phases 0–10). Each has a clear build-phase assignment
so nothing is ambiguous about when it's created.
