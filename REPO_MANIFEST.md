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
| 15 | `.env.example` | Template for env vars (committed) | Phase 0 |
| 16 | `.env` | Actual env vars (gitignored, never committed) | Phase 0 |
| 17 | `.gitignore` | See gitignore section below | Phase 0 |
| 18 | `requirements.txt` | Python dependencies | Phase 0 |
| 19 | `docker-compose.yml` | Qdrant + backend + frontend services | Phase 0 |
| 20 | `Dockerfile.backend` | Backend container | Phase 10 |
| 21 | `Dockerfile.frontend` | Frontend container | Phase 10 |
| 22 | `pyproject.toml` | (optional) if using poetry/hatch instead of requirements.txt | Phase 0 |
| 23 | `README.md` | Project overview, setup, architecture diagram, demo credentials, adversarial examples, deployment link | Phase 10 (final) |

### Data files

| # | File | Purpose | Build phase |
|---|---|---|---|
| 24 | `data/docs/*.pdf` | All 11 source PDFs | Phase 0 (copy in) |
| 25 | `data/docs/claim_submission_guide.md` | Billing markdown doc | Phase 0 (copy in) |
| 26 | `data/mediassist.db` | SQLite database (claims + tickets) | Phase 0 (copy in) |

### Scripts

| # | File | Purpose | Build phase |
|---|---|---|---|
| 27 | `scripts/ingest.py` | Parse PDFs → chunk → embed → store in Qdrant. Run once, standalone | Phase 1 |

### Source — config

| # | File | Purpose | Build phase |
|---|---|---|---|
| 28 | `src/__init__.py` | Package marker | Phase 0 |
| 29 | `src/config/__init__.py` | Package marker | Phase 0 |
| 30 | `src/config/settings.py` | pydantic-settings: all env vars, model names, thresholds | Phase 0 |
| 31 | `src/config/access.py` | ROLE_COLLECTIONS map, SQL_RAG_ROLES set, collection enum | Phase 0 |

### Source — models (Pydantic schemas)

| # | File | Purpose | Build phase |
|---|---|---|---|
| 32 | `src/models/__init__.py` | Package marker | Phase 0 |
| 33 | `src/models/schemas.py` | ChatRequest, ChatResponse, Source, RouteDecision, SQLResult | Phase 0 |
| 34 | `src/models/chunk.py` | ChunkMetadata (validated at ingest), ChunkResult (returned from search) | Phase 1 |
| 35 | `src/models/auth.py` | User, TokenPayload, LoginRequest, LoginResponse | Phase 6 |

### Source — adapters

| # | File | Purpose | Build phase |
|---|---|---|---|
| 36 | `src/adapters/__init__.py` | Package marker | Phase 0 |
| 37 | `src/adapters/qdrant_adapter.py` | Qdrant client wrapper: insert, hybrid_search with RBAC filter | Phase 1-2 |
| 38 | `src/adapters/embedding_adapter.py` | Dense embedding (bge-small-en-v1.5) | Phase 1 |
| 39 | `src/adapters/sparse_adapter.py` | Sparse embedding (FastEmbed) | Phase 1 |
| 40 | `src/adapters/reranker_adapter.py` | Cross-encoder rerank | Phase 3 |
| 41 | `src/adapters/llm_adapter.py` | Groq API wrapper: generate, structured output | Phase 4 |
| 42 | `src/adapters/sql_adapter.py` | SQLite wrapper: execute query, return results | Phase 5 |

### Source — services

| # | File | Purpose | Build phase |
|---|---|---|---|
| 43 | `src/services/__init__.py` | Package marker | Phase 0 |
| 44 | `src/services/ingestion_service.py` | Docling parse → HierarchicalChunker → validate → embed → store | Phase 1 |
| 45 | `src/services/retrieval_service.py` | Embed query → RBAC filter → Qdrant hybrid search → return chunks | Phase 2 |
| 46 | `src/services/rerank_service.py` | Take top-K → cross-encoder score → return top-N with scores | Phase 3 |
| 47 | `src/services/generation_service.py` | Build prompt from chunks → LLM call → append disclaimer → return | Phase 4 |
| 48 | `src/services/router_service.py` | LLM classify query → RouteDecision (sql_rag / hybrid_rag) | Phase 5 |
| 49 | `src/services/sql_rag_service.py` | Role check → LLM generate SQL → clean → execute → LLM narrate | Phase 5 |

### Source — API

| # | File | Purpose | Build phase |
|---|---|---|---|
| 50 | `src/api/__init__.py` | Package marker | Phase 6 |
| 51 | `src/api/app.py` | FastAPI app, mount routers, exception handlers, DI wiring | Phase 6 |
| 52 | `src/api/chat_router.py` | /chat, /collections/{role}, /health endpoints | Phase 6 |
| 53 | `src/api/auth_router.py` | /login endpoint, JWT creation | Phase 6 |

### Source — security

| # | File | Purpose | Build phase |
|---|---|---|---|
| 54 | `src/security/__init__.py` | Package marker | Phase 6 |
| 55 | `src/security/jwt.py` | JWT encode/decode, role extraction, token validation | Phase 6 |

### Source — utils

| # | File | Purpose | Build phase |
|---|---|---|---|
| 56 | `src/utils/__init__.py` | Package marker | Phase 0 |
| 57 | `src/utils/logger.py` | Structured JSON logging, PHI-safe (no query/response text) | Phase 0 |
| 58 | `src/utils/helpers.py` | Shared utilities (request ID generation, text cleaning) | Phase 0 |
| 59 | `src/utils/exceptions.py` | MediBotError, RetrievalError, RBACError, SQLGenerationError, RoutingError | Phase 0 |

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
| 66 | `evals/__init__.py` | Package marker | Phase 8 |
| 67 | `evals/conftest.py` | Shared fixtures: adapter injection, golden set loading, Pydantic models for test cases | Phase 8 |
| 68 | `evals/golden/factual.json` | 23+ factual Q&A cases with expected answers, sources, critical numbers | Phase 8 |
| 69 | `evals/golden/adversarial.json` | 15+ RBAC attack cases with attacker role, target collection, attack type | Phase 8 |
| 70 | `evals/golden/sql.json` | 8+ SQL-RAG cases with expected results | Phase 8 |
| 71 | `evals/test_retrieval.py` | P@3, MRR: dense vs hybrid vs +rerank | Phase 8 |
| 72 | `evals/test_rbac.py` | Leak-rate matrix, fail-closed assertions | Phase 8 |
| 73 | `evals/test_faithfulness.py` | DeepEval faithfulness + custom number-match | Phase 8 |
| 74 | `evals/test_sql.py` | SQL execution success + result match | Phase 8 |
| 75 | `evals/test_routing.py` | Confusion matrix (analytical vs knowledge) | Phase 8 |
| 76 | `evals/test_chunk_quality.py` | Parent heading present, table integrity | Phase 8 |
| 77 | `evals/results/` | Directory (gitignored) for eval output logs | Phase 8 |

### Unit tests

| # | File | Purpose | Build phase |
|---|---|---|---|
| 78 | `tests/__init__.py` | Package marker | Phase 2+ |
| 79 | `tests/test_retrieval_service.py` | Unit test: retrieval with mocked Qdrant | Phase 2 |
| 80 | `tests/test_rerank_service.py` | Unit test: reranking with mocked encoder | Phase 3 |
| 81 | `tests/test_generation_service.py` | Unit test: generation with mocked LLM | Phase 4 |
| 82 | `tests/test_router_service.py` | Unit test: routing with mocked LLM | Phase 5 |
| 83 | `tests/test_sql_rag_service.py` | Unit test: SQL chain with mocked LLM + real SQLite | Phase 5 |
| 84 | `tests/test_auth.py` | Unit test: JWT encode/decode, role extraction | Phase 6 |
| 85 | `tests/conftest.py` | Shared test fixtures, mock adapters | Phase 2 |

### CI

| # | File | Purpose | Build phase |
|---|---|---|---|
| 86 | `.github/workflows/eval.yml` | CI eval gate: Qdrant service, ingest, pytest evals/ | Phase 8 |
| 87 | `.github/workflows/test.yml` | CI unit tests: pytest tests/ | Phase 2 |

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
evals/results/

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
| Documentation (docs/) | 11 |
| Claude Code rules (.claude/rules/) | 3 |
| Root config | 9 |
| Data files | 3 |
| Scripts | 1 |
| Source code (src/) | 32 |
| Frontend | 6 |
| Evals | 12 |
| Unit tests | 8 |
| CI workflows | 2 |
| **Total** | **87** |

Of these, **14 are built** (11 docs + 3 rules). The remaining **73 are built phase by
phase** during development. Each has a clear build-phase assignment so nothing is
ambiguous about when it's created.
