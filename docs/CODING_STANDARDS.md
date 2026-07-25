# MediBot — Coding Standards

> Python engineering standards for the entire codebase. These are not suggestions —
> they are enforced via the Claude Code rules and reviewed on every PR.

---

## 1. No hardcoded values — everything is configurable

### Use pydantic-settings for all configuration

```python
# backend/src/config/settings.py
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # API
    groq_api_key: str
    groq_model: str = "llama-3.1-70b-versatile"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "medibot"

    # Embeddings
    dense_model_name: str = "BAAI/bge-small-en-v1.5"
    sparse_model_name: str = "Qdrant/bm25"

    # Reranker
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 10
    reranker_final_k: int = 3

    # Retrieval
    retrieval_top_k: int = 10

    # SQL
    sqlite_path: str = "backend/ingestion/data/mediassist.db"

    # Auth
    jwt_secret: str = "change-this-in-production"
    jwt_expiry_hours: int = 24

    # Logging
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

# singleton — import this, never construct Settings() in business logic
settings = Settings()
```

### What this means in practice

```python
# WRONG — hardcoded everywhere
chunks = qdrant.search(limit=10)
reranked = reranker.rerank(chunks, top_k=3)
model = "BAAI/bge-small-en-v1.5"

# RIGHT — from config
from src.config.settings import settings

chunks = qdrant.search(limit=settings.retrieval_top_k)
reranked = reranker.rerank(chunks, top_k=settings.reranker_final_k)
model = settings.dense_model_name
```

### The .env.example file (committed to git, .env is not)

```bash
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-70b-versatile
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=medibot
DENSE_MODEL_NAME=BAAI/bge-small-en-v1.5
RERANKER_TOP_K=10
RERANKER_FINAL_K=3
SQLITE_PATH=backend/ingestion/data/mediassist.db
JWT_SECRET=change-this-in-production
LOG_LEVEL=INFO
```

---

## 2. Constants and enums — no magic strings

### Use Literal types and enums, never bare strings in business logic

```python
# WRONG
if role == "billing_executive":  # magic string, typo = silent bug
    ...

# RIGHT — defined once in models
from typing import Literal

Role = Literal["doctor", "nurse", "billing_executive", "technician", "admin"]
Collection = Literal["general", "clinical", "nursing", "billing", "equipment"]
RouteType = Literal["hybrid_rag", "sql_rag"]
ChunkType = Literal["text", "table", "heading", "code"]
```

### Role-collection access map — defined once, not scattered

```python
# backend/src/config/access.py
ROLE_COLLECTIONS: dict[str, list[str]] = {
    "doctor": ["clinical", "nursing", "general"],
    "nurse": ["nursing", "general"],
    "billing_executive": ["billing", "general"],
    "technician": ["equipment", "general"],
    "admin": ["clinical", "nursing", "billing", "equipment", "general"],
}

SQL_RAG_ROLES: set[str] = {"billing_executive", "admin"}
```

```python
# WRONG — access logic duplicated across files
if role in ["billing_executive", "admin"]:
    ...

# RIGHT
from src.config.access import SQL_RAG_ROLES
if role in SQL_RAG_ROLES:
    ...
```

---

## 3. Design principles

### Single Responsibility

Each file does one thing. A service handles business logic, an adapter handles external
communication, a model defines data shapes. If a service is doing HTTP calls or SQL
queries directly, it needs an adapter.

### Dependency Injection

Services receive their dependencies, they don't construct them.

```python
# WRONG — service creates its own adapter
class RetrievalService:
    def __init__(self):
        self.qdrant = QdrantAdapter()      # tightly coupled
        self.embedder = EmbeddingAdapter()  # can't test in isolation

# RIGHT — dependencies injected
class RetrievalService:
    def __init__(self, qdrant: QdrantAdapter, embedder: EmbeddingAdapter):
        self.qdrant = qdrant
        self.embedder = embedder
```

Wiring happens in one place (app.py or a factory), not scattered across the codebase.

### Interface segregation

Adapters expose only what the service needs. The Qdrant adapter exposes `search()` and
`insert()`, not the entire Qdrant client API.

```python
# backend/src/adapters/qdrant_adapter.py
class QdrantAdapter:
    def __init__(self, host: str, port: int, collection: str):
        self._client = QdrantClient(host=host, port=port)
        self._collection = collection

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        role: str,
        limit: int,
    ) -> list[ChunkResult]:
        # metadata filter, hybrid query, return typed results
        ...
```

### Open/Closed — extend via adapters, don't modify services

Swapping Groq for Claude, or Qdrant for pgvector, is a new adapter file — the service
code doesn't change because it depends on the interface, not the implementation.

### Async-safe model calls

FastAPI is async, but local models (embedding, sparse encoder, reranker) are synchronous
CPU-bound calls. Calling them directly in an async service blocks the event loop and
serializes all requests.

Wrap every local-model invocation in `asyncio.to_thread()`:

```python
import asyncio

# WRONG — blocks the event loop
async def search(self, query: str, role: str) -> list[ChunkResult]:
    dense = self.embedder.embed(query)        # CPU-bound, blocking
    sparse = self.sparse.embed(query)          # CPU-bound, blocking
    return await self.qdrant.search(dense, sparse, role)

# RIGHT — offloads to thread pool
async def search(self, query: str, role: str) -> list[ChunkResult]:
    dense = await asyncio.to_thread(self.embedder.embed, query)
    sparse = await asyncio.to_thread(self.sparse.embed, query)
    return await self.qdrant.search(dense, sparse, role)
```

Apply this in retrieval_service (embedding), rerank_service (cross-encoder), and
ingestion_service (batch embedding). Groq API calls are I/O-bound and should use
native async (`httpx.AsyncClient` or the SDK's async method).

### LangGraph patterns

**Three-layer rule:** graph nodes → services → adapters. Each layer only calls the next.

```python
# WRONG — node imports vendor SDK directly
def retrieve_node(state: MediBotState) -> dict:
    client = QdrantClient("localhost", port=6333)  # vendor SDK in node!
    results = client.search(...)

# WRONG — node skips service layer
def retrieve_node(state: MediBotState) -> dict:
    results = qdrant_adapter.hybrid_search(...)  # skipping service!

# RIGHT — node calls service, service calls adapter
def retrieve_node(state: MediBotState) -> dict:
    chunks = retrieval_service.search(state["question"], state["role"])
    return {"chunks": chunks}
```

**State mutations via return values only.** Nodes return a partial dict that gets
merged into state. Never mutate state in-place or via side effects:

```python
# WRONG — mutating state directly
def generate_node(state: MediBotState) -> dict:
    state["answer"] = llm.generate(...)  # side effect!
    return state

# RIGHT — return partial update
def generate_node(state: MediBotState) -> dict:
    answer = generation_service.generate(state["question"], state["chunks"])
    return {"answer": answer}  # LangGraph merges this into state
```

**Graph testing:** test nodes in isolation by passing a fake state dict. Test the
compiled graph end-to-end for integration.

---

## 4. Error handling

### Never swallow exceptions silently

```python
# WRONG
try:
    result = qdrant.search(...)
except Exception:
    pass  # silent failure, debugging nightmare

# RIGHT
try:
    result = qdrant.search(...)
except QdrantException as e:
    logger.error(f"request_id={req_id} qdrant_error={e}")
    raise RetrievalError(f"Vector search failed: {e}") from e
```

### Use custom exceptions, not bare Exception

```python
# backend/src/utils/exceptions.py
class MediBotError(Exception):
    """Base exception for all MediBot errors."""

class RetrievalError(MediBotError):
    """Raised when vector search fails."""

class RBACError(MediBotError):
    """Raised when a role-access check fails."""

class SQLGenerationError(MediBotError):
    """Raised when SQL generation or execution fails."""

class RoutingError(MediBotError):
    """Raised when query routing fails."""
```

### FastAPI exception handlers

Map custom exceptions to HTTP responses in one place:

```python
# backend/src/api/app.py
@app.exception_handler(RBACError)
async def rbac_handler(request, exc):
    return JSONResponse(status_code=403, content={"detail": str(exc)})

@app.exception_handler(RetrievalError)
async def retrieval_handler(request, exc):
    return JSONResponse(status_code=502, content={"detail": "Search unavailable"})
```

### Retry with backoff for external APIs

Groq's free tier has aggressive rate limits (~30 RPM). The LLM adapter must retry on
rate-limit errors with exponential backoff, not fail immediately.

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# WRONG — single attempt, 429 = crash
async def generate(self, prompt: str) -> str:
    return await self._client.chat(prompt)

# RIGHT — retry with backoff
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type(RateLimitError),
)
async def generate(self, prompt: str) -> str:
    return await self._client.chat(prompt)
```

This applies to every LLM call: router, generation, SQL generation, and narration.
Log each retry with the request_id for debugging.

---

## 5. Function and module structure

### Functions are small and do one thing

```python
# WRONG — 80-line function that retrieves, reranks, generates, and formats
def handle_chat(question, role):
    # ... everything in one function

# RIGHT — compose small functions
def handle_chat(question: str, role: str) -> ChatResponse:
    route = router_service.classify(question)
    if route == "sql_rag":
        return sql_rag_service.answer(question, role)
    chunks = retrieval_service.search(question, role)
    reranked = rerank_service.rerank(question, chunks)
    answer = generation_service.generate(question, reranked)
    return format_response(answer, reranked, route, role)
```

### Imports are explicit

```python
# WRONG
from src.services import *

# RIGHT
from src.services.retrieval_service import RetrievalService
from src.services.rerank_service import RerankService
```

### No circular imports

Services import adapters. Adapters import models. Models import nothing from src.
If two services need to call each other, extract the shared logic into a third.

---

## 6. Naming conventions

| Thing | Convention | Example |
|---|---|---|
| Files | snake_case | `retrieval_service.py` |
| Classes | PascalCase | `RetrievalService` |
| Functions | snake_case | `hybrid_search()` |
| Constants | UPPER_SNAKE | `SQL_RAG_ROLES` |
| Private | leading underscore | `_client`, `_build_filter()` |
| Config fields | snake_case matching env var | `groq_api_key` ↔ `GROQ_API_KEY` |
| Type aliases | PascalCase | `Role`, `Collection` |

---

## 7. Docstrings

Every public function and class gets a docstring. Keep it short — what it does, what it
takes, what it returns. Not a novel.

```python
def hybrid_search(self, query: str, role: str) -> list[ChunkResult]:
    """Search Qdrant with dense+sparse hybrid, filtered by role.

    Args:
        query: user's question text.
        role: authenticated user role for RBAC filter.

    Returns:
        Top-k chunks with scores, sorted by fused rank.

    Raises:
        RetrievalError: if Qdrant is unreachable.
        RBACError: if role is not recognized.
    """
```

---

## 8. Testing conventions

- Unit tests in `tests/` — mock adapters, test service logic.
- Eval tests in `backend/evals/` — real adapters, test system behavior against golden set.
- Test files mirror source: `backend/src/services/retrieval_service.py` →
  `tests/test_retrieval_service.py`.
- Use pytest fixtures for adapter injection, not setup/teardown methods.
- Test names describe the behavior: `test_nurse_cannot_retrieve_billing_chunks`,
  not `test_rbac_1`.
