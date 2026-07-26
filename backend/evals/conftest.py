"""Shared pytest fixtures for the eval suite.

Eval tests use REAL adapters against the live stack (Qdrant, SQLite) — unlike unit
tests, which mock them. Fixtures here wire the real dependencies once per session so
the corpus is scanned a single time, not per test.

All heavy adapter imports (sentence_transformers, sklearn, deepeval) are LAZY — inside
fixture bodies, not at module level. This means:
  - test_safety.py (pure function, no adapters) loads in microseconds with zero ML deps.
  - test_sql.py loads only the SQL adapter chain, not the embedding models.
  - test_faithfulness.py loads the full stack including the DeepEval judge.

This isolates the numpy/sklearn binary-incompatibility crash that occurs when
sentence_transformers is imported eagerly on the system-level Anaconda Python.
"""

import pytest

from evals.eval_config import JUDGE_MODEL


@pytest.fixture(scope="session")
def qdrant():
    """The real Qdrant adapter, pointed at the running collection (settings-driven)."""
    from src.adapters.qdrant_adapter import QdrantAdapter
    return QdrantAdapter()


@pytest.fixture(scope="session")
def embedder():
    """Real dense embedding adapter (BGE). Loaded once — the model is expensive."""
    from src.adapters.embedding_adapter import EmbeddingAdapter
    return EmbeddingAdapter()


@pytest.fixture(scope="session")
def sparse():
    """Real sparse (BM25) adapter, loaded once per session."""
    from src.adapters.sparse_adapter import SparseAdapter
    return SparseAdapter()


@pytest.fixture(scope="session")
def reranker():
    """Real cross-encoder reranker adapter, loaded once per session (Track 1)."""
    from src.adapters.reranker_adapter import RerankerAdapter
    return RerankerAdapter()


@pytest.fixture(scope="session")
def retrieval_service(qdrant, embedder, sparse):
    """The real RetrievalService wired with live adapters — the RBAC pre-filter under test.

    Session-scoped and sharing the embedder/sparse fixtures so the models load once.
    Same DI wiring as graph/builder.py — no mocks here, this is an eval, not a unit test.
    """
    from src.services.retrieval_service import RetrievalService
    return RetrievalService(qdrant=qdrant, embedder=embedder, sparse=sparse)


@pytest.fixture(scope="session")
def router_service():
    """The real RouterService wired with the live Groq LLM adapter (Track 5).

    Session-scoped: one adapter, reused across all routing cases. Same DI wiring as
    graph/builder.py. This is an LLM-as-classifier under test — real API calls.
    """
    from src.adapters.llm_adapter import LLMAdapter
    from src.services.router_service import RouterService
    return RouterService(llm=LLMAdapter())


@pytest.fixture(scope="session")
def intent_service():
    """Real IntentService wired with the live Groq LLM adapter (Track 12).

    Session-scoped: one adapter reused across all intent cases. Same DI wiring as
    graph/builder.py. First-turn cases resolve deterministically (no API call); cases
    with history exercise the LLM adjudication path.
    """
    from src.adapters.llm_adapter import LLMAdapter
    from src.services.intent_service import IntentService
    return IntentService(llm=LLMAdapter())


@pytest.fixture(scope="session")
def rerank_service(reranker):
    """Real RerankService over the cross-encoder adapter (Track 3 pipeline)."""
    from src.services.rerank_service import RerankService
    return RerankService(reranker=reranker)


@pytest.fixture(scope="session")
def generation_service():
    """Real GenerationService wired with the Groq generator model (Track 3 pipeline)."""
    from src.adapters.llm_adapter import LLMAdapter
    from src.services.generation_service import GenerationService
    return GenerationService(llm=LLMAdapter())


@pytest.fixture(scope="session")
def judge():
    """The DeepEval judge — a Groq model DIFFERENT from the generator (Track 3).

    Lazy import: deepeval imports langchain.schema which is missing from some
    environments. Only loaded when a test actually requests this fixture.
    """
    from evals.groq_judge import GroqJudge
    return GroqJudge(JUDGE_MODEL)


@pytest.fixture(scope="session")
def sql():
    """Real SQLite adapter against mediassist.db — read-only, query_only pragma enforced."""
    from src.adapters.sql_adapter import SQLAdapter
    return SQLAdapter()


@pytest.fixture(scope="session")
def sql_rag_service(sql):
    """Real SQLRAGService wired with live Groq + SQLite — the SQL correctness path under test.

    Session-scoped: one LLM adapter reused across all SQL cases to avoid per-test
    client construction overhead. Same DI wiring as graph/builder.py.
    """
    from src.adapters.llm_adapter import LLMAdapter
    from src.services.sql_rag_service import SQLRAGService
    return SQLRAGService(llm=LLMAdapter(), sql=sql)


@pytest.fixture(scope="session")
def all_payloads(qdrant):
    """Every stored chunk payload, raw and unvalidated — scanned once per session.

    Raw dicts (not ChunkResult) so a citation-less chunk is COUNTED, not raised on.
    """
    return qdrant.scroll_all_payloads()
