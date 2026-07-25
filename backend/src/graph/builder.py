from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from src.adapters.embedding_adapter import EmbeddingAdapter
from src.adapters.llm_adapter import LLMAdapter
from src.adapters.qdrant_adapter import QdrantAdapter
from src.adapters.reranker_adapter import RerankerAdapter
from src.adapters.sparse_adapter import SparseAdapter
from src.adapters.sql_adapter import SQLAdapter
from src.graph.edges import route_after_propose, route_after_router, should_rewrite
from src.graph.nodes import (
    make_generate_node,
    make_rerank_node,
    make_retrieve_node,
    make_rewrite_node,
    make_router_node,
    make_sql_execute_node,
    make_sql_propose_node,
)
from src.graph.state import MediBotState
from src.services.generation_service import GenerationService
from src.services.hyde_service import HyDEService
from src.services.rerank_service import RerankService
from src.services.retrieval_service import RetrievalService
from src.services.rewrite_service import RewriteService
from src.services.router_service import RouterService
from src.services.sql_rag_service import SQLRAGService


def build_graph(checkpointer: AsyncSqliteSaver):
    """Assemble the MediBot LangGraph and return a compiled graph.

    Wiring order: adapters → services → node factories → graph assembly → compile.
    This function is the single composition root — every dependency is constructed
    and injected here. No service or adapter constructs its own dependencies.

    Args:
        checkpointer: async SQLite checkpointer injected from app startup.
            Passed in (not constructed here) so the caller controls the
            connection lifecycle and can inject a test checkpointer in evals.

    Returns:
        Compiled LangGraph StateGraph ready to invoke per-request.
    """
    # --- Adapters (leaf layer — talk to external systems) ---
    llm = LLMAdapter()
    qdrant = QdrantAdapter()
    embedder = EmbeddingAdapter()
    sparse = SparseAdapter()
    reranker = RerankerAdapter()
    sql = SQLAdapter()

    # --- Services (business logic layer — call adapters, never SDKs) ---
    # HyDE is injected always; it only fires when settings.hyde_enabled is True.
    # Injecting unconditionally keeps the composition root simple — the flag, not the
    # wiring, decides activation (a no-op dependency when disabled).
    hyde_svc = HyDEService(llm=llm)
    retrieval_svc = RetrievalService(
        qdrant=qdrant, embedder=embedder, sparse=sparse, hyde=hyde_svc, reranker=reranker
    )
    rerank_svc = RerankService(reranker=reranker)
    generation_svc = GenerationService(llm=llm)
    router_svc = RouterService(llm=llm)
    rewrite_svc = RewriteService(llm=llm)
    sql_rag_svc = SQLRAGService(llm=llm, sql=sql)

    # --- Graph assembly ---
    graph = StateGraph(MediBotState)

    # Nodes
    graph.add_node("rewrite", make_rewrite_node(rewrite_svc))
    graph.add_node("router", make_router_node(router_svc))
    graph.add_node("retrieve", make_retrieve_node(retrieval_svc))
    graph.add_node("rerank", make_rerank_node(rerank_svc))
    graph.add_node("generate", make_generate_node(generation_svc))
    graph.add_node("sql_rag", make_sql_propose_node(sql_rag_svc))
    graph.add_node("sql_execute", make_sql_execute_node(sql_rag_svc))

    # Edges — START branches on history presence
    graph.add_conditional_edges(
        START,
        should_rewrite,
        {"rewrite": "rewrite", "router": "router"},
    )
    graph.add_edge("rewrite", "router")

    # Router branches on route decision
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"hybrid_rag": "retrieve", "sql_rag": "sql_rag"},
    )

    # Hybrid RAG path: retrieve → rerank → generate → END
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)

    # SQL RAG path: propose → (conditional) → execute → END
    graph.add_conditional_edges(
        "sql_rag",
        route_after_propose,
        {"sql_execute": "sql_execute", END: END},
    )
    graph.add_edge("sql_execute", END)

    return graph.compile(checkpointer=checkpointer)
