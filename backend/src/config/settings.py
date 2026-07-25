from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central config for MediBot. All values come from env vars or .env file.

    Fields with no default are required — startup fails immediately if missing.
    Fields with defaults can be overridden via env var (UPPER_SNAKE case).
    """

    # --- LLM (Groq) ---
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"

    # --- Vector store (Qdrant) ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "medibot"

    # --- Embedding models (local, no API cost) ---
    dense_model_name: str = "BAAI/bge-small-en-v1.5"
    sparse_model_name: str = "Qdrant/bm25"
    dense_vector_size: int = 384  # must match the output dims of dense_model_name

    # Optional HuggingFace token for model downloads (declared so a present .env key
    # does not trip extra_forbidden on host-side runs; unused for public models).
    hf_token: str | None = None

    # --- Chunking ---
    # bge-small-en-v1.5 hard max is 512 tokens; 384 leaves headroom for prepended section heading.
    # overlap preserves cross-boundary context when token-cap splits occur within long paragraphs.
    chunk_token_limit: int = 384
    chunk_overlap_tokens: int = 64
    embedding_batch_size: int = 32

    # --- Reranker (local, no API cost) ---
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 10
    reranker_final_k: int = 3
    # Two-axis router: query specificity tunes how many reranked chunks feed generation.
    # point_lookup wants a tight, precise context; exploratory wants broader context.
    # aggregate falls back to reranker_final_k (rarely hits rerank — usually the SQL path).
    reranker_final_k_point_lookup: int = 3
    reranker_final_k_exploratory: int = 5

    # --- Retrieval ---
    retrieval_top_k: int = 10
    hyde_enabled: bool = False  # HyDE: embed hypothetical answer instead of raw query (dense path only)
    # RBAC access-miss probe: a restricted collection must outscore the user's best hit by
    # this margin to count as "the answer is behind RBAC" (→ denial) vs "topic is absent".
    rbac_probe_margin: float = 0.05

    # --- SQL RAG ---
    # Paths are relative to CWD (repo root locally, /app in Docker). Data now lives
    # under backend/ingestion/data/ (moved with the ingestion service).
    sqlite_path: str = "backend/ingestion/data/mediassist.db"
    sql_hitl_enabled: bool = False          # enable human-in-the-loop SQL approval
    hitl_proposal_ttl_seconds: int = 300    # proposed SQL expires after 5 minutes

    # --- LangGraph checkpointer ---
    checkpoint_db_path: str = "backend/ingestion/data/checkpoints.db"

    # --- Multi-turn memory ---
    # Sliding window: how many prior turns (each = user + assistant) the query rewriter
    # sees. Bounds context/cost and stops stale topics from lingering across a long chat.
    conversation_window_turns: int = 3

    # --- Auth ---
    jwt_secret: str = "change-this-in-production"
    jwt_expiry_hours: int = 24

    # --- Logging ---
    log_level: str = "INFO"

    # --- Evaluation dashboard (admin-only, dev/demo affordance) ---
    # When False, the /evals/* endpoints are not mounted at all (prod default).
    # Requires the eval deps (pytest, deepeval) to be present in the image.
    evals_enabled: bool = False
    eval_results_dir: str = "backend/evals/results"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Module-level singleton. Import this object, never construct Settings() in business logic.
settings = Settings()
