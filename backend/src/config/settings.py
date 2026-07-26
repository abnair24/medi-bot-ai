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
    # Relevance floor on the cross-encoder logit — a deterministic BACKSTOP under the
    # LLM no-relevant sentinel, not a replacement for it. Cross-encoder logits for this
    # corpus run roughly [-11, +11]; a genuinely correct chunk can still score around -6,
    # so this floor is set LOW on purpose — it only drops egregious keyword-collision
    # noise (e.g. "why no response" matching a billing "24 hrs no response" SLA row).
    # If EVERY reranked chunk falls below it, generation refuses instead of stitching junk.
    rerank_relevance_floor: float = -9.0

    # --- Retrieval ---
    retrieval_top_k: int = 10
    hyde_enabled: bool = False  # HyDE: embed hypothetical answer instead of raw query (dense path only)
    # RBAC access-miss probe: a restricted collection's best chunk must beat the user's best
    # accessible chunk by this margin (on the cross-encoder's logit scale — units, not 0-1)
    # to count as "the answer is behind RBAC" (→ denial) vs "topic is absent" (→ no-content).
    rbac_probe_margin: float = 1.0

    # --- SQL RAG ---
    # Paths are relative to CWD (repo root locally, /app in Docker). Data now lives
    # under backend/ingestion/data/ (moved with the ingestion service).
    sqlite_path: str = "backend/ingestion/data/mediassist.db"
    sql_hitl_enabled: bool = False          # enable human-in-the-loop SQL approval
    hitl_proposal_ttl_seconds: int = 300    # proposed SQL expires after 5 minutes

    # --- LangGraph checkpointer ---
    checkpoint_db_path: str = "backend/ingestion/data/checkpoints.db"
    # Per-user chat-session index (list past chats after re-login). Conversation CONTENT
    # lives in the checkpointer; this store holds only session_id → user + timestamps.
    sessions_db_path: str = "backend/ingestion/data/sessions.db"

    # --- Multi-turn memory ---
    # Sliding window: how many prior turns (each = user + assistant) the query rewriter
    # sees. Bounds context/cost and stops stale topics from lingering across a long chat.
    conversation_window_turns: int = 3
    # Hard cap on turns per chat session — after this many completed turns the user must
    # start a new chat. Distinct from the rewriter window above (which only limits context).
    max_turns_per_session: int = 5
    # Intent gate: when a follow-up can't be pinned to ONE prior topic, pause and ask the
    # user which they meant (HITL clarify) instead of guessing. When False, an ambiguous
    # follow-up is refused with a "please name the topic" message rather than interrupting.
    intent_clarify_enabled: bool = True

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
