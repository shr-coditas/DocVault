from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "DocVault"
    log_level: str = "INFO"
    log_json: bool = False

    database_url: str = "postgresql+asyncpg://docvault:docvault@localhost:5432/docvault"
    database_echo: bool = False

    jwt_secret: str = "dev-only-secret-not-for-production"
    access_token_ttl_minutes: int = 60
    refresh_token_ttl_days: int = 7

    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "docvault"
    s3_secret_key: str = "docvault123"
    s3_bucket: str = "docvault"
    s3_region: str = "us-east-1"

    max_upload_size_bytes: int = 25 * 1024 * 1024  # 25 MiB

    allowed_upload_extensions: list[str] = [".pdf", ".txt", ".md", ".csv", ".docx"]

    cors_origins: list[str] = []

    # -- document intelligence --------------------------------------------
    # Chunking is budgeted with the embedding model's own tokenizer. Boundaries
    # remain source-exact because provenance stores character spans separately.
    chunk_target_tokens: int = 300
    chunk_max_tokens: int = 400  # safety below bge-small's 512-token ceiling
    chunk_overlap_tokens: int = 50
    # A short tail is merged when possible and otherwise kept; never discarded.
    chunk_min_tokens: int = 25
    chunk_local_overlap_tokens: int = 32

    # Embedding. Local ONNX via fastembed: no API key, so indexing and the whole
    # test suite run offline. `embedding_dimensions` MUST match the vector(N)
    # column in document_chunks - FastEmbedEmbedder refuses to start otherwise.
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    embedding_batch_size: int = 32
    # ONNX Runtime otherwise spawns one thread per core, which oversubscribes a
    # small API container. The indexing script can raise this via the environment.
    embedding_threads: int = 1
    fastembed_cache_dir: str | None = None
    # bge is asymmetric: queries want this prefix, passages must not have it.
    # Applied inside the embedder so no caller can get it wrong.
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "

    # Indexing script
    index_batch_limit: int = 50
    # a document that has failed this many times is skipped until --retry-failed
    max_index_attempts: int = 3
    index_lease_minutes: int = 15
    index_node_batch_size: int = 500
    index_chunk_write_batch_size: int = 200
    index_artifact_retention_days: int = 7

    # Retrieval. Both are per-request query parameters; these are only the
    # defaults, because tuning top-K and the score threshold against a real
    # corpus is the point of exposing them at all.
    search_default_limit: int = 10
    # Cosine similarity in [-1, 1] (1 = identical). None deliberately means no
    # unmeasured global floor; callers can set one and the evaluation corpus will
    # eventually justify a profile-specific default.
    search_semantic_min_score: float | None = None
    # pgvector iterative scan: an HNSW scan that keeps going when the ACL
    # predicate rejects rows, instead of returning fewer than k. "strict_order"
    # preserves exact distance ordering; "relaxed_order" is faster but may
    # reorder; "off" is pre-0.8 behaviour. Nothing else is accepted.
    search_iterative_scan: str = "strict_order"
    search_default_mode: str = "hybrid"
    search_candidate_limit: int = 40
    search_fusion_limit: int = 50
    search_rerank_limit: int = 20
    search_rrf_k: int = 60
    search_per_document_limit: int = 4
    search_source_overlap_threshold: float = 0.8
    reranker_model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # -- query guardrails --------------------------------------------------
    # A ceiling on the question itself, well below the prompt budget. Long input
    # is where instruction-smuggling hides, and it is also the cheapest thing to
    # refuse - this check runs before any embedding or LLM call is spent.
    guardrail_max_query_chars: int = 2000
    # Zero-width and control characters carry no meaning a user typed on purpose,
    # and are a documented way to hide instructions inside apparently innocent
    # text. Off only if a legitimate corpus turns out to need them.
    guardrail_reject_hidden_characters: bool = True

    # -- answer generation -------------------------------------------------
    # Provider and model are both env vars because every LLM call goes through
    # LangChain's `init_chat_model`: swapping Groq for Claude is a config change,
    # not a code change. Groq is the default - free tier, fast, no card.
    llm_provider: str = "google_genai"  # "groq"
    llm_model: str = "gemini-3.1-flash-lite"  # "llama-3.3-70b-versatile"
    # Near-zero: this is extractive question answering over supplied sources, so
    # invention is the failure mode and variety buys nothing. Not exactly 0 -
    # some providers treat 0 as "unset" rather than "greedy".
    llm_temperature: float = 0.1
    llm_max_output_tokens: int = 10000
    llm_timeout_seconds: float = 30.0

    answer_context_token_budget: int = 3000
    answer_max_sources: int = 8
    llm_context_window_tokens: int = 32768
    llm_reserved_output_tokens: int = 2048
    groq_api_key: str | None = None

    google_studio_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
