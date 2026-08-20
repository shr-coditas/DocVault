from functools import lru_cache

from pydantic import Field, model_validator
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
    access_token_ttl_minutes: int = 1440
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

    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    embedding_batch_size: int = 32

    embedding_threads: int = 1
    fastembed_cache_dir: str | None = None

    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "

    index_batch_limit: int = 50

    max_index_attempts: int = 3
    index_lease_minutes: int = 15
    index_node_batch_size: int = 500
    index_chunk_write_batch_size: int = 200
    index_artifact_retention_days: int = 7

    search_default_limit: int = 10

    search_semantic_min_score: float | None = None

    search_iterative_scan: str = "strict_order"
    search_default_mode: str = "hybrid"
    search_candidate_limit: int = 40
    search_fusion_limit: int = 50
    search_rerank_limit: int = 20
    search_rrf_k: int = 60
    search_per_document_limit: int = 4
    search_source_overlap_threshold: float = 0.8
    reranker_model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    guardrail_max_query_chars: int = 2000
    guardrail_reject_hidden_characters: bool = True

    llm_provider: str  # = "google_genai"  # "groq"
    llm_model: str  # = "gemini-3.1-flash-lite"  # "llama-3.3-70b-versatile"

    llm_temperature: float = 0.1
    llm_max_output_tokens: int = 10000
    llm_timeout_seconds: float = 30.0

    resolver_provider: str | None = None
    resolver_model: str | None = None
    resolver_timeout_seconds: float = 20.0
    resolver_history_max_turns: int = 6
    resolver_history_token_budget: int = 1500

    # -- agent ----------------------------------------------------------
    # The supervised graph. Off means the linear pipeline answers instead:
    # guardrail, classify, one search, one generation.
    agent_enabled: bool = True
    # Falls back to the answering model when unset. Kept separable because the
    # supervisor wants a fast model with reliable structured output, and the
    # answerer wants a good writer - not always the same choice.
    agent_model_provider: str | None = None
    agent_model: str | None = None
    agent_max_output_tokens: int = Field(default=1000, ge=128, le=4096)
    agent_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    # Rounds, not queries: one decision to search costs one, whether it named
    # one wording or three. The supervisor is clamped to these; it can ask for
    # another pass and never for a larger allowance.
    agent_max_searches: int = Field(default=3, ge=1, le=5)
    agent_max_drafts: int = Field(default=2, ge=1, le=3)
    # A backstop, not the bound - see `validate_agent_step_budget`.
    agent_max_steps: int = Field(default=16, ge=6, le=32)
    agent_output_prompt_overlap_chars: int = Field(default=80, ge=40, le=500)
    agent_output_source_overlap_chars: int = Field(default=300, ge=100, le=2000)

    answer_context_token_budget: int = 3000
    answer_max_sources: int = 8
    llm_context_window_tokens: int = 32768
    llm_reserved_output_tokens: int = 2048
    groq_api_key: str | None = None

    google_studio_api_key: str | None = None

    @model_validator(mode="after")
    def validate_agent_step_budget(self) -> "Settings":
        """Keep the graph's recursion limit ahead of the budgets it has to serve.

        Every node execution is one LangGraph superstep. The longest legal path
        is guard and classify, then a supervise/retrieve pair per search, then
        a supervise/write pair per draft, then respond - so a raised budget
        with an unchanged step limit would surface as a GraphRecursionError on a
        live request. Refusing at startup makes it a configuration error, which
        is the kind of failure someone can act on.
        """
        required = 3 + 2 * self.agent_max_searches + 2 * self.agent_max_drafts
        if self.agent_max_steps < required:
            raise ValueError(
                f"agent_max_steps must be at least {required} for "
                f"{self.agent_max_searches} searches and {self.agent_max_drafts} drafts"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
