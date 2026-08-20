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
    agent_enabled: bool = True #use langgraph to answer user queries
    agent_model_provider: str | None = None
    agent_model: str | None = None
    agent_structured_max_output_tokens: int = Field(default=1000, ge=128, le=4096)
    agent_analysis_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    agent_planning_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    agent_evidence_grading_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    agent_max_retrieval_attempts: int = Field(default=2, ge=1, le=3)
    agent_max_rewrites: int = Field(default=1, ge=0, le=2)
    agent_max_subqueries: int = Field(default=3, ge=1, le=3)
    agent_max_generation_attempts: int = Field(default=2, ge=1, le=3)
    agent_max_total_steps: int = Field(default=18, ge=10, le=32)
    agent_checkpoint_retention_hours: int = Field(default=24, ge=1, le=168)
    agent_output_prompt_overlap_chars: int = Field(default=80, ge=40, le=500)
    agent_output_source_overlap_chars: int = Field(default=300, ge=100, le=2000)

    answer_context_token_budget: int = 3000
    answer_max_sources: int = 8
    llm_context_window_tokens: int = 32768
    llm_reserved_output_tokens: int = 2048
    groq_api_key: str | None = None

    google_studio_api_key: str | None = None

    @model_validator(mode="after")
    def validate_agent_attempt_limits(self) -> "Settings":
        if self.agent_max_rewrites >= self.agent_max_retrieval_attempts:
            raise ValueError("agent_max_rewrites must be lower than agent_max_retrieval_attempts")

        # `agent_max_total_steps` becomes the graph's recursion limit, and the
        # corrective-retrieval loop adds nodes per attempt. Left unchecked, a
        # raised retry budget turns into GraphRecursionError on a live request;
        # refusing at startup makes it a configuration error instead.
        attempts = min(self.agent_max_retrieval_attempts, self.agent_max_rewrites + 1)
        generations = self.agent_max_generation_attempts
        # Each node execution is one LangGraph superstep, so the worst-case path
        # is counted node by node: guard, analyze, load_context, resolve_context,
        # plan and finalize run once each; retrieve + grade run per retrieval
        # attempt with one rewrite between attempts; generate + validate run per
        # generation attempt.
        required_steps = 6 + 2 * attempts + (attempts - 1) + 2 * generations
        if self.agent_max_total_steps < required_steps:
            raise ValueError(
                f"agent_max_total_steps must be at least {required_steps} "
                f"for {attempts} retrieval attempts and {generations} generation attempts"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
