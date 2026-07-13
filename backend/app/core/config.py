from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "DocVault"
    environment: Literal["local", "test", "production"] = "local"
    log_level: str = "INFO"
    log_json: bool = False

    database_url: str = "postgresql+asyncpg://docvault:docvault@localhost:5432/docvault"
    database_echo: bool = False

    jwt_secret: str = "dev-only-secret-not-for-production"
    access_token_ttl_minutes: int = 15
    refresh_token_ttl_days: int = 7

    cors_origins: list[str] = []


@lru_cache
def get_settings() -> Settings:
    return Settings()
