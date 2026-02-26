"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: str = "INFO"

    # Telegram
    telegram_bot_token: SecretStr = Field(..., description="Telegram bot token from @BotFather")
    telegram_payment_provider_token: SecretStr | None = Field(
        default=None, description="Stripe provider token for payments"
    )

    # LLM - OpenAI
    openai_api_key: SecretStr | None = Field(default=None, description="OpenAI API key")
    openai_model_primary: str = "gpt-4o-mini"
    openai_model_complex: str = "gpt-4o"

    # LLM - Anthropic (optional, for complex tasks)
    anthropic_api_key: SecretStr | None = Field(default=None, description="Anthropic API key")
    anthropic_model: str = "claude-sonnet-4-20250514"

    # Database - Supabase (optional until persistence is needed)
    supabase_url: str | None = Field(default=None, description="Supabase project URL")
    supabase_anon_key: SecretStr | None = Field(default=None, description="Supabase anon key")
    supabase_service_key: SecretStr | None = Field(default=None, description="Supabase service key")
    database_url: SecretStr | None = Field(default=None, description="Direct PostgreSQL URL")

    @property
    def has_openai(self) -> bool:
        return self.openai_api_key is not None

    @property
    def has_database(self) -> bool:
        return self.supabase_url is not None and self.supabase_service_key is not None

    # Cache - Redis
    redis_url: SecretStr = Field(default=SecretStr("redis://localhost:6379"), description="Redis URL")

    # External APIs
    mindbody_api_key: SecretStr | None = Field(default=None, description="Mindbody API key")
    mindbody_site_id: str = "-99"  # -99 is sandbox
    google_places_api_key: SecretStr | None = Field(default=None, description="Google Places API key")

    # Monitoring - LangSmith
    langsmith_api_key: SecretStr | None = Field(default=None, description="LangSmith API key")
    langsmith_project: str = "girlbot"
    langchain_tracing_v2: bool = False

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
