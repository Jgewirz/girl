"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
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

    # Feature Toggles
    enable_stylist: bool = Field(default=True, description="Enable AI Stylist features")
    enable_fitness: bool = Field(default=False, description="Enable fitness discovery")
    enable_travel: bool = Field(default=False, description="Enable travel planning")

    @property
    def enabled_features(self) -> list[str]:
        """Get list of enabled feature names."""
        features = []
        if self.enable_stylist:
            features.append("stylist")
        if self.enable_fitness:
            features.append("fitness")
        if self.enable_travel:
            features.append("travel")
        return features

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

    # External APIs - Fitness
    mindbody_api_key: SecretStr | None = Field(default=None, description="Mindbody API key")
    mindbody_site_id: str = "-99"  # -99 is sandbox
    google_places_api_key: SecretStr | None = Field(default=None, description="Google Places API key")

    # External APIs - Travel
    amadeus_client_id: SecretStr | None = Field(default=None, description="Amadeus API client ID")
    amadeus_client_secret: SecretStr | None = Field(default=None, description="Amadeus API client secret")
    amadeus_environment: Literal["test", "production"] = "test"
    hotelbeds_api_key: SecretStr | None = Field(default=None, description="Hotelbeds API key")
    hotelbeds_secret: SecretStr | None = Field(default=None, description="Hotelbeds API secret")

    # Monitoring - LangSmith
    langsmith_api_key: SecretStr | None = Field(default=None, description="LangSmith API key")
    langsmith_project: str = "girlbot"
    langchain_tracing_v2: bool = False

    # Cache options
    use_memory_cache: bool = Field(
        default=False,
        description="Use in-memory cache instead of Redis (single-user mode)"
    )

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def has_google_places(self) -> bool:
        """Check if Google Places API is configured."""
        return self.google_places_api_key is not None

    @property
    def has_amadeus(self) -> bool:
        """Check if Amadeus API is configured."""
        return self.amadeus_client_id is not None and self.amadeus_client_secret is not None

    @property
    def has_hotelbeds(self) -> bool:
        """Check if Hotelbeds API is configured."""
        return self.hotelbeds_api_key is not None and self.hotelbeds_secret is not None

    @property
    def can_use_fitness(self) -> bool:
        """Check if fitness feature can be used (enabled and configured)."""
        return self.enable_fitness and self.has_google_places

    @property
    def can_use_travel(self) -> bool:
        """Check if travel feature can be used (enabled and configured)."""
        return self.enable_travel and self.has_amadeus

    # Resilience settings
    resilience_circuit_fail_max: int = Field(
        default=5, description="Number of failures before circuit opens"
    )
    resilience_circuit_reset_timeout: int = Field(
        default=30, description="Seconds before circuit attempts to close"
    )
    resilience_retry_attempts: int = Field(
        default=3, description="Number of retry attempts for failed calls"
    )
    resilience_openai_rate_limit: int = Field(
        default=50, description="OpenAI requests per minute"
    )
    resilience_places_rate_limit: int = Field(
        default=10, description="Google Places requests per second"
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
