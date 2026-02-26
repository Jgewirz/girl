"""Tests for configuration module."""

import os
import pytest


def test_settings_loads():
    """Test that settings can be loaded."""
    from src.config import settings

    assert settings is not None
    assert settings.environment in ["development", "staging", "production"]


def test_settings_has_required_fields():
    """Test that settings has all required fields."""
    from src.config import settings

    # These should exist (may be test values)
    assert hasattr(settings, "telegram_bot_token")
    assert hasattr(settings, "openai_api_key")
    assert hasattr(settings, "supabase_url")


def test_is_production_property():
    """Test the is_production property."""
    from src.config import settings

    # In test env, should not be production
    assert settings.is_production == (settings.environment == "production")


def test_llm_models_configured():
    """Test that LLM models are configured."""
    from src.config import settings

    assert settings.openai_model_primary == "gpt-4o-mini"
    assert settings.openai_model_complex == "gpt-4o"
