"""Pytest configuration and fixtures."""

import os
import pytest

# Set test environment variables before importing settings
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test_token_12345")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_ANON_KEY", "test_anon_key")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test_service_key")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DEBUG", "true")


@pytest.fixture
def user_context():
    """Create a test user context."""
    from src.agent.state import UserContext

    return UserContext(
        telegram_id=123456789,
        username="testuser",
        first_name="Test",
        location="San Francisco",
        fitness_goals=["lose weight", "build strength"],
    )


@pytest.fixture
def agent_state(user_context):
    """Create a test agent state."""
    from src.agent.state import AgentState
    from langchain_core.messages import HumanMessage

    return AgentState(
        messages=[HumanMessage(content="Hello!")],
        user=user_context,
    )
