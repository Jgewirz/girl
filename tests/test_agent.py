"""Tests for the LangGraph agent."""

import pytest
from langchain_core.messages import HumanMessage, AIMessage


def test_user_context_creation(user_context):
    """Test UserContext dataclass."""
    assert user_context.telegram_id == 123456789
    assert user_context.username == "testuser"
    assert user_context.first_name == "Test"
    assert user_context.location == "San Francisco"
    assert "lose weight" in user_context.fitness_goals


def test_agent_state_creation(agent_state):
    """Test AgentState dataclass."""
    assert len(agent_state.messages) == 1
    assert isinstance(agent_state.messages[0], HumanMessage)
    assert agent_state.user is not None
    assert agent_state.next_action == "continue"


def test_agent_state_defaults():
    """Test AgentState default values."""
    from src.agent.state import AgentState

    state = AgentState(messages=[])

    assert state.messages == []
    assert state.user is None
    assert state.current_intent is None
    assert state.selected_tools == []
    assert state.next_action == "continue"
    assert state.error is None


def test_tool_registry_creation():
    """Test ToolRegistry can be created."""
    from src.agent.tools import ToolRegistry

    registry = ToolRegistry()
    assert len(registry) == 0
    assert registry.list_tools() == []


def test_tool_registry_register():
    """Test registering a tool."""
    from src.agent.tools import ToolRegistry

    registry = ToolRegistry()

    def dummy_tool(query: str) -> str:
        return f"Result: {query}"

    registry.register(
        name="dummy_tool",
        description="A dummy tool for testing",
        func=dummy_tool,
    )

    assert len(registry) == 1
    assert "dummy_tool" in registry.list_tools()

    config = registry.get("dummy_tool")
    assert config is not None
    assert config.name == "dummy_tool"
    assert config.description == "A dummy tool for testing"


def test_tool_registry_to_langchain():
    """Test converting registry to LangChain tools."""
    from src.agent.tools import ToolRegistry

    registry = ToolRegistry()

    def search_classes(location: str, class_type: str) -> str:
        return f"Found classes in {location}"

    registry.register(
        name="search_classes",
        description="Search for fitness classes",
        func=search_classes,
    )

    lc_tools = registry.to_langchain_tools()
    assert len(lc_tools) == 1
    assert lc_tools[0].name == "search_classes"


def test_create_agent():
    """Test agent can be created."""
    from src.agent import create_agent

    agent = create_agent()
    assert agent is not None
