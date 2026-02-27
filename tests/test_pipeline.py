"""Pipeline integration tests for the full message processing flow."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.graph import create_agent, process_message, SafeToolNode
from src.agent.state import AgentState, UserContext
from src.agent.tools import register_all_tools, tool_registry
from src.agent.fallbacks import FallbackType, get_fallback


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear and re-register tools for each test."""
    tool_registry._tools.clear()
    register_all_tools()
    yield
    tool_registry._tools.clear()


@pytest.fixture
def mock_settings():
    """Mock settings with test values."""
    with patch("src.agent.graph.settings") as mock_s:
        mock_s.has_openai = True
        mock_s.openai_api_key = MagicMock()
        mock_s.openai_api_key.get_secret_value.return_value = "test-key"
        mock_s.openai_model_primary = "gpt-4o-mini"
        mock_s.openai_model_complex = "gpt-4o"
        yield mock_s


@pytest.fixture
def sample_user_context():
    """Create a sample user context."""
    return UserContext(
        telegram_id=12345,
        first_name="TestUser",
        location="San Francisco",
        fitness_goals=["lose weight"],
        preferred_workout_types=["yoga"],
    )


@pytest.fixture
def sample_agent_state(sample_user_context):
    """Create a sample agent state."""
    return AgentState(
        messages=[HumanMessage(content="Hello, how are you?")],
        user=sample_user_context,
    )


class TestAgentCreation:
    """Tests for agent creation and initialization."""

    def test_create_agent_returns_compiled_graph(self, mock_settings):
        """Test that create_agent returns a compiled state graph."""
        agent = create_agent()
        assert agent is not None
        # Should have nodes
        assert hasattr(agent, "nodes")

    def test_create_agent_with_tools_registered(self, mock_settings):
        """Test that agent is created with registered tools."""
        # Ensure tools are registered
        assert len(tool_registry) > 0

        agent = create_agent()
        assert agent is not None


class TestMessageProcessing:
    """Tests for message processing."""

    @pytest.mark.asyncio
    async def test_process_message_success(self, mock_settings, sample_agent_state):
        """Test successful message processing."""
        mock_response = AIMessage(content="Hello! How can I help you?")

        with patch("src.agent.graph.ChatOpenAI") as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm.bind_tools = MagicMock(return_value=mock_llm)
            mock_llm_class.return_value = mock_llm

            result = await process_message(sample_agent_state)

            assert "messages" in result
            assert len(result["messages"]) > 0
            assert result["next_action"] == "respond"

    @pytest.mark.asyncio
    async def test_process_message_llm_error(self, mock_settings, sample_agent_state):
        """Test that LLM errors are handled gracefully."""
        with patch("src.agent.graph.ChatOpenAI") as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(side_effect=Exception("API Error"))
            mock_llm.bind_tools = MagicMock(return_value=mock_llm)
            mock_llm_class.return_value = mock_llm

            result = await process_message(sample_agent_state)

            # Should return error response, not crash
            assert "messages" in result
            assert "error" in result
            assert result["next_action"] == "respond"

            # Response should be user-friendly
            response_content = result["messages"][0].content
            assert response_content  # Not empty
            assert "stack trace" not in response_content.lower()

    @pytest.mark.asyncio
    async def test_process_message_rate_limit_error(self, mock_settings, sample_agent_state):
        """Test that rate limit errors use proper fallback."""
        with patch("src.agent.graph.ChatOpenAI") as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(
                side_effect=Exception("429 Too Many Requests")
            )
            mock_llm.bind_tools = MagicMock(return_value=mock_llm)
            mock_llm_class.return_value = mock_llm

            result = await process_message(sample_agent_state)

            assert "error" in result
            response_content = result["messages"][0].content
            # Should mention trying again or similar
            assert any(word in response_content.lower() for word in ["moment", "again", "busy"])


class TestSafeToolNode:
    """Tests for SafeToolNode error handling."""

    @pytest.mark.asyncio
    async def test_safe_tool_node_success(self, mock_settings):
        """Test that SafeToolNode passes through successful calls."""
        # Use actual registered tools
        tools = tool_registry.to_langchain_tools()
        assert len(tools) > 0

        safe_node = SafeToolNode(tools)

        # Create state with tool call
        state = AgentState(
            messages=[
                HumanMessage(content="test"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "update_user_location", "args": {"location": "NYC", "telegram_id": 123}, "id": "call_123"}],
                ),
            ],
            user=UserContext(telegram_id=123),
        )

        # Mock the inner tool node to return success
        with patch.object(safe_node, "_tool_node") as mock_inner:
            mock_inner.ainvoke = AsyncMock(
                return_value={"messages": [ToolMessage(content="Location updated!", tool_call_id="call_123")]}
            )

            result = await safe_node(state)

            assert "messages" in result
            assert result["messages"][0].content == "Location updated!"

    @pytest.mark.asyncio
    async def test_safe_tool_node_handles_exception(self, mock_settings):
        """Test that SafeToolNode catches exceptions and returns fallback."""
        # Use actual registered tools
        tools = tool_registry.to_langchain_tools()

        safe_node = SafeToolNode(tools)

        state = AgentState(
            messages=[
                HumanMessage(content="test"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "update_user_location", "args": {"location": "NYC", "telegram_id": 123}, "id": "call_123"}],
                ),
            ],
            user=UserContext(telegram_id=123),
        )

        with patch.object(safe_node, "_tool_node") as mock_inner:
            mock_inner.ainvoke = AsyncMock(side_effect=Exception("Tool crashed!"))

            result = await safe_node(state)

            # Should not raise, should return fallback
            assert "messages" in result
            assert len(result["messages"]) > 0
            # Should be a ToolMessage with error info
            assert isinstance(result["messages"][0], ToolMessage)
            assert "error" in result["messages"][0].content.lower()


class TestFullPipeline:
    """End-to-end pipeline tests."""

    @pytest.mark.asyncio
    async def test_full_pipeline_simple_message(self, mock_settings):
        """Test full pipeline with a simple message."""
        agent = create_agent()

        initial_state = AgentState(
            messages=[HumanMessage(content="Hi there!")],
            user=UserContext(telegram_id=12345, first_name="Test"),
        )

        mock_response = AIMessage(content="Hello! How can I help you today?")

        with patch("src.agent.graph.ChatOpenAI") as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm.bind_tools = MagicMock(return_value=mock_llm)
            mock_llm_class.return_value = mock_llm

            result = await agent.ainvoke(initial_state)

            assert "messages" in result
            # Should have at least the original message plus response
            assert len(result["messages"]) >= 2

    @pytest.mark.asyncio
    async def test_pipeline_with_tool_call(self, mock_settings):
        """Test pipeline when LLM decides to call a tool."""
        agent = create_agent()

        initial_state = AgentState(
            messages=[HumanMessage(content="Find yoga studios near me")],
            user=UserContext(telegram_id=12345, location="San Francisco"),
        )

        # First call returns tool call, second call returns response
        tool_call_response = AIMessage(
            content="",
            tool_calls=[{
                "name": "search_fitness_studios",
                "args": {"query": "yoga studios", "location": "San Francisco"},
                "id": "call_123",
            }],
        )
        final_response = AIMessage(content="Here are some yoga studios I found...")

        call_count = 0

        async def mock_invoke(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tool_call_response
            return final_response

        with patch("src.agent.graph.ChatOpenAI") as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = mock_invoke
            mock_llm.bind_tools = MagicMock(return_value=mock_llm)
            mock_llm_class.return_value = mock_llm

            # Mock the places client to return empty results
            with patch("src.agent.tools.places.get_places_client") as mock_places:
                mock_places.return_value = None

                result = await agent.ainvoke(initial_state)

                assert "messages" in result


class TestErrorHandlingChain:
    """Tests for the error handling chain."""

    def test_fallback_messages_are_friendly(self):
        """Test that all fallback messages are user-friendly."""
        unfriendly_patterns = [
            "exception",
            "stack trace",
            "traceback",
            "null",
            "undefined",
            "500",
            "error code",
        ]

        test_types = [
            FallbackType.VISION_UNAVAILABLE,
            FallbackType.PLACES_UNAVAILABLE,
            FallbackType.TIMEOUT,
            FallbackType.RATE_LIMITED,
            FallbackType.UNKNOWN_ERROR,
        ]

        for ft in test_types:
            msg = get_fallback(ft)
            msg_lower = msg.lower()

            for pattern in unfriendly_patterns:
                assert pattern not in msg_lower, (
                    f"Unfriendly pattern '{pattern}' found in {ft.name}: {msg}"
                )

    def test_all_fallbacks_suggest_next_action(self):
        """Test that fallbacks suggest what to do next."""
        action_words = ["try", "please", "could", "instead", "can", "help"]

        test_types = [
            FallbackType.VISION_UNAVAILABLE,
            FallbackType.PHOTO_DOWNLOAD_FAILED,
            FallbackType.UNKNOWN_ERROR,
        ]

        for ft in test_types:
            msg = get_fallback(ft)
            has_action = any(word in msg.lower() for word in action_words)
            assert has_action, f"{ft.name} should suggest an action: {msg}"


class TestContextHandling:
    """Tests for user context handling in the pipeline."""

    @pytest.mark.asyncio
    async def test_user_context_included_in_messages(self, mock_settings):
        """Test that user context is included when processing messages."""
        user = UserContext(
            telegram_id=12345,
            first_name="Alice",
            location="New York",
            fitness_goals=["build strength"],
        )

        state = AgentState(
            messages=[HumanMessage(content="Hi")],
            user=user,
        )

        captured_messages = []

        async def capture_invoke(messages):
            captured_messages.extend(messages)
            return AIMessage(content="Hello!")

        with patch("src.agent.graph.ChatOpenAI") as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = capture_invoke
            mock_llm.bind_tools = MagicMock(return_value=mock_llm)
            mock_llm_class.return_value = mock_llm

            await process_message(state)

            # Check that user context was included
            all_content = " ".join(str(m.content) for m in captured_messages)
            assert "Alice" in all_content or "12345" in all_content
            assert "New York" in all_content


class TestResilienceIntegration:
    """Tests for resilience patterns in the pipeline."""

    @pytest.mark.asyncio
    async def test_timeout_produces_friendly_message(self, mock_settings, sample_agent_state):
        """Test that timeout errors produce friendly messages."""
        import asyncio

        with patch("src.agent.graph.ChatOpenAI") as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(
                side_effect=asyncio.TimeoutError("Request timed out")
            )
            mock_llm.bind_tools = MagicMock(return_value=mock_llm)
            mock_llm_class.return_value = mock_llm

            result = await process_message(sample_agent_state)

            assert "error" in result
            response = result["messages"][0].content
            # Should not expose technical details
            assert "asyncio" not in response.lower()
            assert "timeout" not in response.lower() or "moment" in response.lower()
