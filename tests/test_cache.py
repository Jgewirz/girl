"""Tests for Redis cache and session management."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cache.session import ConversationSession, SessionManager


class TestConversationSession:
    """Tests for ConversationSession dataclass."""

    def test_create_session(self):
        """Test creating a new session."""
        session = ConversationSession(telegram_id=12345)

        assert session.telegram_id == 12345
        assert session.messages == []
        assert session.first_name is None
        assert session.location is None
        assert session.fitness_goals == []
        assert session.message_count == 0

    def test_add_message(self):
        """Test adding messages to session."""
        session = ConversationSession(telegram_id=12345)

        session.add_message("human", "Hello!")
        session.add_message("assistant", "Hi there!")

        assert len(session.messages) == 2
        assert session.messages[0]["role"] == "human"
        assert session.messages[0]["content"] == "Hello!"
        assert session.messages[1]["role"] == "assistant"
        assert session.message_count == 2

    def test_message_limit(self):
        """Test that messages are limited to 20."""
        session = ConversationSession(telegram_id=12345)

        # Add 25 messages
        for i in range(25):
            session.add_message("human", f"Message {i}")

        # Should only keep last 20
        assert len(session.messages) == 20
        assert session.messages[0]["content"] == "Message 5"
        assert session.messages[-1]["content"] == "Message 24"

    def test_to_langchain_messages(self):
        """Test conversion to LangChain message format."""
        session = ConversationSession(telegram_id=12345)
        session.add_message("human", "Hello")
        session.add_message("assistant", "Hi!")

        lc_messages = session.to_langchain_messages()

        assert len(lc_messages) == 2
        assert lc_messages[0].content == "Hello"
        assert lc_messages[1].content == "Hi!"

    def test_json_serialization(self):
        """Test JSON serialization round-trip."""
        session = ConversationSession(
            telegram_id=12345,
            first_name="Test",
            location="San Francisco",
            fitness_goals=["strength", "flexibility"],
        )
        session.add_message("human", "Hello")

        # Serialize and deserialize
        json_str = session.to_json()
        restored = ConversationSession.from_json(json_str)

        assert restored.telegram_id == 12345
        assert restored.first_name == "Test"
        assert restored.location == "San Francisco"
        assert restored.fitness_goals == ["strength", "flexibility"]
        assert len(restored.messages) == 1


class TestSessionManager:
    """Tests for SessionManager."""

    @pytest.fixture
    def mock_redis_client(self):
        """Create a mock Redis client."""
        client = MagicMock()
        client.get = AsyncMock(return_value=None)
        client.set = AsyncMock(return_value=True)
        client.delete = AsyncMock(return_value=True)
        client.incr = AsyncMock(return_value=1)
        client.expire = AsyncMock(return_value=True)
        client.close = AsyncMock()
        return client

    @pytest.fixture
    def session_manager(self, mock_redis_client):
        """Create a SessionManager with mock Redis."""
        return SessionManager(mock_redis_client)

    @pytest.mark.asyncio
    async def test_get_session_new(self, session_manager, mock_redis_client):
        """Test getting a session that doesn't exist creates new one."""
        mock_redis_client.get = AsyncMock(return_value=None)

        session = await session_manager.get_session(12345)

        assert session.telegram_id == 12345
        assert session.messages == []

    @pytest.mark.asyncio
    async def test_get_session_existing(self, session_manager, mock_redis_client):
        """Test getting an existing session."""
        existing_session = ConversationSession(
            telegram_id=12345,
            first_name="Test",
            location="NYC",
        )
        mock_redis_client.get = AsyncMock(return_value=existing_session.to_json())

        session = await session_manager.get_session(12345)

        assert session.telegram_id == 12345
        assert session.first_name == "Test"
        assert session.location == "NYC"

    @pytest.mark.asyncio
    async def test_save_session(self, session_manager, mock_redis_client):
        """Test saving a session."""
        session = ConversationSession(telegram_id=12345)
        session.add_message("human", "Hello")

        result = await session_manager.save_session(session)

        assert result is True
        mock_redis_client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_session(self, session_manager, mock_redis_client):
        """Test deleting a session."""
        result = await session_manager.delete_session(12345)

        assert result is True
        mock_redis_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_user_info(self, session_manager, mock_redis_client):
        """Test updating user info."""
        mock_redis_client.get = AsyncMock(return_value=None)

        session = await session_manager.update_user_info(
            telegram_id=12345,
            first_name="Alice",
            location="LA",
        )

        assert session.first_name == "Alice"
        assert session.location == "LA"

    @pytest.mark.asyncio
    async def test_rate_limit_allowed(self, session_manager, mock_redis_client):
        """Test rate limiting when under limit."""
        mock_redis_client.incr = AsyncMock(return_value=5)

        is_allowed, count = await session_manager.check_rate_limit(12345, max_requests=30)

        assert is_allowed is True
        assert count == 5

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, session_manager, mock_redis_client):
        """Test rate limiting when over limit."""
        mock_redis_client.incr = AsyncMock(return_value=31)

        is_allowed, count = await session_manager.check_rate_limit(12345, max_requests=30)

        assert is_allowed is False
        assert count == 31

    @pytest.mark.asyncio
    async def test_rate_limit_first_request(self, session_manager, mock_redis_client):
        """Test that first request sets expiry."""
        mock_redis_client.incr = AsyncMock(return_value=1)

        await session_manager.check_rate_limit(12345)

        mock_redis_client.expire.assert_called_once()
