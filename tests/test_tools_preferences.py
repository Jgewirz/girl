"""Tests for user preference tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.tools.preferences import (
    update_user_location,
    update_fitness_goals,
    update_workout_preferences,
    get_user_preferences,
    register_preference_tools,
)
from src.agent.tools import tool_registry
from src.cache.session import ConversationSession


class TestUpdateUserLocation:
    """Tests for update_user_location tool."""

    @pytest.mark.asyncio
    async def test_set_new_location(self):
        """Test setting location for the first time."""
        mock_session = ConversationSession(telegram_id=12345)
        mock_session_mgr = MagicMock()
        mock_session_mgr.get_session = AsyncMock(return_value=mock_session)
        mock_session_mgr.save_session = AsyncMock(return_value=True)

        with patch(
            "src.agent.tools.preferences.get_session_manager",
            return_value=mock_session_mgr,
        ):
            result = await update_user_location("San Francisco", 12345)

        assert "San Francisco" in result
        assert "remember" in result.lower()
        assert mock_session.location == "San Francisco"
        mock_session_mgr.save_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_existing_location(self):
        """Test updating an existing location."""
        mock_session = ConversationSession(telegram_id=12345, location="New York")
        mock_session_mgr = MagicMock()
        mock_session_mgr.get_session = AsyncMock(return_value=mock_session)
        mock_session_mgr.save_session = AsyncMock(return_value=True)

        with patch(
            "src.agent.tools.preferences.get_session_manager",
            return_value=mock_session_mgr,
        ):
            result = await update_user_location("Los Angeles", 12345)

        assert "Updated" in result
        assert "New York" in result
        assert "Los Angeles" in result
        assert mock_session.location == "Los Angeles"

    @pytest.mark.asyncio
    async def test_location_update_error(self):
        """Test handling of errors during location update."""
        with patch(
            "src.agent.tools.preferences.get_session_manager",
            side_effect=Exception("Redis error"),
        ):
            result = await update_user_location("Chicago", 12345)

        assert "noted" in result.lower()
        assert "couldn't save" in result.lower()


class TestUpdateFitnessGoals:
    """Tests for update_fitness_goals tool."""

    @pytest.mark.asyncio
    async def test_add_new_goals(self):
        """Test adding fitness goals."""
        mock_session = ConversationSession(telegram_id=12345)
        mock_session_mgr = MagicMock()
        mock_session_mgr.get_session = AsyncMock(return_value=mock_session)
        mock_session_mgr.save_session = AsyncMock(return_value=True)

        with patch(
            "src.agent.tools.preferences.get_session_manager",
            return_value=mock_session_mgr,
        ):
            result = await update_fitness_goals(
                ["lose weight", "build muscle"], 12345
            )

        assert "Added" in result
        assert "lose weight" in result
        assert mock_session.fitness_goals == ["lose weight", "build muscle"]

    @pytest.mark.asyncio
    async def test_avoid_duplicate_goals(self):
        """Test that duplicate goals are not added."""
        mock_session = ConversationSession(
            telegram_id=12345, fitness_goals=["lose weight"]
        )
        mock_session_mgr = MagicMock()
        mock_session_mgr.get_session = AsyncMock(return_value=mock_session)
        mock_session_mgr.save_session = AsyncMock(return_value=True)

        with patch(
            "src.agent.tools.preferences.get_session_manager",
            return_value=mock_session_mgr,
        ):
            result = await update_fitness_goals(
                ["lose weight", "build muscle"], 12345
            )

        assert "build muscle" in result
        # Should not duplicate "lose weight"
        assert mock_session.fitness_goals.count("lose weight") == 1
        assert "build muscle" in mock_session.fitness_goals

    @pytest.mark.asyncio
    async def test_all_goals_already_exist(self):
        """Test when all goals already exist."""
        mock_session = ConversationSession(
            telegram_id=12345, fitness_goals=["lose weight", "build muscle"]
        )
        mock_session_mgr = MagicMock()
        mock_session_mgr.get_session = AsyncMock(return_value=mock_session)
        mock_session_mgr.save_session = AsyncMock(return_value=True)

        with patch(
            "src.agent.tools.preferences.get_session_manager",
            return_value=mock_session_mgr,
        ):
            result = await update_fitness_goals(["Lose Weight"], 12345)  # Different case

        assert "already have" in result.lower()


class TestUpdateWorkoutPreferences:
    """Tests for update_workout_preferences tool."""

    @pytest.mark.asyncio
    async def test_add_workout_preferences(self):
        """Test adding workout preferences."""
        mock_session = ConversationSession(telegram_id=12345)
        mock_session_mgr = MagicMock()
        mock_session_mgr.get_session = AsyncMock(return_value=mock_session)
        mock_session_mgr.save_session = AsyncMock(return_value=True)

        with patch(
            "src.agent.tools.preferences.get_session_manager",
            return_value=mock_session_mgr,
        ):
            result = await update_workout_preferences(["yoga", "pilates"], 12345)

        assert "Noted" in result
        assert "yoga" in result
        assert mock_session.preferred_workout_types == ["yoga", "pilates"]

    @pytest.mark.asyncio
    async def test_merge_workout_preferences(self):
        """Test merging with existing preferences."""
        mock_session = ConversationSession(
            telegram_id=12345, preferred_workout_types=["yoga"]
        )
        mock_session_mgr = MagicMock()
        mock_session_mgr.get_session = AsyncMock(return_value=mock_session)
        mock_session_mgr.save_session = AsyncMock(return_value=True)

        with patch(
            "src.agent.tools.preferences.get_session_manager",
            return_value=mock_session_mgr,
        ):
            result = await update_workout_preferences(["pilates", "spinning"], 12345)

        assert "pilates" in result
        assert "spinning" in result
        assert len(mock_session.preferred_workout_types) == 3


class TestGetUserPreferences:
    """Tests for get_user_preferences helper."""

    @pytest.mark.asyncio
    async def test_get_preferences(self):
        """Test retrieving user preferences."""
        mock_session = ConversationSession(
            telegram_id=12345,
            first_name="Alice",
            location="NYC",
            fitness_goals=["strength"],
            preferred_workout_types=["crossfit"],
        )
        mock_session_mgr = MagicMock()
        mock_session_mgr.get_session = AsyncMock(return_value=mock_session)

        with patch(
            "src.agent.tools.preferences.get_session_manager",
            return_value=mock_session_mgr,
        ):
            prefs = await get_user_preferences(12345)

        assert prefs["location"] == "NYC"
        assert prefs["first_name"] == "Alice"
        assert prefs["fitness_goals"] == ["strength"]
        assert prefs["preferred_workout_types"] == ["crossfit"]

    @pytest.mark.asyncio
    async def test_get_preferences_error(self):
        """Test handling errors when getting preferences."""
        with patch(
            "src.agent.tools.preferences.get_session_manager",
            side_effect=Exception("Redis error"),
        ):
            prefs = await get_user_preferences(12345)

        assert prefs == {}


class TestPreferenceToolRegistration:
    """Tests for preference tool registration."""

    def test_register_preference_tools(self):
        """Test that preference tools are registered."""
        # Clear existing registrations
        tool_registry._tools.clear()

        register_preference_tools()

        tools = tool_registry.list_tools()
        assert "update_user_location" in tools
        assert "update_fitness_goals" in tools
        assert "update_workout_preferences" in tools

    def test_tools_have_correct_schemas(self):
        """Test that tools have proper input schemas."""
        if "update_user_location" not in tool_registry.list_tools():
            register_preference_tools()

        location_tool = tool_registry.get("update_user_location")
        goals_tool = tool_registry.get("update_fitness_goals")
        workout_tool = tool_registry.get("update_workout_preferences")

        assert location_tool is not None
        assert goals_tool is not None
        assert workout_tool is not None

        # Verify schemas exist
        assert location_tool.args_schema is not None
        assert goals_tool.args_schema is not None
        assert workout_tool.args_schema is not None
