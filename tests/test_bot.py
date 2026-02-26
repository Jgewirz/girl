"""Tests for Telegram bot handlers."""

import pytest
from unittest.mock import AsyncMock, MagicMock


def test_handlers_module_imports():
    """Test that handlers module can be imported."""
    from src.bot import handlers

    assert hasattr(handlers, "start_command")
    assert hasattr(handlers, "help_command")
    assert hasattr(handlers, "handle_message")
    assert hasattr(handlers, "setup_handlers")


def test_get_agent_singleton():
    """Test that get_agent returns the same instance."""
    from src.bot.handlers import get_agent

    agent1 = get_agent()
    agent2 = get_agent()

    assert agent1 is agent2


@pytest.mark.asyncio
async def test_start_command():
    """Test /start command handler."""
    from src.bot.handlers import start_command

    # Mock update and context
    update = MagicMock()
    update.effective_user.id = 12345
    update.effective_user.username = "testuser"
    update.effective_user.first_name = "Test"
    update.message.reply_text = AsyncMock()

    context = MagicMock()

    await start_command(update, context)

    update.message.reply_text.assert_called_once()
    call_args = update.message.reply_text.call_args[0][0]
    assert "Test" in call_args  # Should include user's first name
    assert "GirlBot" in call_args


@pytest.mark.asyncio
async def test_help_command():
    """Test /help command handler."""
    from src.bot.handlers import help_command

    update = MagicMock()
    update.message.reply_text = AsyncMock()

    context = MagicMock()

    await help_command(update, context)

    update.message.reply_text.assert_called_once()
    call_args = update.message.reply_text.call_args[0][0]
    assert "Fitness" in call_args
    assert "Wellness" in call_args
