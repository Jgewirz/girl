"""Telegram bot application factory."""

from telegram.ext import Application, PicklePersistence

from src.config import settings
from src.config.logging import get_logger

from .handlers import setup_handlers

logger = get_logger(__name__)


def create_bot_application() -> Application:
    """Create and configure the Telegram bot application."""
    logger.info("creating_bot_application", environment=settings.environment)

    # Create application with persistence for user data
    builder = Application.builder()
    builder.token(settings.telegram_bot_token.get_secret_value())

    # Add persistence for conversation history (simple file-based for dev)
    if not settings.is_production:
        persistence = PicklePersistence(filepath="bot_data.pickle")
        builder.persistence(persistence)

    app = builder.build()

    # Register handlers
    setup_handlers(app)

    logger.info("bot_application_created")
    return app
