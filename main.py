"""Main entry point for GirlBot."""

import sys

from src.agent.tools import register_all_tools
from src.bot import create_bot_application
from src.cache.redis import close_pool
from src.config import settings
from src.config.logging import get_logger, setup_logging


async def shutdown(app, logger):
    """Graceful shutdown handler."""
    logger.info("shutdown_initiated")

    # Close Redis connection pool
    await close_pool()

    logger.info("shutdown_complete")


def main() -> None:
    """Run the bot."""
    # Initialize logging
    setup_logging()
    logger = get_logger(__name__)

    logger.info(
        "starting_girlbot",
        environment=settings.environment,
        debug=settings.debug,
        llm_model=settings.openai_model_primary,
    )

    # Validate required settings
    try:
        _ = settings.telegram_bot_token.get_secret_value()
        if not settings.has_openai:
            logger.error("missing_openai_key")
            print("Error: OPENAI_API_KEY is required. Check your .env file.")
            sys.exit(1)
    except Exception as e:
        logger.error("configuration_error", error=str(e))
        print("Error: Missing required configuration. Check your .env file.")
        print("Required: TELEGRAM_BOT_TOKEN, OPENAI_API_KEY")
        sys.exit(1)

    # Log Redis configuration
    logger.info(
        "redis_configured",
        url=settings.redis_url.get_secret_value().split("@")[-1] if "@" in settings.redis_url.get_secret_value() else "localhost",
    )

    # Register agent tools
    register_all_tools()
    logger.info(
        "tools_registered",
        places_enabled=settings.google_places_api_key is not None,
    )

    # Create and run the bot
    app = create_bot_application()

    # Register shutdown handler
    async def on_shutdown():
        await shutdown(app, logger)

    app.post_shutdown = on_shutdown

    logger.info("bot_starting_polling")
    print("Bot is running! Press Ctrl+C to stop.")

    app.run_polling(
        allowed_updates=["message", "callback_query"],
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
