"""Telegram bot message handlers."""

from langchain_core.messages import HumanMessage
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.agent import AgentState, create_agent
from src.agent.state import UserContext
from src.cache.session import get_session_manager
from src.config.logging import get_logger

logger = get_logger(__name__)

# Create agent instance at module level (reused across requests)
_agent = None

# Rate limit settings
RATE_LIMIT_MAX_REQUESTS = 30  # per minute
RATE_LIMIT_WINDOW = 60  # seconds


def get_agent():
    """Get or create the agent instance."""
    global _agent
    if _agent is None:
        _agent = create_agent()
    return _agent


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    user = update.effective_user
    logger.info("start_command", user_id=user.id, username=user.username)

    # Initialize or update session in Redis
    try:
        session_mgr = await get_session_manager()
        await session_mgr.update_user_info(
            telegram_id=user.id,
            first_name=user.first_name,
            username=user.username,
        )
    except Exception as e:
        logger.warning("session_init_failed", user_id=user.id, error=str(e))

    welcome_message = f"""Hey {user.first_name}! I'm GirlBot, your fitness and wellness assistant.

I can help you with:
- Finding fitness classes near you
- Workout suggestions and tips
- Wellness and self-care recommendations
- Healthy living advice

Just send me a message and let's get started! What are you looking to do today?"""

    await update.message.reply_text(welcome_message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    help_text = """Here's what I can help you with:

**Fitness**
- Find yoga, pilates, spin classes
- Get workout recommendations
- Track your fitness goals

**Wellness**
- Self-care suggestions
- Nutrition tips
- Mindfulness and recovery

**Commands**
/start - Start fresh
/help - Show this message
/settings - Your preferences
/clear - Clear conversation history

Just type naturally - I'm here to help!"""

    await update.message.reply_text(help_text, parse_mode="Markdown")


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /settings command."""
    user = update.effective_user

    # Load session from Redis
    location = "Not set yet"
    fitness_goals = []

    try:
        session_mgr = await get_session_manager()
        session = await session_mgr.get_session(user.id)
        if session.location:
            location = session.location
        if session.fitness_goals:
            fitness_goals = session.fitness_goals
    except Exception as e:
        logger.warning("settings_load_failed", user_id=user.id, error=str(e))

    goals_text = ", ".join(fitness_goals) if fitness_goals else "Not set yet"

    settings_text = f"""**Your Settings**

Name: {user.first_name or 'Not set'}
Location: {location}
Fitness Goals: {goals_text}

To update your location, just tell me where you are!
Example: "I'm in San Francisco"

To set fitness goals, tell me what you want to achieve!
Example: "I want to build strength and improve flexibility" """

    await update.message.reply_text(settings_text, parse_mode="Markdown")


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /clear command - clears conversation history."""
    user = update.effective_user
    logger.info("clear_command", user_id=user.id)

    try:
        session_mgr = await get_session_manager()
        session = await session_mgr.get_session(user.id)

        # Clear messages but keep user preferences
        session.messages = []
        session.conversation_summary = None
        session.message_count = 0

        await session_mgr.save_session(session)
        await update.message.reply_text(
            "Conversation cleared! Feel free to start fresh."
        )
    except Exception as e:
        logger.error("clear_failed", user_id=user.id, error=str(e))
        await update.message.reply_text(
            "Couldn't clear the conversation. Please try again."
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming text messages via LangGraph agent."""
    user = update.effective_user
    message_text = update.message.text

    logger.info(
        "incoming_message",
        user_id=user.id,
        username=user.username,
        message_length=len(message_text),
    )

    # Check rate limit
    try:
        session_mgr = await get_session_manager()
        is_allowed, count = await session_mgr.check_rate_limit(
            telegram_id=user.id,
            max_requests=RATE_LIMIT_MAX_REQUESTS,
            window_seconds=RATE_LIMIT_WINDOW,
        )

        if not is_allowed:
            await update.message.reply_text(
                "You're sending messages too quickly! Please wait a moment before trying again."
            )
            return
    except Exception as e:
        logger.warning("rate_limit_check_failed", user_id=user.id, error=str(e))
        # Continue on rate limit failure - fail open

    # Show typing indicator
    await update.message.chat.send_action("typing")

    try:
        # Get session from Redis
        session_mgr = await get_session_manager()
        session = await session_mgr.get_session(user.id)

        # Update user info if changed
        if user.first_name and session.first_name != user.first_name:
            session.first_name = user.first_name
        if user.username and session.username != user.username:
            session.username = user.username

        # Build user context from session
        user_context = UserContext(
            telegram_id=user.id,
            username=session.username,
            first_name=session.first_name,
            location=session.location,
            fitness_goals=session.fitness_goals,
            preferred_workout_types=session.preferred_workout_types,
            conversation_summary=session.conversation_summary,
        )

        # Get conversation history from session
        conversation_history = session.to_langchain_messages()

        # Build initial state
        initial_state = AgentState(
            messages=conversation_history + [HumanMessage(content=message_text)],
            user=user_context,
        )

        # Run the agent
        agent = get_agent()
        result = await agent.ainvoke(initial_state)

        # Extract response
        response_messages = result.get("messages", [])
        if response_messages:
            last_message = response_messages[-1]
            response_text = (
                last_message.content
                if hasattr(last_message, "content")
                else str(last_message)
            )
        else:
            response_text = "I'm not sure how to respond to that. Could you rephrase?"

        # Update session with new messages
        session.add_message("human", message_text)
        session.add_message("assistant", response_text)

        # Save session to Redis
        await session_mgr.save_session(session)

        logger.info(
            "response_sent",
            user_id=user.id,
            response_length=len(response_text),
            session_message_count=session.message_count,
        )

        await update.message.reply_text(response_text)

    except Exception as e:
        logger.error("message_handler_error", user_id=user.id, error=str(e))
        await update.message.reply_text(
            "Oops! Something went wrong on my end. Please try again in a moment."
        )


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors in the bot."""
    logger.error(
        "bot_error",
        error=str(context.error),
        update=str(update) if update else None,
    )


def setup_handlers(app: Application) -> None:
    """Register all handlers with the application."""
    # Command handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CommandHandler("clear", clear_command))

    # Message handler (catch-all for text)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Error handler
    app.add_error_handler(error_handler)

    logger.info("handlers_registered")
