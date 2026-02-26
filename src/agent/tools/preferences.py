"""User preference extraction and management tools."""

from pydantic import BaseModel, Field

from src.cache.session import get_session_manager
from src.config.logging import get_logger

from .registry import tool_registry

logger = get_logger(__name__)


class UpdateLocationInput(BaseModel):
    """Input schema for updating user location."""

    location: str = Field(
        description="The user's location (city, neighborhood, or address). "
        "Examples: 'San Francisco', 'Brooklyn, NY', 'downtown Seattle', 'near Central Park'"
    )
    telegram_id: int = Field(description="The user's Telegram ID")


class UpdateFitnessGoalsInput(BaseModel):
    """Input schema for updating fitness goals."""

    goals: list[str] = Field(
        description="List of fitness goals extracted from conversation. "
        "Examples: ['lose weight', 'build muscle', 'improve flexibility', 'reduce stress']"
    )
    telegram_id: int = Field(description="The user's Telegram ID")


class UpdateWorkoutPreferencesInput(BaseModel):
    """Input schema for updating workout preferences."""

    workout_types: list[str] = Field(
        description="List of preferred workout types. "
        "Examples: ['yoga', 'pilates', 'spinning', 'crossfit', 'swimming', 'boxing']"
    )
    telegram_id: int = Field(description="The user's Telegram ID")


async def update_user_location(location: str, telegram_id: int) -> str:
    """
    Update the user's location preference.

    Call this when the user mentions where they are or where they want to search.
    Examples of triggers:
    - "I'm in San Francisco"
    - "I live in Brooklyn"
    - "Find studios near downtown Chicago"
    - "I'm looking for gyms in Austin"

    Args:
        location: The location mentioned by the user
        telegram_id: User's Telegram ID

    Returns:
        Confirmation message
    """
    logger.info(
        "update_user_location_called",
        telegram_id=telegram_id,
        location=location,
    )

    try:
        session_mgr = await get_session_manager()
        session = await session_mgr.get_session(telegram_id)

        old_location = session.location
        session.location = location
        await session_mgr.save_session(session)

        if old_location:
            logger.info(
                "location_updated",
                telegram_id=telegram_id,
                old_location=old_location,
                new_location=location,
            )
            return f"Updated your location from {old_location} to {location}."
        else:
            logger.info(
                "location_set",
                telegram_id=telegram_id,
                location=location,
            )
            return f"Got it! I'll remember you're in {location} for future searches."

    except Exception as e:
        logger.error("update_location_error", telegram_id=telegram_id, error=str(e))
        return "I noted your location but couldn't save it permanently."


async def update_fitness_goals(goals: list[str], telegram_id: int) -> str:
    """
    Update the user's fitness goals.

    Call this when the user mentions their fitness objectives.
    Examples of triggers:
    - "I want to lose weight"
    - "I'm trying to build strength"
    - "My goal is to be more flexible"
    - "I want to reduce stress and improve my mental health"

    Args:
        goals: List of fitness goals
        telegram_id: User's Telegram ID

    Returns:
        Confirmation message
    """
    logger.info(
        "update_fitness_goals_called",
        telegram_id=telegram_id,
        goals=goals,
    )

    try:
        session_mgr = await get_session_manager()
        session = await session_mgr.get_session(telegram_id)

        # Merge with existing goals, avoiding duplicates
        existing = set(g.lower() for g in session.fitness_goals)
        new_goals = [g for g in goals if g.lower() not in existing]

        if new_goals:
            session.fitness_goals.extend(new_goals)
            await session_mgr.save_session(session)

            logger.info(
                "fitness_goals_updated",
                telegram_id=telegram_id,
                new_goals=new_goals,
                total_goals=session.fitness_goals,
            )
            return f"Added to your fitness goals: {', '.join(new_goals)}. I'll keep these in mind for recommendations!"
        else:
            return "I already have those goals noted for you!"

    except Exception as e:
        logger.error("update_goals_error", telegram_id=telegram_id, error=str(e))
        return "I noted your goals but couldn't save them permanently."


async def update_workout_preferences(workout_types: list[str], telegram_id: int) -> str:
    """
    Update the user's preferred workout types.

    Call this when the user expresses preference for certain activities.
    Examples of triggers:
    - "I love yoga"
    - "I prefer pilates over other workouts"
    - "I'm really into spinning and cycling"
    - "I enjoy swimming and water aerobics"

    Args:
        workout_types: List of preferred workout types
        telegram_id: User's Telegram ID

    Returns:
        Confirmation message
    """
    logger.info(
        "update_workout_preferences_called",
        telegram_id=telegram_id,
        workout_types=workout_types,
    )

    try:
        session_mgr = await get_session_manager()
        session = await session_mgr.get_session(telegram_id)

        # Merge with existing preferences
        existing = set(w.lower() for w in session.preferred_workout_types)
        new_types = [w for w in workout_types if w.lower() not in existing]

        if new_types:
            session.preferred_workout_types.extend(new_types)
            await session_mgr.save_session(session)

            logger.info(
                "workout_preferences_updated",
                telegram_id=telegram_id,
                new_types=new_types,
                total_types=session.preferred_workout_types,
            )
            return f"Noted! You enjoy: {', '.join(new_types)}. I'll prioritize these in searches!"
        else:
            return "I already know you like those activities!"

    except Exception as e:
        logger.error("update_preferences_error", telegram_id=telegram_id, error=str(e))
        return "I noted your preferences but couldn't save them permanently."


async def get_user_preferences(telegram_id: int) -> dict:
    """
    Get the user's saved preferences.

    This is a helper function (not a tool) for other tools to use.

    Args:
        telegram_id: User's Telegram ID

    Returns:
        Dictionary with user preferences
    """
    try:
        session_mgr = await get_session_manager()
        session = await session_mgr.get_session(telegram_id)

        return {
            "location": session.location,
            "fitness_goals": session.fitness_goals,
            "preferred_workout_types": session.preferred_workout_types,
            "first_name": session.first_name,
        }
    except Exception as e:
        logger.error("get_preferences_error", telegram_id=telegram_id, error=str(e))
        return {}


def register_preference_tools() -> None:
    """Register preference management tools with the tool registry."""

    tool_registry.register(
        name="update_user_location",
        description=(
            "Update the user's location for future searches. Call this whenever "
            "the user mentions where they are, where they live, or where they want "
            "to find fitness studios. This helps provide better, location-aware results."
        ),
        func=update_user_location,
        args_schema=UpdateLocationInput,
        calls_per_minute=60,
    )

    tool_registry.register(
        name="update_fitness_goals",
        description=(
            "Save the user's fitness goals for personalized recommendations. "
            "Call this when users mention their objectives like losing weight, "
            "building muscle, improving flexibility, reducing stress, etc."
        ),
        func=update_fitness_goals,
        args_schema=UpdateFitnessGoalsInput,
        calls_per_minute=60,
    )

    tool_registry.register(
        name="update_workout_preferences",
        description=(
            "Save the user's preferred workout types. Call this when users express "
            "that they enjoy or prefer certain activities like yoga, pilates, "
            "spinning, swimming, crossfit, boxing, etc."
        ),
        func=update_workout_preferences,
        args_schema=UpdateWorkoutPreferencesInput,
        calls_per_minute=60,
    )

    logger.info("preference_tools_registered")
