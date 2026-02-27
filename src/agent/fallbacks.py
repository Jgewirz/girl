"""Graceful degradation patterns and fallback responses.

This module provides user-friendly fallback messages for all failure scenarios.
Technical details are logged internally while users receive helpful, actionable messages.

Usage:
    from src.agent.fallbacks import get_fallback, FallbackType

    # Get a fallback message for a specific failure type
    message = get_fallback(FallbackType.VISION_UNAVAILABLE, context={"query": "my outfit"})
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from src.config.logging import get_logger

logger = get_logger(__name__)


class FallbackType(Enum):
    """Types of failures that can occur."""

    # Vision/Photo Analysis
    VISION_UNAVAILABLE = auto()
    VISION_RATE_LIMITED = auto()
    PHOTO_DOWNLOAD_FAILED = auto()
    PHOTO_TOO_LARGE = auto()
    PHOTO_INVALID_FORMAT = auto()

    # Places/Fitness Discovery
    PLACES_UNAVAILABLE = auto()
    PLACES_RATE_LIMITED = auto()
    PLACES_NO_RESULTS = auto()
    LOCATION_REQUIRED = auto()

    # Travel Services
    FLIGHTS_UNAVAILABLE = auto()
    HOTELS_UNAVAILABLE = auto()
    TRAVEL_NOT_CONFIGURED = auto()

    # Session/Cache
    SESSION_UNAVAILABLE = auto()
    CACHE_UNAVAILABLE = auto()

    # Configuration
    FEATURE_DISABLED = auto()
    API_KEY_MISSING = auto()
    API_KEY_INVALID = auto()

    # General
    UNKNOWN_ERROR = auto()
    TIMEOUT = auto()
    RATE_LIMITED = auto()
    SERVICE_OVERLOADED = auto()

    # Tool-specific
    TOOL_EXECUTION_FAILED = auto()
    TOOL_NOT_FOUND = auto()


class ErrorCategory(Enum):
    """Category of error for determining retry behavior."""

    TRANSIENT = "transient"  # Try again later (network, timeout)
    PERMANENT = "permanent"  # Won't work without changes (config, auth)
    RESOURCE = "resource"  # Rate limited, quota exceeded


@dataclass
class FallbackResponse:
    """A fallback response with user message and metadata."""

    message: str
    category: ErrorCategory
    suggest_retry: bool = True
    alternative_action: str | None = None
    retry_after_seconds: int | None = None


# Fallback templates organized by type
FALLBACKS: dict[FallbackType, FallbackResponse] = {
    # Vision/Photo Analysis
    FallbackType.VISION_UNAVAILABLE: FallbackResponse(
        message=(
            "I can't analyze photos right now. Try describing your outfit in text "
            "and I'll help with style advice!"
        ),
        category=ErrorCategory.TRANSIENT,
        suggest_retry=True,
        alternative_action="Describe your outfit: colors, items, occasion",
    ),
    FallbackType.VISION_RATE_LIMITED: FallbackResponse(
        message=(
            "I've been looking at a lot of photos! Give me a minute to rest my eyes, "
            "then send your photo again."
        ),
        category=ErrorCategory.RESOURCE,
        suggest_retry=True,
        retry_after_seconds=60,
    ),
    FallbackType.PHOTO_DOWNLOAD_FAILED: FallbackResponse(
        message=(
            "I had trouble downloading your photo. Could you try sending it again? "
            "Make sure it's not too large (under 10MB works best)."
        ),
        category=ErrorCategory.TRANSIENT,
        suggest_retry=True,
    ),
    FallbackType.PHOTO_TOO_LARGE: FallbackResponse(
        message=(
            "That photo is a bit too large for me to process. Could you resize it "
            "or send a smaller version? Under 10MB works best."
        ),
        category=ErrorCategory.PERMANENT,
        suggest_retry=False,
        alternative_action="Resize the image or use a lower quality setting",
    ),
    FallbackType.PHOTO_INVALID_FORMAT: FallbackResponse(
        message=(
            "I couldn't read that image format. Could you try sending it as a "
            "JPEG or PNG?"
        ),
        category=ErrorCategory.PERMANENT,
        suggest_retry=False,
        alternative_action="Convert to JPEG or PNG format",
    ),

    # Places/Fitness Discovery
    FallbackType.PLACES_UNAVAILABLE: FallbackResponse(
        message=(
            "I'm having trouble searching for places right now. In the meantime, "
            "try searching on Google Maps - it has great fitness studio listings!"
        ),
        category=ErrorCategory.TRANSIENT,
        suggest_retry=True,
        alternative_action="Search on Google Maps",
    ),
    FallbackType.PLACES_RATE_LIMITED: FallbackResponse(
        message=(
            "I've done a lot of searching recently. Give me a moment, then I'll "
            "be ready to find studios for you again."
        ),
        category=ErrorCategory.RESOURCE,
        suggest_retry=True,
        retry_after_seconds=30,
    ),
    FallbackType.PLACES_NO_RESULTS: FallbackResponse(
        message=(
            "I couldn't find any fitness studios matching that search. Try being "
            "more specific about the type (yoga, pilates, gym) or location."
        ),
        category=ErrorCategory.PERMANENT,
        suggest_retry=False,
        alternative_action="Try a different search term or expand the area",
    ),
    FallbackType.LOCATION_REQUIRED: FallbackResponse(
        message=(
            "To find fitness studios near you, I need to know your location. "
            "You can tell me your city or area, like 'yoga studios in Brooklyn'."
        ),
        category=ErrorCategory.PERMANENT,
        suggest_retry=False,
        alternative_action="Share your location or specify a city",
    ),

    # Travel Services
    FallbackType.FLIGHTS_UNAVAILABLE: FallbackResponse(
        message=(
            "I can't search flights right now. Try checking Skyscanner or "
            "Google Flights - they often have the same deals I'd find!"
        ),
        category=ErrorCategory.TRANSIENT,
        suggest_retry=True,
        alternative_action="Visit skyscanner.com or flights.google.com",
    ),
    FallbackType.HOTELS_UNAVAILABLE: FallbackResponse(
        message=(
            "I'm having trouble searching hotels. Booking.com or Hotels.com "
            "are great alternatives to check in the meantime!"
        ),
        category=ErrorCategory.TRANSIENT,
        suggest_retry=True,
        alternative_action="Visit booking.com or hotels.com",
    ),
    FallbackType.TRAVEL_NOT_CONFIGURED: FallbackResponse(
        message=(
            "Travel planning isn't set up on this bot yet. I can still help "
            "with style, fitness, and wellness advice though!"
        ),
        category=ErrorCategory.PERMANENT,
        suggest_retry=False,
        alternative_action="Ask about style advice or fitness instead",
    ),

    # Session/Cache
    FallbackType.SESSION_UNAVAILABLE: FallbackResponse(
        message=(
            "Note: I might forget our conversation if I restart. But I'm still "
            "here to help! What would you like to know?"
        ),
        category=ErrorCategory.TRANSIENT,
        suggest_retry=False,
    ),
    FallbackType.CACHE_UNAVAILABLE: FallbackResponse(
        message="",  # Silent - no user-facing message, just internal handling
        category=ErrorCategory.TRANSIENT,
        suggest_retry=False,
    ),

    # Configuration
    FallbackType.FEATURE_DISABLED: FallbackResponse(
        message=(
            "That feature isn't available on this bot. But I can still help "
            "with style advice, outfit analysis, and color season discovery!"
        ),
        category=ErrorCategory.PERMANENT,
        suggest_retry=False,
        alternative_action="Try asking about style or outfit advice",
    ),
    FallbackType.API_KEY_MISSING: FallbackResponse(
        message=(
            "I'm not fully set up to do that yet. In the meantime, "
            "let me know what else I can help with!"
        ),
        category=ErrorCategory.PERMANENT,
        suggest_retry=False,
    ),
    FallbackType.API_KEY_INVALID: FallbackResponse(
        message=(
            "I'm having some technical difficulties. The bot administrator "
            "has been notified. Is there something else I can help with?"
        ),
        category=ErrorCategory.PERMANENT,
        suggest_retry=False,
    ),

    # General
    FallbackType.UNKNOWN_ERROR: FallbackResponse(
        message=(
            "Something unexpected happened. Let's try a different approach - "
            "what are you trying to do? I'll find another way to help."
        ),
        category=ErrorCategory.TRANSIENT,
        suggest_retry=True,
        alternative_action="Describe what you'd like to accomplish",
    ),
    FallbackType.TIMEOUT: FallbackResponse(
        message=(
            "That took longer than expected. Could you try again? If it "
            "keeps happening, try a simpler request."
        ),
        category=ErrorCategory.TRANSIENT,
        suggest_retry=True,
    ),
    FallbackType.RATE_LIMITED: FallbackResponse(
        message=(
            "I'm getting a lot of requests right now! Give me a minute and "
            "then try again."
        ),
        category=ErrorCategory.RESOURCE,
        suggest_retry=True,
        retry_after_seconds=60,
    ),
    FallbackType.SERVICE_OVERLOADED: FallbackResponse(
        message=(
            "I'm a bit overwhelmed right now. Try again in a few minutes - "
            "I'll be ready to help!"
        ),
        category=ErrorCategory.TRANSIENT,
        suggest_retry=True,
        retry_after_seconds=120,
    ),

    # Tool-specific
    FallbackType.TOOL_EXECUTION_FAILED: FallbackResponse(
        message=(
            "I ran into a problem trying to do that. Let me try a different "
            "approach - could you tell me more about what you need?"
        ),
        category=ErrorCategory.TRANSIENT,
        suggest_retry=True,
    ),
    FallbackType.TOOL_NOT_FOUND: FallbackResponse(
        message=(
            "I'm not sure how to do that specific thing, but I might be able "
            "to help another way. What are you trying to accomplish?"
        ),
        category=ErrorCategory.PERMANENT,
        suggest_retry=False,
    ),
}


def get_fallback(
    fallback_type: FallbackType,
    context: dict[str, Any] | None = None,
    error: Exception | None = None,
) -> str:
    """Get a user-friendly fallback message for a failure type.

    Args:
        fallback_type: The type of failure
        context: Optional context for customizing the message
        error: Optional exception for logging

    Returns:
        User-friendly fallback message
    """
    fallback = FALLBACKS.get(fallback_type, FALLBACKS[FallbackType.UNKNOWN_ERROR])

    # Log the technical details
    if error:
        logger.error(
            "fallback_triggered",
            fallback_type=fallback_type.name,
            error_type=type(error).__name__,
            error_message=str(error),
            category=fallback.category.value,
            context=context,
        )
    else:
        logger.warning(
            "fallback_triggered",
            fallback_type=fallback_type.name,
            category=fallback.category.value,
            context=context,
        )

    return fallback.message


def get_fallback_response(
    fallback_type: FallbackType,
    context: dict[str, Any] | None = None,
    error: Exception | None = None,
) -> FallbackResponse:
    """Get the full fallback response object for a failure type.

    Args:
        fallback_type: The type of failure
        context: Optional context for customizing the message
        error: Optional exception for logging

    Returns:
        FallbackResponse with message and metadata
    """
    fallback = FALLBACKS.get(fallback_type, FALLBACKS[FallbackType.UNKNOWN_ERROR])

    # Log the technical details
    if error:
        logger.error(
            "fallback_triggered",
            fallback_type=fallback_type.name,
            error_type=type(error).__name__,
            error_message=str(error),
            category=fallback.category.value,
            suggest_retry=fallback.suggest_retry,
            context=context,
        )
    else:
        logger.warning(
            "fallback_triggered",
            fallback_type=fallback_type.name,
            category=fallback.category.value,
            context=context,
        )

    return fallback


def categorize_exception(error: Exception) -> FallbackType:
    """Categorize an exception into a fallback type.

    Args:
        error: The exception to categorize

    Returns:
        Appropriate FallbackType for the error
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    # Timeout errors
    if "timeout" in error_str or "timeout" in error_type:
        return FallbackType.TIMEOUT

    # Rate limiting
    if "429" in error_str or "rate limit" in error_str or "too many" in error_str:
        return FallbackType.RATE_LIMITED

    # Authentication/API key issues
    if "401" in error_str or "403" in error_str or "unauthorized" in error_str:
        return FallbackType.API_KEY_INVALID
    if "api key" in error_str or "api_key" in error_str:
        return FallbackType.API_KEY_MISSING

    # Service unavailable
    if "503" in error_str or "502" in error_str or "service unavailable" in error_str:
        return FallbackType.SERVICE_OVERLOADED

    # Connection errors
    if "connection" in error_str or "network" in error_str:
        return FallbackType.UNKNOWN_ERROR

    return FallbackType.UNKNOWN_ERROR


def format_with_alternative(message: str, alternative: str | None) -> str:
    """Format a message with an alternative action suggestion.

    Args:
        message: The main message
        alternative: Optional alternative action

    Returns:
        Formatted message with alternative if provided
    """
    if alternative:
        return f"{message}\n\nAlternatively: {alternative}"
    return message


# Convenience functions for common fallback scenarios


def vision_fallback(error: Exception | None = None) -> str:
    """Get fallback message for vision service failures."""
    return get_fallback(FallbackType.VISION_UNAVAILABLE, error=error)


def places_fallback(query: str | None = None, error: Exception | None = None) -> str:
    """Get fallback message for places service failures."""
    message = get_fallback(FallbackType.PLACES_UNAVAILABLE, error=error)
    if query:
        message = message.replace(
            "search on Google Maps",
            f"search '{query}' on Google Maps"
        )
    return message


def tool_fallback(tool_name: str, error: Exception | None = None) -> str:
    """Get fallback message for generic tool failures."""
    return get_fallback(
        FallbackType.TOOL_EXECUTION_FAILED,
        context={"tool": tool_name},
        error=error,
    )


def unknown_fallback(error: Exception | None = None) -> str:
    """Get fallback message for unknown/unexpected failures."""
    return get_fallback(FallbackType.UNKNOWN_ERROR, error=error)


# Agent-level catch-all wrapper


async def with_fallback(
    coro,
    fallback_type: FallbackType = FallbackType.UNKNOWN_ERROR,
    context: dict[str, Any] | None = None,
):
    """Execute a coroutine with automatic fallback on failure.

    Args:
        coro: The coroutine to execute
        fallback_type: The fallback type to use on failure
        context: Optional context for logging

    Returns:
        Result of the coroutine, or fallback message on failure
    """
    try:
        return await coro
    except Exception as e:
        # Try to categorize the exception for a more specific fallback
        categorized = categorize_exception(e)
        if categorized != FallbackType.UNKNOWN_ERROR:
            fallback_type = categorized

        return get_fallback(fallback_type, context=context, error=e)
