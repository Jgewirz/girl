"""Google Places tool for fitness studio discovery."""


from pydantic import BaseModel, Field

from src.agent.fallbacks import FallbackType, get_fallback, places_fallback
from src.config.logging import get_logger
from src.services.places import ACTIVITY_TO_PLACE_TYPES, PlaceResult, get_places_client

from .registry import tool_registry

logger = get_logger(__name__)


class SearchFitnessStudioInput(BaseModel):
    """Input schema for fitness studio search."""

    query: str = Field(
        description="Search query describing what the user is looking for. "
        "Examples: 'yoga studios', 'gyms near me', 'pilates classes', 'spin studio'"
    )
    location: str | None = Field(
        default=None,
        description="Location to search in. If not provided, will use user's saved location. "
        "Examples: 'San Francisco', 'Brooklyn, NY', 'downtown Chicago'"
    )
    activity_type: str | None = Field(
        default=None,
        description="Specific activity type to filter by. "
        f"Options: {', '.join(ACTIVITY_TO_PLACE_TYPES.keys())}"
    )


class GetStudioDetailsInput(BaseModel):
    """Input schema for getting studio details."""

    place_id: str = Field(description="Google Place ID of the studio to get details for")


def _format_results_for_agent(results: list[PlaceResult]) -> str:
    """Format search results for the LLM to process."""
    if not results:
        return "No fitness studios found matching your criteria. Try a different location or search term."

    output_parts = [f"Found {len(results)} fitness studios:\n"]

    for i, place in enumerate(results, 1):
        parts = [f"{i}. **{place.name}**"]

        if place.rating:
            rating_str = f"⭐ {place.rating}"
            if place.rating_count:
                rating_str += f" ({place.rating_count} reviews)"
            parts.append(rating_str)

        parts.append(f"   📍 {place.address}")

        if place.is_open is not None:
            status = "🟢 Open now" if place.is_open else "🔴 Closed"
            parts.append(f"   {status}")

        if place.website:
            parts.append(f"   🔗 {place.website}")

        # Include place_id for potential follow-up
        parts.append(f"   [ID: {place.place_id}]")

        output_parts.append("\n".join(parts))

    return "\n\n".join(output_parts)


def _format_details_for_agent(place: PlaceResult | None) -> str:
    """Format place details for the LLM to process."""
    if not place:
        return "Could not find details for this studio. The place may no longer exist."

    parts = [f"**{place.name}**\n"]

    if place.rating:
        rating_str = f"⭐ Rating: {place.rating}/5"
        if place.rating_count:
            rating_str += f" ({place.rating_count} reviews)"
        parts.append(rating_str)

    parts.append(f"📍 Address: {place.address}")

    if place.is_open is not None:
        status = "🟢 Currently Open" if place.is_open else "🔴 Currently Closed"
        parts.append(status)

    if place.phone:
        parts.append(f"📞 Phone: {place.phone}")

    if place.website:
        parts.append(f"🌐 Website: {place.website}")

    if place.google_maps_url:
        parts.append(f"🗺️ Maps: {place.google_maps_url}")

    if place.types:
        # Filter to relevant types
        relevant_types = [t for t in place.types if not t.startswith("point_of_interest")]
        if relevant_types:
            parts.append(f"📋 Type: {', '.join(relevant_types[:3])}")

    return "\n".join(parts)


async def search_fitness_studios(
    query: str,
    location: str | None = None,
    activity_type: str | None = None,
) -> str:
    """
    Search for fitness studios, gyms, yoga studios, and wellness centers.

    This tool searches Google Places to find fitness-related businesses
    based on the user's query and location.

    Args:
        query: What the user is looking for (e.g., "yoga studios", "gyms")
        location: City or area to search in
        activity_type: Specific activity (yoga, pilates, gym, etc.)

    Returns:
        Formatted string with search results for the LLM to present
    """
    logger.info(
        "search_fitness_studios_called",
        query=query,
        location=location,
        activity_type=activity_type,
    )

    client = await get_places_client()
    if not client:
        return get_fallback(FallbackType.FEATURE_DISABLED, context={"feature": "fitness"})

    # Build search query
    search_query = query
    if location:
        search_query = f"{query} in {location}"
    elif activity_type:
        search_query = f"{activity_type} studios {query}"

    try:
        # Use text search for flexibility
        results = await client.search_text(
            query=search_query,
            max_results=8,  # Keep results manageable for chat
        )

        return _format_results_for_agent(results)

    except Exception as e:
        logger.error("search_fitness_studios_error", error=str(e))
        return places_fallback(query=search_query, error=e)


async def get_studio_details(place_id: str) -> str:
    """
    Get detailed information about a specific fitness studio.

    Use this when the user wants more information about a specific
    studio from the search results.

    Args:
        place_id: The Google Place ID of the studio

    Returns:
        Formatted string with studio details
    """
    logger.info("get_studio_details_called", place_id=place_id)

    client = await get_places_client()
    if not client:
        return get_fallback(FallbackType.FEATURE_DISABLED, context={"feature": "fitness"})

    try:
        result = await client.get_place_details(place_id)
        return _format_details_for_agent(result)

    except Exception as e:
        logger.error("get_studio_details_error", place_id=place_id, error=str(e))
        return places_fallback(error=e)


def register_places_tools() -> None:
    """Register Google Places tools with the tool registry."""

    # Search tool
    tool_registry.register(
        name="search_fitness_studios",
        description=(
            "Search for fitness studios, gyms, yoga studios, pilates studios, "
            "and wellness centers. Use this when users ask about finding places "
            "to work out or take fitness classes. IMPORTANT: If the user has a "
            "saved location in their context, use it for the 'location' parameter "
            "unless they specify a different location in their message."
        ),
        func=search_fitness_studios,
        args_schema=SearchFitnessStudioInput,
        calls_per_minute=30,
        cache_ttl_seconds=60 * 30,  # 30 minutes
    )

    # Details tool
    tool_registry.register(
        name="get_studio_details",
        description=(
            "Get detailed information about a specific fitness studio including "
            "phone number, website, hours, and ratings. Use this when a user wants "
            "more information about a particular studio from search results."
        ),
        func=get_studio_details,
        args_schema=GetStudioDetailsInput,
        calls_per_minute=60,
        cache_ttl_seconds=60 * 60,  # 1 hour
    )

    logger.info("places_tools_registered")
