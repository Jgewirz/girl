"""Google Places API client for fitness studio discovery."""

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

from src.cache.redis import RedisClient
from src.config import settings
from src.config.logging import get_logger

logger = get_logger(__name__)

# Google Places API endpoints
PLACES_BASE_URL = "https://places.googleapis.com/v1"
NEARBY_SEARCH_URL = f"{PLACES_BASE_URL}/places:searchNearby"
PLACE_DETAILS_URL = f"{PLACES_BASE_URL}/places"
TEXT_SEARCH_URL = f"{PLACES_BASE_URL}/places:searchText"

# Cache settings
CACHE_TTL_SEARCH = 60 * 30  # 30 minutes for search results
CACHE_TTL_DETAILS = 60 * 60 * 24  # 24 hours for place details
CACHE_PREFIX = "places:"


class PlaceType(str, Enum):
    """Fitness-related place types for Google Places API."""

    GYM = "gym"
    YOGA_STUDIO = "yoga_studio"
    PILATES_STUDIO = "pilates_studio"
    FITNESS_CENTER = "fitness_center"
    SPORTS_CLUB = "sports_club"
    SWIMMING_POOL = "swimming_pool"
    SPA = "spa"


# Mapping of user-friendly terms to place types
ACTIVITY_TO_PLACE_TYPES: dict[str, list[str]] = {
    "yoga": ["yoga_studio", "gym", "fitness_center"],
    "pilates": ["pilates_studio", "gym", "fitness_center"],
    "gym": ["gym", "fitness_center", "sports_club"],
    "fitness": ["fitness_center", "gym", "sports_club"],
    "spin": ["gym", "fitness_center"],
    "cycling": ["gym", "fitness_center"],
    "swimming": ["swimming_pool", "sports_club"],
    "spa": ["spa"],
    "wellness": ["spa", "yoga_studio", "fitness_center"],
    "crossfit": ["gym", "fitness_center"],
    "boxing": ["gym", "sports_club"],
    "martial arts": ["gym", "sports_club"],
    "dance": ["gym", "fitness_center"],
    "barre": ["pilates_studio", "fitness_center", "gym"],
}


@dataclass
class PlaceResult:
    """Structured result from Places API."""

    place_id: str
    name: str
    address: str
    rating: float | None = None
    rating_count: int | None = None
    price_level: int | None = None
    is_open: bool | None = None
    types: list[str] | None = None
    phone: str | None = None
    website: str | None = None
    google_maps_url: str | None = None
    photo_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "place_id": self.place_id,
            "name": self.name,
            "address": self.address,
            "rating": self.rating,
            "rating_count": self.rating_count,
            "price_level": self.price_level,
            "is_open": self.is_open,
            "types": self.types,
            "phone": self.phone,
            "website": self.website,
            "google_maps_url": self.google_maps_url,
            "photo_url": self.photo_url,
        }

    def format_for_display(self) -> str:
        """Format place for Telegram message display."""
        parts = [f"**{self.name}**"]

        if self.rating:
            stars = "⭐" * int(self.rating)
            rating_text = f"{self.rating}"
            if self.rating_count:
                rating_text += f" ({self.rating_count} reviews)"
            parts.append(f"{stars} {rating_text}")

        parts.append(f"📍 {self.address}")

        if self.is_open is not None:
            status = "🟢 Open now" if self.is_open else "🔴 Closed"
            parts.append(status)

        if self.phone:
            parts.append(f"📞 {self.phone}")

        return "\n".join(parts)


class PlacesClient:
    """Client for Google Places API (New) with caching."""

    def __init__(self, api_key: str, redis_client: RedisClient | None = None):
        self._api_key = api_key
        self._redis = redis_client
        self._http_client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0),
                headers={
                    "X-Goog-Api-Key": self._api_key,
                    "Content-Type": "application/json",
                },
            )
        return self._http_client

    def _cache_key(self, prefix: str, *args: str) -> str:
        """Generate cache key from arguments."""
        key_data = ":".join(str(a) for a in args)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:12]
        return f"{CACHE_PREFIX}{prefix}:{key_hash}"

    async def _get_cached(self, key: str) -> str | None:
        """Get cached result if available."""
        if self._redis:
            return await self._redis.get(key)
        return None

    async def _set_cached(self, key: str, value: str, ttl: int) -> None:
        """Cache a result."""
        if self._redis:
            await self._redis.set(key, value, ttl_seconds=ttl)

    async def search_nearby(
        self,
        latitude: float,
        longitude: float,
        activity_type: str | None = None,
        radius_meters: int = 5000,
        max_results: int = 10,
    ) -> list[PlaceResult]:
        """
        Search for fitness places near a location.

        Args:
            latitude: Location latitude
            longitude: Location longitude
            activity_type: Type of activity (yoga, gym, pilates, etc.)
            radius_meters: Search radius in meters (default 5km)
            max_results: Maximum results to return

        Returns:
            List of PlaceResult objects
        """
        # Determine place types based on activity
        if activity_type:
            activity_lower = activity_type.lower()
            included_types = ACTIVITY_TO_PLACE_TYPES.get(
                activity_lower, ["gym", "fitness_center"]
            )
        else:
            included_types = ["gym", "fitness_center", "yoga_studio"]

        # Check cache
        cache_key = self._cache_key(
            "nearby", str(latitude), str(longitude), activity_type or "all", str(radius_meters)
        )
        cached = await self._get_cached(cache_key)
        if cached:
            import json
            logger.debug("places_cache_hit", cache_key=cache_key)
            data = json.loads(cached)
            return [PlaceResult(**p) for p in data]

        # Build request
        request_body = {
            "includedTypes": included_types,
            "maxResultCount": max_results,
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": latitude, "longitude": longitude},
                    "radius": float(radius_meters),
                }
            },
        }

        # Field mask for response
        field_mask = ",".join([
            "places.id",
            "places.displayName",
            "places.formattedAddress",
            "places.rating",
            "places.userRatingCount",
            "places.priceLevel",
            "places.currentOpeningHours",
            "places.types",
            "places.nationalPhoneNumber",
            "places.websiteUri",
            "places.googleMapsUri",
        ])

        try:
            client = await self._get_client()
            response = await client.post(
                NEARBY_SEARCH_URL,
                json=request_body,
                headers={"X-Goog-FieldMask": field_mask},
            )

            if response.status_code == 200:
                data = response.json()
                results = self._parse_places_response(data)

                # Cache results
                import json
                cache_data = json.dumps([r.to_dict() for r in results])
                await self._set_cached(cache_key, cache_data, CACHE_TTL_SEARCH)

                logger.info(
                    "places_search_success",
                    activity=activity_type,
                    result_count=len(results),
                )
                return results
            else:
                logger.error(
                    "places_search_error",
                    status_code=response.status_code,
                    response=response.text[:500],
                )
                return []

        except httpx.RequestError as e:
            logger.error("places_request_error", error=str(e))
            return []

    async def search_text(
        self,
        query: str,
        location: tuple[float, float] | None = None,
        radius_meters: int = 10000,
        max_results: int = 10,
    ) -> list[PlaceResult]:
        """
        Search for places using text query.

        Args:
            query: Search query (e.g., "yoga studios in San Francisco")
            location: Optional (lat, lng) to bias results
            radius_meters: Search radius when location provided
            max_results: Maximum results to return

        Returns:
            List of PlaceResult objects
        """
        # Check cache
        loc_str = f"{location[0]},{location[1]}" if location else "none"
        cache_key = self._cache_key("text", query, loc_str)
        cached = await self._get_cached(cache_key)
        if cached:
            import json
            logger.debug("places_cache_hit", cache_key=cache_key)
            data = json.loads(cached)
            return [PlaceResult(**p) for p in data]

        # Build request
        request_body: dict[str, Any] = {
            "textQuery": query,
            "maxResultCount": max_results,
        }

        if location:
            request_body["locationBias"] = {
                "circle": {
                    "center": {"latitude": location[0], "longitude": location[1]},
                    "radius": float(radius_meters),
                }
            }

        # Field mask
        field_mask = ",".join([
            "places.id",
            "places.displayName",
            "places.formattedAddress",
            "places.rating",
            "places.userRatingCount",
            "places.priceLevel",
            "places.currentOpeningHours",
            "places.types",
            "places.nationalPhoneNumber",
            "places.websiteUri",
            "places.googleMapsUri",
        ])

        try:
            client = await self._get_client()
            response = await client.post(
                TEXT_SEARCH_URL,
                json=request_body,
                headers={"X-Goog-FieldMask": field_mask},
            )

            if response.status_code == 200:
                data = response.json()
                results = self._parse_places_response(data)

                # Cache results
                import json
                cache_data = json.dumps([r.to_dict() for r in results])
                await self._set_cached(cache_key, cache_data, CACHE_TTL_SEARCH)

                logger.info(
                    "places_text_search_success",
                    query=query,
                    result_count=len(results),
                )
                return results
            else:
                logger.error(
                    "places_text_search_error",
                    status_code=response.status_code,
                    response=response.text[:500],
                )
                return []

        except httpx.RequestError as e:
            logger.error("places_text_request_error", error=str(e))
            return []

    async def get_place_details(self, place_id: str) -> PlaceResult | None:
        """
        Get detailed information about a specific place.

        Args:
            place_id: Google Place ID

        Returns:
            PlaceResult with full details, or None if not found
        """
        # Check cache
        cache_key = self._cache_key("details", place_id)
        cached = await self._get_cached(cache_key)
        if cached:
            import json
            logger.debug("places_details_cache_hit", place_id=place_id)
            return PlaceResult(**json.loads(cached))

        # Field mask for details
        field_mask = ",".join([
            "id",
            "displayName",
            "formattedAddress",
            "rating",
            "userRatingCount",
            "priceLevel",
            "currentOpeningHours",
            "types",
            "nationalPhoneNumber",
            "internationalPhoneNumber",
            "websiteUri",
            "googleMapsUri",
        ])

        try:
            client = await self._get_client()
            url = f"{PLACE_DETAILS_URL}/{place_id}"
            response = await client.get(
                url,
                headers={"X-Goog-FieldMask": field_mask},
            )

            if response.status_code == 200:
                data = response.json()
                result = self._parse_single_place(data)

                if result:
                    # Cache result
                    import json
                    await self._set_cached(
                        cache_key, json.dumps(result.to_dict()), CACHE_TTL_DETAILS
                    )

                logger.info("places_details_success", place_id=place_id)
                return result
            else:
                logger.error(
                    "places_details_error",
                    place_id=place_id,
                    status_code=response.status_code,
                )
                return None

        except httpx.RequestError as e:
            logger.error("places_details_request_error", place_id=place_id, error=str(e))
            return None

    def _parse_places_response(self, data: dict[str, Any]) -> list[PlaceResult]:
        """Parse Places API response into PlaceResult objects."""
        results = []
        for place in data.get("places", []):
            result = self._parse_single_place(place)
            if result:
                results.append(result)
        return results

    def _parse_single_place(self, place: dict[str, Any]) -> PlaceResult | None:
        """Parse a single place object."""
        try:
            place_id = place.get("id", "")
            if not place_id:
                return None

            # Extract display name
            display_name = place.get("displayName", {})
            name = display_name.get("text", "") if isinstance(display_name, dict) else str(display_name)

            # Extract opening hours
            is_open = None
            opening_hours = place.get("currentOpeningHours", {})
            if opening_hours:
                is_open = opening_hours.get("openNow")

            return PlaceResult(
                place_id=place_id,
                name=name,
                address=place.get("formattedAddress", ""),
                rating=place.get("rating"),
                rating_count=place.get("userRatingCount"),
                price_level=place.get("priceLevel"),
                is_open=is_open,
                types=place.get("types"),
                phone=place.get("nationalPhoneNumber") or place.get("internationalPhoneNumber"),
                website=place.get("websiteUri"),
                google_maps_url=place.get("googleMapsUri"),
            )
        except Exception as e:
            logger.warning("places_parse_error", error=str(e))
            return None

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()


# Singleton instance
_places_client: PlacesClient | None = None


async def get_places_client() -> PlacesClient | None:
    """Get or create the Places client singleton."""
    global _places_client

    if not settings.google_places_api_key:
        logger.warning("google_places_not_configured")
        return None

    if _places_client is None:
        # Try to get Redis client for caching
        redis_client = None
        try:
            redis_client = await RedisClient.create()
        except Exception as e:
            logger.warning("places_redis_unavailable", error=str(e))

        _places_client = PlacesClient(
            api_key=settings.google_places_api_key.get_secret_value(),
            redis_client=redis_client,
        )

    return _places_client
