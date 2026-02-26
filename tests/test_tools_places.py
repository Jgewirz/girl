"""Tests for Google Places tool integration."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.places import (
    PlaceResult,
    PlacesClient,
    ACTIVITY_TO_PLACE_TYPES,
)
from src.agent.tools.places import (
    search_fitness_studios,
    get_studio_details,
    register_places_tools,
    _format_results_for_agent,
    _format_details_for_agent,
)
from src.agent.tools import tool_registry


class TestPlaceResult:
    """Tests for PlaceResult dataclass."""

    def test_create_place_result(self):
        """Test creating a PlaceResult."""
        place = PlaceResult(
            place_id="abc123",
            name="Test Yoga Studio",
            address="123 Main St, San Francisco, CA",
            rating=4.5,
            rating_count=100,
            is_open=True,
        )

        assert place.place_id == "abc123"
        assert place.name == "Test Yoga Studio"
        assert place.rating == 4.5
        assert place.is_open is True

    def test_to_dict(self):
        """Test serialization to dictionary."""
        place = PlaceResult(
            place_id="abc123",
            name="Test Studio",
            address="123 Main St",
            rating=4.0,
        )

        data = place.to_dict()

        assert data["place_id"] == "abc123"
        assert data["name"] == "Test Studio"
        assert data["rating"] == 4.0

    def test_format_for_display(self):
        """Test formatting for Telegram display."""
        place = PlaceResult(
            place_id="abc123",
            name="Yoga Flow Studio",
            address="456 Oak Ave, NYC",
            rating=4.8,
            rating_count=250,
            is_open=True,
            phone="+1-555-0123",
        )

        display = place.format_for_display()

        assert "Yoga Flow Studio" in display
        assert "4.8" in display
        assert "250 reviews" in display
        assert "456 Oak Ave" in display
        assert "Open now" in display
        assert "+1-555-0123" in display


class TestPlacesClient:
    """Tests for PlacesClient."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        redis = MagicMock()
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock(return_value=True)
        return redis

    @pytest.fixture
    def places_client(self, mock_redis):
        """Create PlacesClient with mocked dependencies."""
        return PlacesClient(api_key="test_api_key", redis_client=mock_redis)

    def test_activity_to_place_types_mapping(self):
        """Test that activity types are properly mapped."""
        assert "yoga_studio" in ACTIVITY_TO_PLACE_TYPES["yoga"]
        assert "gym" in ACTIVITY_TO_PLACE_TYPES["gym"]
        assert "pilates_studio" in ACTIVITY_TO_PLACE_TYPES["pilates"]
        assert "swimming_pool" in ACTIVITY_TO_PLACE_TYPES["swimming"]

    @pytest.mark.asyncio
    async def test_search_nearby_success(self, places_client, mock_redis):
        """Test successful nearby search."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "places": [
                {
                    "id": "place1",
                    "displayName": {"text": "Yoga Studio A"},
                    "formattedAddress": "123 Main St",
                    "rating": 4.5,
                    "userRatingCount": 100,
                    "currentOpeningHours": {"openNow": True},
                }
            ]
        }

        with patch.object(places_client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            results = await places_client.search_nearby(
                latitude=37.7749,
                longitude=-122.4194,
                activity_type="yoga",
            )

            assert len(results) == 1
            assert results[0].name == "Yoga Studio A"
            assert results[0].rating == 4.5

    @pytest.mark.asyncio
    async def test_search_nearby_cached(self, places_client, mock_redis):
        """Test that cached results are returned."""
        cached_data = json.dumps([
            {
                "place_id": "cached_place",
                "name": "Cached Studio",
                "address": "456 Cache St",
                "rating": 4.0,
                "rating_count": None,
                "price_level": None,
                "is_open": None,
                "types": None,
                "phone": None,
                "website": None,
                "google_maps_url": None,
                "photo_url": None,
            }
        ])
        mock_redis.get = AsyncMock(return_value=cached_data)

        results = await places_client.search_nearby(
            latitude=37.7749,
            longitude=-122.4194,
        )

        assert len(results) == 1
        assert results[0].name == "Cached Studio"

    @pytest.mark.asyncio
    async def test_search_nearby_api_error(self, places_client, mock_redis):
        """Test handling of API errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch.object(places_client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            results = await places_client.search_nearby(
                latitude=37.7749,
                longitude=-122.4194,
            )

            assert results == []

    @pytest.mark.asyncio
    async def test_text_search_success(self, places_client, mock_redis):
        """Test text search functionality."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "places": [
                {
                    "id": "place2",
                    "displayName": {"text": "CrossFit Box"},
                    "formattedAddress": "789 Gym Blvd",
                    "rating": 4.8,
                    "userRatingCount": 500,
                }
            ]
        }

        with patch.object(places_client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            results = await places_client.search_text(
                query="crossfit gyms in San Francisco"
            )

            assert len(results) == 1
            assert results[0].name == "CrossFit Box"

    @pytest.mark.asyncio
    async def test_get_place_details_success(self, places_client, mock_redis):
        """Test getting place details."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "detail_place",
            "displayName": {"text": "Premium Fitness"},
            "formattedAddress": "100 Fitness Way",
            "rating": 4.9,
            "userRatingCount": 1000,
            "nationalPhoneNumber": "+1-555-9999",
            "websiteUri": "https://premiumfitness.com",
            "googleMapsUri": "https://maps.google.com/?q=place",
        }

        with patch.object(places_client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            result = await places_client.get_place_details("detail_place")

            assert result is not None
            assert result.name == "Premium Fitness"
            assert result.phone == "+1-555-9999"
            assert result.website == "https://premiumfitness.com"


class TestPlacesTools:
    """Tests for the agent tools."""

    def test_format_results_empty(self):
        """Test formatting empty results."""
        output = _format_results_for_agent([])
        assert "No fitness studios found" in output

    def test_format_results_with_data(self):
        """Test formatting results with data."""
        results = [
            PlaceResult(
                place_id="p1",
                name="Studio One",
                address="1 First St",
                rating=4.5,
                rating_count=50,
                is_open=True,
            ),
            PlaceResult(
                place_id="p2",
                name="Studio Two",
                address="2 Second St",
                rating=4.0,
            ),
        ]

        output = _format_results_for_agent(results)

        assert "Found 2 fitness studios" in output
        assert "Studio One" in output
        assert "Studio Two" in output
        assert "4.5" in output
        assert "Open now" in output

    def test_format_details_none(self):
        """Test formatting when place is None."""
        output = _format_details_for_agent(None)
        assert "Could not find details" in output

    def test_format_details_with_data(self):
        """Test formatting place details."""
        place = PlaceResult(
            place_id="p1",
            name="Awesome Gym",
            address="123 Gym St",
            rating=4.7,
            rating_count=200,
            is_open=False,
            phone="+1-555-1234",
            website="https://awesomegym.com",
            google_maps_url="https://maps.google.com/awesome",
        )

        output = _format_details_for_agent(place)

        assert "Awesome Gym" in output
        assert "4.7" in output
        assert "200 reviews" in output
        assert "Currently Closed" in output
        assert "+1-555-1234" in output
        assert "https://awesomegym.com" in output

    @pytest.mark.asyncio
    async def test_search_fitness_studios_no_client(self):
        """Test search when Places client is not configured."""
        with patch("src.agent.tools.places.get_places_client", return_value=None):
            result = await search_fitness_studios("yoga studios")

        assert "not currently available" in result

    @pytest.mark.asyncio
    async def test_search_fitness_studios_success(self):
        """Test successful studio search."""
        mock_client = MagicMock()
        mock_client.search_text = AsyncMock(
            return_value=[
                PlaceResult(
                    place_id="test1",
                    name="Test Yoga",
                    address="Test Address",
                    rating=4.5,
                )
            ]
        )

        with patch(
            "src.agent.tools.places.get_places_client",
            return_value=mock_client,
        ):
            result = await search_fitness_studios(
                query="yoga studios",
                location="San Francisco",
            )

        assert "Test Yoga" in result
        assert "4.5" in result

    @pytest.mark.asyncio
    async def test_get_studio_details_success(self):
        """Test successful details retrieval."""
        mock_client = MagicMock()
        mock_client.get_place_details = AsyncMock(
            return_value=PlaceResult(
                place_id="detail1",
                name="Detail Studio",
                address="Detail Address",
                rating=4.8,
                phone="+1-555-0000",
            )
        )

        with patch(
            "src.agent.tools.places.get_places_client",
            return_value=mock_client,
        ):
            result = await get_studio_details("detail1")

        assert "Detail Studio" in result
        assert "4.8" in result
        assert "+1-555-0000" in result


class TestToolRegistration:
    """Tests for tool registration."""

    def test_register_places_tools(self):
        """Test that places tools are registered correctly."""
        # Clear any existing registrations
        tool_registry._tools.clear()

        register_places_tools()

        assert "search_fitness_studios" in tool_registry.list_tools()
        assert "get_studio_details" in tool_registry.list_tools()

    def test_tools_convert_to_langchain(self):
        """Test that registered tools convert to LangChain format."""
        # Ensure tools are registered
        if "search_fitness_studios" not in tool_registry.list_tools():
            register_places_tools()

        lc_tools = tool_registry.to_langchain_tools()

        tool_names = [t.name for t in lc_tools]
        assert "search_fitness_studios" in tool_names
        assert "get_studio_details" in tool_names
