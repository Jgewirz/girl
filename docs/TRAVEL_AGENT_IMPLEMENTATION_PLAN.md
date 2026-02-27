# Travel Agent Implementation Plan for GirlBot

## Executive Summary

This document outlines the implementation plan for adding agentic travel planning capabilities (flights, hotels, trip itineraries) to GirlBot. Based on industry best practices from [McKinsey](https://www.mckinsey.com/industries/travel/our-insights/remapping-travel-with-agentic-ai), [Sabre](https://www.sabre.com/insights/the-agentic-blueprint/), and [Google's Agentic Booking](https://blog.google/products/search/agentic-plans-booking-travel-canvas-ai-mode/), the architecture leverages the existing LangGraph framework with a supervisor pattern for coordinating specialized travel agents.

---

## 1. Architecture Overview

### 1.1 Multi-Agent Supervisor Pattern

Based on [LangGraph multi-agent patterns](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/), we'll implement a **supervisor-worker architecture**:

```
User Request
     │
     ▼
┌─────────────────┐
│ Travel Planner  │  ← Supervisor Agent (orchestrates workflow)
│   Supervisor    │
└────────┬────────┘
         │
    ┌────┴────┬─────────┬──────────┐
    ▼         ▼         ▼          ▼
┌───────┐ ┌───────┐ ┌────────┐ ┌─────────┐
│Flight │ │Hotel  │ │Activity│ │Itinerary│
│Agent  │ │Agent  │ │Agent   │ │Agent    │
└───┬───┘ └───┬───┘ └───┬────┘ └────┬────┘
    │         │         │           │
    ▼         ▼         ▼           ▼
 Amadeus   Hotelbeds  Google     RAG-based
   API       API      Places     Planner
```

### 1.2 Key Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Human-in-the-Loop** | Confirmation required before any booking transaction |
| **Graceful Degradation** | Fallback to deep links if booking APIs unavailable |
| **Budget Awareness** | Budget Officer agent audits all recommendations |
| **Self-Healing** | Autonomous rebooking on disruptions (with user approval) |
| **RAG for Personalization** | Vector DB stores "Travel DNA" for context-aware suggestions |

---

## 2. New Data Models

### 2.1 Travel Profile Extension

Add to `src/agent/state.py`:

```python
@dataclass
class TravelProfile:
    """User's travel preferences and history."""

    # Preferences
    preferred_airlines: list[str] = field(default_factory=list)
    preferred_hotel_chains: list[str] = field(default_factory=list)
    seat_preference: Literal["window", "aisle", "middle", "no_preference"] = "no_preference"
    cabin_class: Literal["economy", "premium_economy", "business", "first"] = "economy"
    hotel_star_rating_min: int = 3
    hotel_amenities_required: list[str] = field(default_factory=list)  # ["wifi", "gym", "pool"]

    # Budget
    daily_budget_usd: float | None = None
    max_flight_price_usd: float | None = None
    max_hotel_price_per_night_usd: float | None = None

    # Loyalty programs
    loyalty_programs: dict[str, str] = field(default_factory=dict)  # {"delta": "123456789"}

    # Dietary / accessibility
    dietary_restrictions: list[str] = field(default_factory=list)
    accessibility_needs: list[str] = field(default_factory=list)

    # Travel history (for RAG personalization)
    past_destinations: list[str] = field(default_factory=list)
    liked_trip_types: list[str] = field(default_factory=list)  # ["beach", "adventure", "cultural"]
    disliked_trip_types: list[str] = field(default_factory=list)


@dataclass
class TripPlan:
    """Active trip being planned."""

    trip_id: str
    status: Literal["planning", "booked", "in_progress", "completed", "cancelled"]

    # Destinations
    origin: str
    destinations: list[str]

    # Dates
    departure_date: date
    return_date: date | None = None
    is_one_way: bool = False
    flexible_dates: bool = False

    # Components
    flights: list["FlightBooking"] = field(default_factory=list)
    hotels: list["HotelBooking"] = field(default_factory=list)
    activities: list["ActivityBooking"] = field(default_factory=list)

    # Budget tracking
    total_budget_usd: float | None = None
    spent_usd: float = 0.0

    # Itinerary
    daily_itinerary: dict[str, list[str]] = field(default_factory=dict)  # {"2025-03-15": ["Flight...", "Check-in..."]}

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_modified: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FlightBooking:
    """Flight segment in a trip."""

    booking_id: str | None = None
    status: Literal["searching", "priced", "held", "booked", "cancelled"] = "searching"

    # Flight details
    airline: str | None = None
    flight_number: str | None = None
    origin_airport: str | None = None
    destination_airport: str | None = None
    departure_time: datetime | None = None
    arrival_time: datetime | None = None
    cabin_class: str | None = None

    # Pricing
    price_usd: float | None = None
    price_currency: str = "USD"

    # Amadeus specifics
    amadeus_offer_id: str | None = None
    amadeus_order_id: str | None = None
    pnr: str | None = None


@dataclass
class HotelBooking:
    """Hotel reservation in a trip."""

    booking_id: str | None = None
    status: Literal["searching", "priced", "held", "booked", "cancelled"] = "searching"

    # Hotel details
    hotel_name: str | None = None
    hotel_chain: str | None = None
    address: str | None = None
    star_rating: int | None = None

    # Stay details
    check_in_date: date | None = None
    check_out_date: date | None = None
    room_type: str | None = None

    # Pricing
    price_per_night_usd: float | None = None
    total_price_usd: float | None = None

    # API specifics
    hotelbeds_rate_key: str | None = None
    confirmation_number: str | None = None


@dataclass
class ActivityBooking:
    """Activity/experience in a trip."""

    activity_id: str
    name: str
    date: date
    time: time | None = None
    location: str
    price_usd: float | None = None
    booking_url: str | None = None
    notes: str | None = None
```

### 2.2 Updated UserContext

```python
@dataclass
class UserContext:
    """Extended with travel profile."""

    # ... existing fields ...

    # Travel (new)
    travel_profile: TravelProfile | None = None
    active_trips: list[TripPlan] = field(default_factory=list)
    passport_countries: list[str] = field(default_factory=list)  # For visa requirements
```

---

## 3. API Integrations

### 3.1 Flight APIs

#### Primary: Amadeus Self-Service API

Based on [Amadeus documentation](https://developers.amadeus.com/self-service/category/flights):

```python
# src/services/flights.py

from dataclasses import dataclass
from enum import Enum

AMADEUS_BASE_URL = "https://api.amadeus.com/v2"

class AmadeusClient:
    """Amadeus Flight API client."""

    def __init__(self, client_id: str, client_secret: str):
        self._client_id = client_id
        self._client_secret = client_secret
        self._access_token: str | None = None
        self._token_expires_at: datetime | None = None

    async def _ensure_auth(self) -> str:
        """OAuth 2.0 authentication."""
        if self._access_token and self._token_expires_at > datetime.utcnow():
            return self._access_token

        # Get new token
        response = await self._http.post(
            "https://api.amadeus.com/v1/security/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            }
        )
        data = response.json()
        self._access_token = data["access_token"]
        self._token_expires_at = datetime.utcnow() + timedelta(seconds=data["expires_in"] - 60)
        return self._access_token

    async def search_flights(
        self,
        origin: str,              # IATA code (LAX)
        destination: str,         # IATA code (JFK)
        departure_date: date,
        return_date: date | None = None,
        adults: int = 1,
        cabin_class: str = "ECONOMY",
        max_results: int = 10,
        max_price: float | None = None,
    ) -> list[FlightOffer]:
        """
        Search for flight offers.

        Best practice: Always call price_offers() before booking
        to get up-to-date pricing.
        """
        token = await self._ensure_auth()

        params = {
            "originLocationCode": origin,
            "destinationLocationCode": destination,
            "departureDate": departure_date.isoformat(),
            "adults": adults,
            "travelClass": cabin_class,
            "max": max_results,
            "currencyCode": "USD",
        }

        if return_date:
            params["returnDate"] = return_date.isoformat()

        if max_price:
            params["maxPrice"] = int(max_price)

        response = await self._http.get(
            f"{AMADEUS_BASE_URL}/shopping/flight-offers",
            params=params,
            headers={"Authorization": f"Bearer {token}"}
        )

        return self._parse_flight_offers(response.json())

    async def price_offers(self, offers: list[FlightOffer]) -> list[FlightOffer]:
        """
        Confirm pricing before booking.

        CRITICAL: Amadeus docs recommend calling this immediately
        before booking to avoid price changes.
        """
        token = await self._ensure_auth()

        response = await self._http.post(
            f"{AMADEUS_BASE_URL}/shopping/flight-offers/pricing",
            json={"data": {"type": "flight-offers-pricing", "flightOffers": [o.raw for o in offers]}},
            headers={"Authorization": f"Bearer {token}"}
        )

        return self._parse_flight_offers(response.json())

    async def create_order(
        self,
        offer: FlightOffer,
        travelers: list[TravelerInfo],
        contact: ContactInfo,
    ) -> FlightOrder:
        """
        Create a flight booking.

        NOTE: For Self-Service API, you need an airline consolidator
        to issue actual tickets. This creates a PNR.
        """
        token = await self._ensure_auth()

        response = await self._http.post(
            f"{AMADEUS_BASE_URL}/booking/flight-orders",
            json={
                "data": {
                    "type": "flight-order",
                    "flightOffers": [offer.raw],
                    "travelers": [t.to_amadeus_format() for t in travelers],
                    "contacts": [contact.to_amadeus_format()],
                }
            },
            headers={"Authorization": f"Bearer {token}"}
        )

        return self._parse_order(response.json())
```

#### Fallback: Kiwi.com Tequila API

For cases where Amadeus doesn't have coverage:

```python
class KiwiClient:
    """Kiwi.com Tequila API as fallback."""

    async def search_flights(self, origin: str, destination: str, ...) -> list[FlightOffer]:
        """Kiwi provides affiliate links, not direct booking."""
        pass
```

### 3.2 Hotel APIs

#### Primary: Hotelbeds APItude

Based on [Hotelbeds documentation](https://developer.hotelbeds.com/documentation/hotels/booking-api/):

```python
# src/services/hotels.py

class HotelbedsClient:
    """Hotelbeds Hotel Booking API client."""

    def __init__(self, api_key: str, secret: str):
        self._api_key = api_key
        self._secret = secret

    def _generate_signature(self) -> str:
        """Generate X-Signature header (API key + secret + timestamp)."""
        import hashlib
        timestamp = str(int(time.time()))
        to_hash = f"{self._api_key}{self._secret}{timestamp}"
        return hashlib.sha256(to_hash.encode()).hexdigest()

    async def search_availability(
        self,
        destination_code: str,      # Hotelbeds destination code
        check_in: date,
        check_out: date,
        rooms: int = 1,
        adults: int = 2,
        children: int = 0,
        min_rating: int = 3,
        max_price: float | None = None,
    ) -> list[HotelAvailability]:
        """
        Search hotel availability.

        Hotelbeds booking flow:
        1. availability (this method) - get available hotels
        2. checkrates (for "recheck" rateTypes) - confirm pricing
        3. booking - create reservation
        """
        response = await self._http.post(
            "https://api.test.hotelbeds.com/hotel-api/1.0/hotels",
            json={
                "stay": {
                    "checkIn": check_in.isoformat(),
                    "checkOut": check_out.isoformat(),
                },
                "occupancies": [{
                    "rooms": rooms,
                    "adults": adults,
                    "children": children,
                }],
                "destination": {"code": destination_code},
                "filter": {
                    "minCategory": min_rating,
                    "maxRate": max_price,
                }
            },
            headers={
                "Api-key": self._api_key,
                "X-Signature": self._generate_signature(),
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

        return self._parse_availability(response.json())

    async def check_rates(self, rate_keys: list[str]) -> list[HotelRate]:
        """
        Recheck rates before booking.

        Required for rates with rateType="RECHECK".
        Returns up-to-date pricing and availability.
        """
        response = await self._http.post(
            "https://api.test.hotelbeds.com/hotel-api/1.0/checkrates",
            json={"rooms": [{"rateKey": key} for key in rate_keys]},
            headers=self._get_headers()
        )
        return self._parse_rates(response.json())

    async def create_booking(
        self,
        rate_key: str,
        holder: GuestInfo,
        rooms: list[RoomGuests],
        client_reference: str,
    ) -> HotelBookingConfirmation:
        """
        Create hotel booking.

        rate_key: From availability or checkrates response
        client_reference: Your internal booking reference
        """
        response = await self._http.post(
            "https://api.test.hotelbeds.com/hotel-api/1.0/bookings",
            json={
                "holder": holder.to_hotelbeds_format(),
                "rooms": [{"rateKey": rate_key, "paxes": r.to_hotelbeds_format()} for r in rooms],
                "clientReference": client_reference,
            },
            headers=self._get_headers()
        )
        return self._parse_booking(response.json())
```

#### Fallback: Booking.com Affiliate Links

```python
def generate_booking_com_link(
    destination: str,
    check_in: date,
    check_out: date,
    adults: int = 2,
) -> str:
    """Generate Booking.com affiliate deep link."""
    base_url = "https://www.booking.com/searchresults.html"
    params = {
        "ss": destination,
        "checkin": check_in.isoformat(),
        "checkout": check_out.isoformat(),
        "group_adults": adults,
        "aid": BOOKING_AFFILIATE_ID,  # Your affiliate ID
    }
    return f"{base_url}?{urlencode(params)}"
```

### 3.3 Destination/Activity APIs

Leverage existing Google Places integration for activities:

```python
# Extend src/services/places.py

ACTIVITY_PLACE_TYPES = {
    "restaurants": ["restaurant", "cafe", "bar"],
    "attractions": ["tourist_attraction", "museum", "art_gallery"],
    "outdoors": ["park", "hiking_area", "beach"],
    "nightlife": ["night_club", "bar", "casino"],
    "shopping": ["shopping_mall", "clothing_store", "market"],
    "spa": ["spa", "beauty_salon"],
}

async def search_activities(
    self,
    destination: str,
    activity_type: str,
    date: date | None = None,
) -> list[ActivityResult]:
    """Search for activities at destination."""
    pass
```

---

## 4. Agent Implementation

### 4.1 Supervisor Agent

```python
# src/agent/travel/supervisor.py

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

TRAVEL_SUPERVISOR_PROMPT = """You are a Travel Planning Supervisor coordinating a team of specialized agents.

Your team:
1. **Flight Agent**: Searches and books flights via Amadeus API
2. **Hotel Agent**: Finds and books accommodations via Hotelbeds
3. **Activity Agent**: Discovers local experiences via Google Places
4. **Itinerary Agent**: Compiles everything into a cohesive day-by-day plan

Your responsibilities:
- Understand the user's travel request and break it into subtasks
- Delegate to the appropriate specialist agent
- Ensure budget constraints are respected (use Budget Officer tool)
- Compile final recommendations for user approval
- NEVER book anything without explicit user confirmation

Workflow for trip planning:
1. Gather requirements (destination, dates, budget, preferences)
2. Search flights → present top 3 options with prices
3. Search hotels → present top 3 options matching dates
4. Search activities → suggest based on user interests
5. Create itinerary → compile into day-by-day plan
6. Confirm with user → only proceed to booking after approval

Remember:
- Present tiered options: Budget / Balanced / Premium
- Always show total trip cost before booking
- Flag any visa requirements or travel advisories
- Respect user's travel profile preferences
"""

class TravelSupervisorState(TypedDict):
    """State for travel planning workflow."""

    messages: Annotated[list[BaseMessage], add_messages]
    user: UserContext
    active_trip: TripPlan | None

    # Subtask results
    flight_options: list[FlightOffer]
    hotel_options: list[HotelAvailability]
    activity_options: list[ActivityResult]

    # Workflow control
    current_phase: Literal["gathering", "searching", "presenting", "confirming", "booking", "complete"]
    awaiting_user_input: bool

    # Budget tracking
    estimated_total: float
    budget_warning: str | None


def create_travel_supervisor() -> CompiledStateGraph:
    """Create the travel planning supervisor graph."""

    graph = StateGraph(TravelSupervisorState)

    # Add specialist agent nodes
    graph.add_node("flight_agent", flight_agent_node)
    graph.add_node("hotel_agent", hotel_agent_node)
    graph.add_node("activity_agent", activity_agent_node)
    graph.add_node("itinerary_agent", itinerary_agent_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("budget_check", budget_check_node)
    graph.add_node("user_confirmation", user_confirmation_node)

    # Define edges
    graph.add_edge(START, "supervisor")

    # Supervisor routes to specialists
    graph.add_conditional_edges(
        "supervisor",
        route_to_specialist,
        {
            "flights": "flight_agent",
            "hotels": "hotel_agent",
            "activities": "activity_agent",
            "itinerary": "itinerary_agent",
            "budget_check": "budget_check",
            "confirm": "user_confirmation",
            "respond": END,
        }
    )

    # Specialists return to supervisor
    for agent in ["flight_agent", "hotel_agent", "activity_agent", "itinerary_agent"]:
        graph.add_edge(agent, "budget_check")

    graph.add_edge("budget_check", "supervisor")
    graph.add_edge("user_confirmation", "supervisor")

    return graph.compile()
```

### 4.2 Flight Agent

```python
# src/agent/travel/flight_agent.py

FLIGHT_AGENT_PROMPT = """You are a Flight Search Specialist.

Your capabilities:
- Search flights using Amadeus API
- Compare prices across airlines
- Find optimal connections for multi-city trips
- Apply user preferences (seat, cabin class, airline)

When searching:
1. Search for exact dates first
2. If flexible_dates=True, also check ±3 days
3. Always show price breakdown (base + taxes)
4. Flag any long layovers (>4 hours)
5. Prefer direct flights when price difference <20%

Output format:
Return top 3 options as:
- Option 1 (Best Value): ...
- Option 2 (Fastest): ...
- Option 3 (Cheapest): ...
"""

async def flight_agent_node(state: TravelSupervisorState) -> dict:
    """Execute flight search and return options."""

    trip = state["active_trip"]
    user_prefs = state["user"].travel_profile

    # Build search parameters
    search_params = {
        "origin": trip.origin,
        "destination": trip.destinations[0],
        "departure_date": trip.departure_date,
        "return_date": trip.return_date,
        "cabin_class": user_prefs.cabin_class if user_prefs else "ECONOMY",
        "max_price": user_prefs.max_flight_price_usd if user_prefs else None,
    }

    # Execute search
    client = await get_amadeus_client()
    offers = await client.search_flights(**search_params)

    # Filter by user preferences
    if user_prefs and user_prefs.preferred_airlines:
        preferred = [o for o in offers if o.airline in user_prefs.preferred_airlines]
        offers = preferred + [o for o in offers if o not in preferred]

    # Sort and select top options
    best_value = min(offers, key=lambda o: o.price * 0.7 + o.duration_minutes * 0.3)
    fastest = min(offers, key=lambda o: o.duration_minutes)
    cheapest = min(offers, key=lambda o: o.price)

    return {
        "flight_options": [best_value, fastest, cheapest],
        "messages": [AIMessage(content=format_flight_options([best_value, fastest, cheapest]))],
    }
```

### 4.3 Hotel Agent

```python
# src/agent/travel/hotel_agent.py

HOTEL_AGENT_PROMPT = """You are a Hotel Booking Specialist.

Your capabilities:
- Search hotels via Hotelbeds API
- Match user preferences (star rating, amenities, location)
- Calculate per-night and total costs
- Check cancellation policies

When searching:
1. Filter by user's minimum star rating
2. Prioritize hotels with required amenities
3. Consider location (city center vs. airport vs. attractions)
4. Always mention cancellation policy
5. Flag any resort fees or hidden charges

Output format:
Return top 3 options as:
- Option 1 (Best Location): ...
- Option 2 (Best Amenities): ...
- Option 3 (Best Price): ...
"""

async def hotel_agent_node(state: TravelSupervisorState) -> dict:
    """Execute hotel search and return options."""

    trip = state["active_trip"]
    user_prefs = state["user"].travel_profile

    # Calculate nights
    nights = (trip.return_date - trip.departure_date).days if trip.return_date else 1

    # Build search
    client = await get_hotelbeds_client()
    availability = await client.search_availability(
        destination_code=get_hotelbeds_destination_code(trip.destinations[0]),
        check_in=trip.departure_date,
        check_out=trip.return_date or trip.departure_date + timedelta(days=1),
        min_rating=user_prefs.hotel_star_rating_min if user_prefs else 3,
        max_price=user_prefs.max_hotel_price_per_night_usd * nights if user_prefs else None,
    )

    # Score and rank
    scored = []
    for hotel in availability:
        score = calculate_hotel_score(
            hotel,
            required_amenities=user_prefs.hotel_amenities_required if user_prefs else [],
            preferred_chains=user_prefs.preferred_hotel_chains if user_prefs else [],
        )
        scored.append((hotel, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_hotels = [h for h, _ in scored[:3]]

    return {
        "hotel_options": top_hotels,
        "messages": [AIMessage(content=format_hotel_options(top_hotels))],
    }
```

### 4.4 Itinerary Agent with RAG

```python
# src/agent/travel/itinerary_agent.py

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

ITINERARY_AGENT_PROMPT = """You are a Travel Itinerary Specialist.

Your capabilities:
- Create day-by-day travel itineraries
- Balance activities with rest time
- Account for travel time between locations
- Personalize based on user's "Travel DNA" (past trips, preferences)

When creating itineraries:
1. First day: Light activities (jet lag recovery)
2. Peak activities: Days 2-3 of trip
3. Last day: Flexible schedule for packing/departure
4. Include meal recommendations
5. Add buffer time between activities
6. Consider opening hours and reservation needs

RAG Context:
Use the vector database to retrieve:
- Similar itineraries from past users
- Destination-specific tips
- Seasonal considerations
"""

class ItineraryAgent:
    """RAG-powered itinerary generation."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore: FAISS | None = None

    async def initialize_rag(self, destination: str):
        """Load destination-specific knowledge."""
        # Load from pre-computed embeddings or build on-demand
        pass

    async def generate_itinerary(
        self,
        trip: TripPlan,
        activities: list[ActivityResult],
        user_profile: TravelProfile,
    ) -> dict[str, list[str]]:
        """Generate day-by-day itinerary."""

        # Retrieve similar itineraries
        similar = await self.vectorstore.asimilarity_search(
            f"itinerary for {trip.destinations[0]} {user_profile.liked_trip_types}",
            k=3
        )

        # Generate with RAG context
        prompt = f"""
        Create a day-by-day itinerary for:
        - Destination: {trip.destinations[0]}
        - Dates: {trip.departure_date} to {trip.return_date}
        - User preferences: {user_profile.liked_trip_types}
        - Available activities: {[a.name for a in activities]}

        Reference itineraries:
        {[doc.page_content for doc in similar]}

        Output as JSON: {{"2025-03-15": ["9:00 AM - Activity 1", "12:00 PM - Lunch at..."], ...}}
        """

        response = await self.llm.ainvoke(prompt)
        return json.loads(response.content)
```

---

## 5. Tool Registration

### 5.1 New Travel Tools

Add to `src/agent/tools/travel.py`:

```python
from pydantic import BaseModel, Field

# === Input Schemas ===

class SearchFlightsInput(BaseModel):
    """Input for flight search."""
    origin: str = Field(description="Origin airport IATA code (e.g., 'LAX')")
    destination: str = Field(description="Destination airport IATA code (e.g., 'JFK')")
    departure_date: str = Field(description="Departure date in YYYY-MM-DD format")
    return_date: str | None = Field(default=None, description="Return date for round trips")
    flexible_dates: bool = Field(default=False, description="Search ±3 days if True")
    cabin_class: str = Field(default="economy", description="Cabin class: economy, premium_economy, business, first")


class SearchHotelsInput(BaseModel):
    """Input for hotel search."""
    destination: str = Field(description="City or destination name")
    check_in: str = Field(description="Check-in date in YYYY-MM-DD format")
    check_out: str = Field(description="Check-out date in YYYY-MM-DD format")
    guests: int = Field(default=2, description="Number of guests")
    min_stars: int = Field(default=3, description="Minimum star rating (1-5)")


class CreateTripInput(BaseModel):
    """Input for creating a new trip plan."""
    origin: str = Field(description="Starting city")
    destinations: list[str] = Field(description="List of destinations to visit")
    departure_date: str = Field(description="Trip start date")
    return_date: str | None = Field(default=None, description="Trip end date")
    budget_usd: float | None = Field(default=None, description="Total trip budget in USD")


class BookFlightInput(BaseModel):
    """Input for booking a flight."""
    trip_id: str = Field(description="Active trip ID")
    offer_index: int = Field(description="Index of selected flight offer (1, 2, or 3)")
    confirm: bool = Field(default=False, description="Must be True to proceed with booking")


class BookHotelInput(BaseModel):
    """Input for booking a hotel."""
    trip_id: str = Field(description="Active trip ID")
    hotel_index: int = Field(description="Index of selected hotel (1, 2, or 3)")
    confirm: bool = Field(default=False, description="Must be True to proceed with booking")


# === Tool Functions ===

async def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str | None = None,
    flexible_dates: bool = False,
    cabin_class: str = "economy",
) -> str:
    """Search for available flights."""
    client = await get_amadeus_client()

    offers = await client.search_flights(
        origin=origin.upper(),
        destination=destination.upper(),
        departure_date=date.fromisoformat(departure_date),
        return_date=date.fromisoformat(return_date) if return_date else None,
        cabin_class=cabin_class.upper(),
    )

    if not offers:
        return f"No flights found from {origin} to {destination} on {departure_date}."

    # Format top 3
    result = f"Found {len(offers)} flights. Top options:\n\n"
    for i, offer in enumerate(offers[:3], 1):
        result += f"**Option {i}**: {offer.airline} {offer.flight_number}\n"
        result += f"  - Departs: {offer.departure_time.strftime('%H:%M')} → Arrives: {offer.arrival_time.strftime('%H:%M')}\n"
        result += f"  - Duration: {offer.duration_minutes // 60}h {offer.duration_minutes % 60}m\n"
        result += f"  - Price: ${offer.price:.2f} USD\n\n"

    return result


async def search_hotels(
    destination: str,
    check_in: str,
    check_out: str,
    guests: int = 2,
    min_stars: int = 3,
) -> str:
    """Search for available hotels."""
    client = await get_hotelbeds_client()

    availability = await client.search_availability(
        destination_code=get_destination_code(destination),
        check_in=date.fromisoformat(check_in),
        check_out=date.fromisoformat(check_out),
        adults=guests,
        min_rating=min_stars,
    )

    if not availability:
        return f"No hotels found in {destination} for those dates."

    nights = (date.fromisoformat(check_out) - date.fromisoformat(check_in)).days

    result = f"Found {len(availability)} hotels in {destination}. Top options:\n\n"
    for i, hotel in enumerate(availability[:3], 1):
        result += f"**Option {i}**: {hotel.name} {'⭐' * hotel.star_rating}\n"
        result += f"  - Location: {hotel.address}\n"
        result += f"  - Price: ${hotel.price_per_night:.2f}/night (${hotel.price_per_night * nights:.2f} total)\n"
        result += f"  - Amenities: {', '.join(hotel.amenities[:5])}\n\n"

    return result


async def create_trip_plan(
    telegram_id: int,
    origin: str,
    destinations: list[str],
    departure_date: str,
    return_date: str | None = None,
    budget_usd: float | None = None,
) -> str:
    """Create a new trip plan."""
    trip = TripPlan(
        trip_id=f"trip_{uuid4().hex[:8]}",
        status="planning",
        origin=origin,
        destinations=destinations,
        departure_date=date.fromisoformat(departure_date),
        return_date=date.fromisoformat(return_date) if return_date else None,
        total_budget_usd=budget_usd,
    )

    # Save to session
    session = await get_session(telegram_id)
    session.active_trips.append(trip)
    await save_session(session)

    return f"Created trip plan {trip.trip_id}:\n" \
           f"- From: {origin}\n" \
           f"- To: {', '.join(destinations)}\n" \
           f"- Dates: {departure_date} → {return_date or 'one-way'}\n" \
           f"- Budget: ${budget_usd:.2f} USD" if budget_usd else ""


async def book_flight(
    telegram_id: int,
    trip_id: str,
    offer_index: int,
    confirm: bool = False,
) -> str:
    """Book a flight from searched options."""
    if not confirm:
        return "⚠️ Booking requires confirmation. Set confirm=True to proceed."

    # Get trip and cached offers
    session = await get_session(telegram_id)
    trip = next((t for t in session.active_trips if t.trip_id == trip_id), None)

    if not trip:
        return f"Trip {trip_id} not found."

    # Execute booking (with consolidator for Self-Service API)
    # ... booking logic ...

    return f"✅ Flight booked! Confirmation: {confirmation_number}"


# === Registration ===

def register_travel_tools(registry: ToolRegistry) -> None:
    """Register all travel planning tools."""

    registry.register(
        name="search_flights",
        description="Search for available flights between airports. Returns top 3 options with prices.",
        func=search_flights,
        args_schema=SearchFlightsInput,
        cache_ttl_seconds=300,  # 5 min cache for flight prices
        timeout_seconds=30.0,
    )

    registry.register(
        name="search_hotels",
        description="Search for available hotels in a destination. Returns top 3 options with prices.",
        func=search_hotels,
        args_schema=SearchHotelsInput,
        cache_ttl_seconds=600,  # 10 min cache for hotel prices
        timeout_seconds=30.0,
    )

    registry.register(
        name="create_trip_plan",
        description="Create a new trip plan to track flights, hotels, and activities.",
        func=create_trip_plan,
        args_schema=CreateTripInput,
    )

    registry.register(
        name="book_flight",
        description="Book a flight from search results. REQUIRES confirm=True parameter.",
        func=book_flight,
        args_schema=BookFlightInput,
    )

    registry.register(
        name="book_hotel",
        description="Book a hotel from search results. REQUIRES confirm=True parameter.",
        func=book_hotel,
        args_schema=BookHotelInput,
    )

    registry.register(
        name="get_trip_itinerary",
        description="Generate a day-by-day itinerary for an active trip.",
        func=get_trip_itinerary,
        args_schema=GetItineraryInput,
    )

    registry.register(
        name="update_travel_preferences",
        description="Save user's travel preferences (airlines, hotels, budget, etc.)",
        func=update_travel_preferences,
        args_schema=UpdateTravelPrefsInput,
    )
```

---

## 6. Error Handling & Resilience

### 6.1 Circuit Breaker Pattern

```python
# src/services/resilience.py

from pybreaker import CircuitBreaker

# Per-API circuit breakers
amadeus_breaker = CircuitBreaker(
    fail_max=5,           # Open after 5 failures
    reset_timeout=30,     # Try again after 30 seconds
    state_storage=redis_storage,
)

hotelbeds_breaker = CircuitBreaker(
    fail_max=5,
    reset_timeout=30,
    state_storage=redis_storage,
)

@amadeus_breaker
async def amadeus_search_with_breaker(client, **params):
    """Flight search with circuit breaker protection."""
    return await client.search_flights(**params)
```

### 6.2 Autonomous Recovery (Self-Healing)

Based on [Sabre's Agentic Blueprint](https://www.sabre.com/insights/the-agentic-blueprint/):

```python
# src/agent/travel/disruption_handler.py

class DisruptionHandler:
    """Handle travel disruptions autonomously."""

    async def detect_disruption(self, booking: FlightBooking) -> Disruption | None:
        """Check for flight disruptions via FlightStats or similar API."""
        pass

    async def find_alternatives(
        self,
        disrupted_booking: FlightBooking,
        user_profile: TravelProfile,
    ) -> list[FlightOffer]:
        """Find rebooking options within user's policy."""
        return await self.amadeus.search_flights(
            origin=disrupted_booking.origin_airport,
            destination=disrupted_booking.destination_airport,
            departure_date=disrupted_booking.departure_time.date(),
            max_price=disrupted_booking.price_usd * 1.5,  # Allow 50% overage
        )

    async def propose_rebooking(
        self,
        disruption: Disruption,
        alternatives: list[FlightOffer],
        trip: TripPlan,
    ) -> str:
        """Create rebooking proposal for user approval."""

        # Calculate hotel impact
        hotel_changes = self._calculate_hotel_impact(disruption, trip)

        message = f"""
⚠️ **Flight Disruption Detected**

Your flight {disruption.original_flight} has been {disruption.type}.

**Recommended Action:**
I found an alternative: {alternatives[0].airline} {alternatives[0].flight_number}
- Departs: {alternatives[0].departure_time}
- Additional cost: ${alternatives[0].price - disruption.original_price:.2f}

{f"Your hotel check-in may need adjustment: {hotel_changes}" if hotel_changes else ""}

Reply "Rebook" to confirm, or "Options" to see more alternatives.
        """

        return message
```

---

## 7. Caching Strategy

### 7.1 Redis Key Structure

```python
# Travel-specific cache keys

CACHE_KEYS = {
    # Flight searches (5 min TTL - prices change frequently)
    "flights:{origin}:{dest}:{date}:{class}": 300,

    # Hotel searches (10 min TTL)
    "hotels:{destination}:{checkin}:{checkout}": 600,

    # Destination info (24 hour TTL)
    "destination:{code}:info": 86400,

    # User's active trip (7 day TTL, refreshed on access)
    "trip:{user_id}:{trip_id}": 604800,

    # Price alerts (no TTL, explicitly managed)
    "alert:{user_id}:{route}": None,

    # RAG embeddings (24 hour TTL)
    "embeddings:{destination}": 86400,
}
```

### 7.2 Cache Invalidation

```python
async def invalidate_on_booking(trip_id: str):
    """Invalidate relevant caches after booking."""
    await redis.delete(f"flights:*")  # Pattern delete
    await redis.delete(f"hotels:*")
    # Keep trip cache
```

---

## 8. Security Considerations

### 8.1 API Key Management

```python
# src/config/settings.py additions

class Settings(BaseSettings):
    # Travel APIs
    amadeus_client_id: SecretStr | None = None
    amadeus_client_secret: SecretStr | None = None
    hotelbeds_api_key: SecretStr | None = None
    hotelbeds_secret: SecretStr | None = None

    # Affiliates
    booking_affiliate_id: str | None = None

    @property
    def has_amadeus(self) -> bool:
        return bool(self.amadeus_client_id and self.amadeus_client_secret)

    @property
    def has_hotelbeds(self) -> bool:
        return bool(self.hotelbeds_api_key and self.hotelbeds_secret)
```

### 8.2 PII Handling

```python
# Never log or cache:
# - Full credit card numbers
# - Passport numbers
# - Date of birth
# - Full addresses

@dataclass
class TravelerInfo:
    """Traveler info with PII protection."""

    first_name: str
    last_name: str
    email: str
    phone: str
    date_of_birth: date  # Required for flights
    passport_number: str | None = None  # Only used at booking time, never cached

    def to_safe_log(self) -> dict:
        """Return loggable version without PII."""
        return {
            "first_name": self.first_name,
            "last_name": self.last_name[0] + "***",
            "email": self.email.split("@")[0][:3] + "***@" + self.email.split("@")[1],
        }
```

---

## 9. Testing Strategy

### 9.1 Mock APIs for Development

```python
# tests/mocks/amadeus_mock.py

class MockAmadeusClient:
    """Mock Amadeus API for testing."""

    async def search_flights(self, **params) -> list[FlightOffer]:
        return [
            FlightOffer(
                offer_id="mock_1",
                airline="Mock Airlines",
                flight_number="MA123",
                price=350.00,
                duration_minutes=180,
                # ...
            ),
            # More mock offers
        ]
```

### 9.2 Integration Tests

```python
# tests/integration/test_travel_agent.py

@pytest.mark.asyncio
async def test_flight_search_tool():
    """Test flight search returns valid results."""
    result = await search_flights(
        origin="LAX",
        destination="JFK",
        departure_date="2025-06-15",
    )

    assert "Option 1" in result
    assert "Price:" in result


@pytest.mark.asyncio
async def test_booking_requires_confirmation():
    """Test that booking without confirm=True fails safely."""
    result = await book_flight(
        telegram_id=12345,
        trip_id="test_trip",
        offer_index=1,
        confirm=False,
    )

    assert "requires confirmation" in result.lower()
```

---

## 10. Implementation Phases

### Phase 1: Foundation (Week 1-2)

- [ ] Add `TravelProfile`, `TripPlan`, `FlightBooking`, `HotelBooking` to `state.py`
- [ ] Create `src/services/flights.py` with Amadeus client (search only)
- [ ] Create `src/services/hotels.py` with Hotelbeds client (search only)
- [ ] Add environment variables to `settings.py`
- [ ] Register `search_flights` and `search_hotels` tools
- [ ] Unit tests with mock APIs

### Phase 2: Trip Planning (Week 3-4)

- [ ] Implement `create_trip_plan` tool
- [ ] Add trip persistence to Redis session
- [ ] Create `TravelSupervisorState` and basic supervisor graph
- [ ] Integrate activity search (extend existing Places)
- [ ] Add budget tracking to supervisor
- [ ] Integration tests

### Phase 3: Booking Flow (Week 5-6)

- [ ] Implement Amadeus `price_offers` and `create_order`
- [ ] Implement Hotelbeds `check_rates` and `create_booking`
- [ ] Add human-in-the-loop confirmation flow
- [ ] Implement `book_flight` and `book_hotel` tools with confirm requirement
- [ ] Add booking error handling and fallback to affiliate links
- [ ] End-to-end tests

### Phase 4: Intelligence (Week 7-8)

- [ ] Implement RAG for itinerary generation (FAISS + embeddings)
- [ ] Build itinerary agent with personalization
- [ ] Add travel profile learning from interactions
- [ ] Implement disruption detection (optional: FlightStats API)
- [ ] Create autonomous rebooking proposal flow
- [ ] Performance optimization

### Phase 5: Polish (Week 9-10)

- [ ] Add Telegram inline keyboards for option selection
- [ ] Implement price alerts (background job)
- [ ] Add trip sharing (generate shareable link)
- [ ] Comprehensive documentation
- [ ] Load testing
- [ ] Production deployment

---

## 11. Cost Analysis

### 11.1 API Costs (Estimated)

| API | Pricing Model | Est. Monthly Cost |
|-----|--------------|-------------------|
| Amadeus Self-Service | €0.001-0.025/call | $50-200 |
| Hotelbeds | Revenue share | Commission-based |
| Google Places | $17/1000 calls | $50-100 |
| OpenAI (GPT-4) | $0.03/1K tokens | $100-300 |
| OpenAI Embeddings | $0.0001/1K tokens | $10-30 |
| Redis (managed) | Per GB | $20-50 |

**Estimated total: $230-680/month** at 1000 users

### 11.2 Cost Optimization

- Cache aggressively (flight prices valid ~5 min)
- Use GPT-4o-mini for 90% of interactions
- Route only complex planning to GPT-4o
- Batch embedding generation for destinations
- Use affiliate links as fallback (generate revenue vs. API cost)

---

## 12. File Structure

```
src/
├── agent/
│   ├── travel/
│   │   ├── __init__.py
│   │   ├── supervisor.py      # Travel planning supervisor graph
│   │   ├── flight_agent.py    # Flight specialist node
│   │   ├── hotel_agent.py     # Hotel specialist node
│   │   ├── activity_agent.py  # Activity specialist node
│   │   ├── itinerary_agent.py # RAG-powered itinerary generator
│   │   └── disruption.py      # Self-healing disruption handler
│   └── tools/
│       └── travel.py          # Travel tool registrations
├── services/
│   ├── flights.py             # Amadeus API client
│   ├── hotels.py              # Hotelbeds API client
│   └── resilience.py          # Circuit breakers, retry logic
├── cache/
│   └── travel_cache.py        # Travel-specific caching
└── config/
    └── settings.py            # Extended with travel API keys
```

---

## References

- [McKinsey: Remapping Travel with Agentic AI](https://www.mckinsey.com/industries/travel/our-insights/remapping-travel-with-agentic-ai)
- [Sabre's Agentic Blueprint](https://www.sabre.com/insights/the-agentic-blueprint/)
- [Google Agentic Travel Booking](https://blog.google/products/search/agentic-plans-booking-travel-canvas-ai-mode/)
- [LangGraph Multi-Agent Tutorial](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/)
- [Amadeus API Documentation](https://developers.amadeus.com/self-service/category/flights)
- [Hotelbeds Booking API](https://developer.hotelbeds.com/documentation/hotels/booking-api/)
- [Agentic RAG Patterns](https://weaviate.io/blog/what-is-agentic-rag)
- [GitHub: langgraph-travel-agent](https://github.com/HarimxChoi/langgraph-travel-agent)
- [AWS: Multi-agent with LangGraph](https://aws.amazon.com/blogs/machine-learning/build-multi-agent-systems-with-langgraph-and-amazon-bedrock/)
