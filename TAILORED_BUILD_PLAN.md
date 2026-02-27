# GirlBot: Tailored Build Plan

> A principle-driven implementation guide that completes the GirlBot system while preserving flexibility for optimal implementation decisions.

---

## Philosophy

This plan provides **outcomes and constraints**, not step-by-step instructions. Each prompt defines:
- **What** needs to exist (deliverables)
- **Why** it matters (context)
- **Boundaries** (must-haves and must-nots)
- **Quality signals** (how to know it's done well)

Implementation details are left to the implementing agent's judgment.

---

## Current State Summary

**Implemented (Do Not Rebuild):**
- LangGraph agent with tool binding (`src/agent/graph.py`)
- Tool registry pattern (`src/agent/tools/registry.py`)
- AI Stylist tools: color analysis, outfit feedback, wardrobe, makeup (`src/agent/tools/stylist.py`)
- Fitness discovery via Google Places (`src/agent/tools/places.py`)
- User preferences persistence (`src/agent/tools/preferences.py`)
- Redis session management with rate limiting (`src/cache/`)
- GPT-4V vision service (`src/services/vision.py`)
- Telegram bot with text handlers (`src/bot/`)
- Pydantic settings and structlog (`src/config/`)
- **[A.1 COMPLETE]** Photo message handling (`src/bot/photo_handler.py`)
- **[A.2 COMPLETE]** Resilience layer (`src/services/resilience.py`)
- **[A.3 COMPLETE]** Configuration & Installation (`src/cli.py`, feature toggles)
- **[A.4 COMPLETE]** Graceful Degradation (`src/agent/fallbacks.py`)

**Gaps to Fill:**
1. ~~Photo message handling in Telegram~~ **DONE**
2. ~~Error resilience layer (circuit breakers, retries, fallbacks)~~ **DONE**
3. ~~Configuration/installation system for end users~~ **DONE**
4. ~~Graceful degradation patterns throughout~~ **DONE**
5. Travel services (flights, hotels) - optional feature module
6. Body type styling data

---

## Module Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│  Telegram Bot ←→ Handlers ←→ Rate Limiter ←→ Session Manager   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      AGENT ORCHESTRATION                        │
│  LangGraph State Machine ←→ Tool Registry ←→ LLM Router        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        TOOL MODULES                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Stylist │ │ Fitness │ │ Travel  │ │  Prefs  │ │ [Custom]│   │
│  │  Tools  │ │  Tools  │ │  Tools  │ │  Tools  │ │  Tools  │   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
└───────┼──────────┼──────────┼──────────┼──────────┼─────────────┘
        ↓          ↓          ↓          ↓          ↓
┌─────────────────────────────────────────────────────────────────┐
│                     SERVICE LAYER                               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐               │
│  │ Vision  │ │ Places  │ │ Flights │ │ Hotels  │  ← Resilience │
│  │ Service │ │ Service │ │ Service │ │ Service │    Wrapper    │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE                               │
│  Redis (Sessions/Cache) │ External APIs │ Configuration        │
└─────────────────────────────────────────────────────────────────┘
```

**Key Principle**: Every vertical slice (Stylist, Fitness, Travel, etc.) should be independently toggleable. A user who only wants the Stylist feature shouldn't need Google Places or Amadeus API keys.

---

## PROMPT PLAN

### PHASE A: Core Completion (Required)

These prompts complete the foundational system.

---

#### Prompt A.1 — Photo Message Handler

**Context:** The Telegram bot handles text messages but photos go unprocessed. The vision service and stylist tools exist and work. This prompt wires them together.

**Deliverables:**
- Photo handler in `src/bot/handlers.py` that:
  - Receives photos from Telegram updates
  - Downloads the highest resolution available
  - Determines user intent (color analysis vs outfit feedback vs wardrobe cataloging)
  - Routes to appropriate stylist tool via the agent
  - Returns formatted response to user

**Boundaries:**
- Must work with existing `process_message()` in `src/agent/graph.py`
- Must handle photos with or without captions
- Must gracefully handle download failures
- Must respect rate limiting already in place
- Photo intent detection can be simple (caption keywords) or sophisticated (ask user) — implementer's choice

**Quality Signals:**
- User sends selfie → gets color season analysis
- User sends outfit photo → gets personalized feedback
- User sends clothing item with "add to wardrobe" → item cataloged
- User sends photo with no caption → bot handles gracefully (either asks or makes reasonable default)
- Download failure → user gets friendly error, not crash

**Error Handling Requirements:**
- Telegram file download timeout → retry once, then apologize
- Photo too large → inform user of size limit
- Vision API failure → use cached analysis if available, otherwise graceful message
- Session unavailable → create ephemeral session, warn about persistence

---

#### Prompt A.2 — Resilience Layer

**Context:** Services call external APIs (OpenAI, Google Places, future Amadeus/Hotelbeds). These can fail, rate limit, or timeout. The system needs defensive patterns.

**Deliverables:**
- `src/services/resilience.py` providing:
  - Circuit breaker factory (configurable thresholds per service)
  - Retry decorator factory (configurable attempts, backoff strategy)
  - Rate limiter factory (configurable requests/period)
  - Unified `resilient_call()` wrapper combining all three
  - Fallback chain helper for graceful degradation

**Boundaries:**
- Must use existing dependencies: `pybreaker`, `tenacity`, `aiolimiter`
- Must be opt-in per service (not forced on everything)
- Must log all failures with structured context
- Must expose circuit state for health checks
- Configuration should come from settings, not hardcoded

**Quality Signals:**
- Service fails 5 times → circuit opens → immediate failures for 30s → half-open → test → close/reopen
- Transient failure → automatic retry with backoff
- Rate limit hit → request queued, not rejected
- All failures logged with: service name, error type, attempt count, circuit state

**Integration Points:**
- Update `src/services/vision.py` to use resilience wrapper
- Update `src/services/places.py` to use resilience wrapper
- Future services should follow the same pattern

**Error Handling Philosophy:**
```
Primary API call
  ↓ fails
Retry with backoff (up to N attempts)
  ↓ still fails
Check circuit breaker (if open, skip to fallback)
  ↓ circuit closed, try anyway
Check cache for stale-but-usable result
  ↓ no cache
Try alternative API (if configured)
  ↓ no alternative or also fails
Return graceful degradation message
  ↓
Log everything, never crash
```

---

#### Prompt A.3 — Configuration & Installation System

**Context:** Users need to install and configure GirlBot without editing source code. Currently, configuration exists but setup is manual.

**Deliverables:**
- `setup.py` or `pyproject.toml` for pip installation
- `girlbot` CLI entry point with subcommands:
  - `girlbot init` — interactive setup wizard
  - `girlbot run` — start the bot
  - `girlbot check` — validate configuration and connectivity
  - `girlbot features` — list available/enabled features
- `.env.example` updated with all variables, grouped by feature
- Feature toggle system in settings:
  - `GIRLBOT_FEATURES=stylist,fitness` (comma-separated)
  - Or individual: `ENABLE_STYLIST=true`, `ENABLE_TRAVEL=false`

**Boundaries:**
- Must work on Windows, macOS, Linux
- Must not require Docker (Docker optional, not required)
- Must provide clear error messages for missing configuration
- Must allow partial configuration (e.g., Stylist only, no fitness)
- Redis must be optional for single-user deployments (fallback to in-memory)

**Quality Signals:**
- Fresh install: `pip install -e .` → `girlbot init` → guided setup → `girlbot run` → working bot
- Missing API key for enabled feature → clear error naming the key and where to get it
- `girlbot check` reports: Redis connection, API key validity, feature status
- User can disable Travel feature → no Amadeus errors even if keys missing

**Configuration Hierarchy:**
```
1. Environment variables (highest priority)
2. .env file in working directory
3. ~/.girlbot/config (user defaults)
4. Built-in defaults (lowest priority)
```

---

#### Prompt A.4 — Graceful Degradation Patterns

**Context:** The system should always respond helpfully, even when services fail. This prompt establishes degradation patterns across all tools.

**Deliverables:**
- Fallback response templates in `src/agent/fallbacks.py`
- Tool-level degradation in each tool module
- Agent-level catch-all for unexpected failures
- User-facing error messages that are helpful, not technical

**Boundaries:**
- Never expose stack traces or internal errors to users
- Always suggest an alternative action when something fails
- Distinguish between "try again later" and "this won't work"
- Log technical details while showing friendly messages

**Degradation Examples:**

| Failure | User Sees | Log Contains |
|---------|-----------|--------------|
| Vision API down | "I can't analyze photos right now. Try describing your outfit in text and I'll help!" | Full error, circuit state |
| Places API quota | "I'm having trouble searching right now. In the meantime, try searching '[query]' on Google Maps." | Quota error, retry-after |
| Redis unavailable | (Works normally but warns) "Note: I might forget our conversation if I restart." | Connection error |
| Unknown tool error | "Something unexpected happened. Let's try a different approach — what are you trying to do?" | Full traceback |

**Quality Signals:**
- Every tool has a defined fallback behavior
- Agent never returns empty/null response
- Errors are categorized: transient vs permanent vs configuration
- User always has a next action suggested

---

### PHASE B: Feature Modules (Optional)

These prompts add optional feature modules. Each is independent.

---

#### Prompt B.1 — Body Type Styling Data

**Context:** The Stylist provides color advice but not silhouette/fit guidance. Body type data enables personalized fit recommendations.

**Deliverables:**
- `src/data/body_types.py` with styling reference data
- Integration with `suggest_outfit` tool
- Optional body type field in `StyleProfile`

**Data Requirements (5 types minimum):**
- Hourglass, Pear, Apple, Rectangle, Inverted Triangle
- Each type needs: description, styling goals, recommended silhouettes, specific garment suggestions
- All language must be body-positive (no "hide" or "disguise" — use "balance" and "showcase")

**Boundaries:**
- Pure data module, no external dependencies
- Body type is optional in profile (many users won't provide)
- Advice should enhance, not replace, color-based suggestions

**Quality Signals:**
- User with body type set gets silhouette advice in outfit suggestions
- User without body type still gets full color-based advice
- Language review: no negative framing of any body type

---

#### Prompt B.2 — Travel Services Foundation

**Context:** Travel planning is a documented future feature. This prompt creates the service layer (not the tools yet).

**Deliverables:**
- `src/services/flights.py` — Amadeus API client
- `src/services/hotels.py` — Hotelbeds API client (or alternative)
- Feature flag: `ENABLE_TRAVEL`
- Affiliate link fallbacks when APIs unavailable

**Boundaries:**
- Must work in sandbox/test mode for development
- Must handle missing API keys gracefully (feature disabled)
- Must include caching (flights: 5min TTL, hotels: 10min TTL)
- Must use resilience layer from A.2

**Quality Signals:**
- With valid keys: real search results
- With sandbox keys: test data, clearly marked
- Without keys: feature disabled, no errors
- API failure: affiliate link fallback (Booking.com, Skyscanner)

**Data Models:**
- `FlightOffer`: airline, route, times, duration, stops, price, booking_url
- `HotelOffer`: name, stars, address, price_per_night, amenities, booking_url
- `TravelProfile`: preferences (airlines, cabin class, hotel stars, budget)
- `TripPlan`: destinations, dates, flights, hotels, activities, status

---

#### Prompt B.3 — Travel Tools & Agent Integration

**Context:** With travel services built, this prompt creates the tools and integrates them with the agent.

**Deliverables:**
- `src/agent/tools/travel.py` with:
  - `search_flights` tool
  - `search_hotels` tool
  - `create_trip` tool
  - `update_travel_preferences` tool
- Travel state in `AgentState` (active trip tracking)
- System prompt additions for travel persona

**Boundaries:**
- Tools must gracefully handle service unavailability
- Trip creation must not auto-book (always require confirmation)
- Budget tracking must be accurate
- Must integrate with existing tool registry pattern

**Quality Signals:**
- "Find flights LAX to Tokyo" → formatted flight options
- "Book the first one" → confirmation prompt, not immediate booking
- Travel feature disabled → agent explains feature not available
- Complex trip → maintains state across multiple messages

---

#### Prompt B.4 — Wardrobe Intelligence Engine

**Context:** Beyond cataloging items, the system should analyze wardrobe composition and generate smart outfit combinations.

**Deliverables:**
- `src/agent/tools/wardrobe_engine.py` with:
  - `analyze_wardrobe` — gaps, color cohesion, versatility score
  - `generate_outfit` — occasion-based outfit from owned items
  - `suggest_additions` — strategic pieces to maximize combinations
- Capsule wardrobe methodology support

**Boundaries:**
- Must work with any wardrobe size (1 item to 100+)
- Must respect user's color season in all suggestions
- Outfit generation must explain reasoning
- Must handle "not enough items" gracefully

**Quality Signals:**
- Small wardrobe (5 items) → "Great start! You could add X to create Y more outfits"
- Color clash in wardrobe → identified and explained
- Outfit request for occasion not covered → honest about limitations + suggestions
- Capsule suggestion → concrete list with reasoning

---

### PHASE C: Production Hardening

These prompts prepare the system for real-world deployment.

---

#### Prompt C.1 — Comprehensive Test Suite

**Context:** Current tests cover ~60% of functionality. Production requires higher coverage and specific edge case testing.

**Deliverables:**
- `tests/test_tools_stylist.py` — all stylist tools
- `tests/test_resilience.py` — circuit breaker, retry, fallback behavior
- `tests/test_handlers_photo.py` — photo processing edge cases
- `tests/test_e2e_flows.py` — complete user journeys
- `tests/test_degradation.py` — failure mode behaviors

**Test Categories:**

**Unit Tests:**
- Each tool with mocked services
- Session serialization/deserialization
- Rate limiting logic
- Configuration validation

**Integration Tests:**
- Agent + tools + mocked services
- Redis session persistence
- Photo download + vision + response

**Edge Case Tests:**
- Empty wardrobe scenarios
- New user (no profile) interactions
- API timeout during multi-step flow
- Concurrent requests from same user
- Very long messages (truncation)
- Unicode/emoji in user input

**Quality Signals:**
- `pytest` passes with no warnings
- Coverage > 80% on core modules
- All failure modes have explicit tests
- Tests run in < 30 seconds (mocked external calls)

---

#### Prompt C.2 — Observability & Health

**Context:** Production systems need monitoring, health checks, and structured logging.

**Deliverables:**
- `/health` endpoint (if running with webhook mode)
- `girlbot status` CLI command showing:
  - Redis connection status
  - Circuit breaker states
  - API key validity
  - Recent error summary
- Structured logging with correlation IDs
- Metrics collection hooks (optional Prometheus/StatsD)

**Boundaries:**
- Health checks must be fast (< 100ms)
- Logs must not contain sensitive data (redact API keys, user photos)
- Must work in polling mode (no HTTP server required)

**Quality Signals:**
- `girlbot status` returns clear, actionable information
- Logs can be parsed by log aggregators (JSON format)
- Each request traceable via correlation ID
- Circuit breaker state visible for debugging

---

#### Prompt C.3 — Documentation & Deployment

**Context:** Users need to deploy the bot. This prompt creates deployment documentation and optional containerization.

**Deliverables:**
- `docs/DEPLOYMENT.md` covering:
  - Local development setup
  - Production deployment options (VPS, Railway, Fly.io, etc.)
  - Environment variable reference
  - Troubleshooting guide
- `Dockerfile` (optional but recommended)
- `docker-compose.yml` with Redis (optional)
- GitHub Actions workflow for CI (optional)

**Boundaries:**
- Docker must be optional, not required
- Documentation must cover non-Docker deployment
- Must address Windows/macOS/Linux differences
- Must explain API key acquisition for each service

**Quality Signals:**
- New user can deploy following only the docs
- Common errors have documented solutions
- Upgrade path documented (how to update safely)
- Backup/restore process for user data (Redis persistence)

---

### PHASE D: Enhancement Modules (Future)

These prompts extend functionality beyond MVP. Include only as needed.

---

#### Prompt D.1 — Style Learning System

**Outcome:** System learns user preferences from interactions over time.

**Key Capabilities:**
- Track outfit feedback (loved/liked/disliked)
- Infer style archetypes from patterns
- Adjust recommendations based on history
- Generate style summary on demand

---

#### Prompt D.2 — Monetization Infrastructure

**Outcome:** Affiliate links and product recommendations.

**Key Capabilities:**
- Amazon Associates integration
- Booking.com affiliate links
- Disclosure compliance
- Revenue tracking (for operator)

---

#### Prompt D.3 — Multi-Language Support

**Outcome:** Bot responds in user's language.

**Key Capabilities:**
- Language detection from user messages
- Response translation
- Localized color names and styling terms
- RTL language support

---

#### Prompt D.4 — Database Persistence Layer

**Outcome:** Long-term storage beyond Redis TTL.

**Key Capabilities:**
- Supabase/PostgreSQL integration
- User profile persistence
- Wardrobe image storage
- Analytics data

---

## Implementation Sequence

### Recommended Order (Minimal Viable Product)

```
A.1 (Photo Handler) ─────────────────────────────┐
                                                 │
A.2 (Resilience Layer) ──────────────────────────┤
                                                 ├─→ Working MVP
A.3 (Configuration/Installation) ────────────────┤
                                                 │
A.4 (Graceful Degradation) ──────────────────────┘

Then:
C.1 (Test Suite) ───→ Production confidence

Then (parallel, as needed):
B.1 (Body Type Data)
B.4 (Wardrobe Intelligence)
C.2 (Observability)
C.3 (Documentation)

Then (if travel feature wanted):
B.2 (Travel Services) ───→ B.3 (Travel Tools)
```

### Alternative: Feature-First Order

If specific features are priority:

```
For Stylist-Only Deployment:
  A.1 → A.2 → A.3 → A.4 → B.1 → B.4 → C.1 → C.3

For Travel-Focused Deployment:
  A.1 → A.2 → A.3 → A.4 → B.2 → B.3 → C.1 → C.3

For Full Feature Set:
  All prompts in recommended order
```

---

## Cross-Cutting Concerns

Apply these principles to ALL prompts:

### Error Handling

Every function should handle:
1. **Expected errors** — API failures, validation errors, missing data
2. **Unexpected errors** — catch-all that logs and returns graceful response
3. **Partial failures** — some data available, some not

Pattern:
```python
async def some_tool(input: Input) -> str:
    try:
        result = await service.call(input)
        return format_success(result)
    except ServiceUnavailable as e:
        logger.warning("service_unavailable", error=str(e))
        return fallback_response(input)
    except ValidationError as e:
        logger.info("validation_error", error=str(e))
        return user_friendly_validation_message(e)
    except Exception as e:
        logger.exception("unexpected_error")
        return "Something unexpected happened. Let me try a different approach."
```

### Configuration

Every feature should:
1. Have an enable/disable flag
2. Work with minimal configuration
3. Validate configuration at startup
4. Provide clear errors for missing required config

### Logging

Every significant action should log:
1. What happened (action name)
2. Who triggered it (user_id, anonymized)
3. Relevant context (tool name, input summary)
4. Outcome (success/failure, duration)

Never log:
- Full API keys (redact to last 4 chars)
- Photo data
- Personal user information beyond ID

### Testing

Every new code path should have:
1. Happy path test
2. Failure mode test
3. Edge case test (empty input, max size input, etc.)

---

## Quality Checklist

Before considering any prompt complete:

- [ ] All new functions have type hints
- [ ] All new functions have docstrings
- [ ] Error cases return user-friendly messages
- [ ] Logs use structured format with context
- [ ] Tests cover happy path and main failure modes
- [ ] Configuration is documented in `.env.example`
- [ ] Feature can be disabled without errors
- [ ] No hardcoded API keys or secrets
- [ ] Works offline/degraded when external services unavailable

---

## File Quick Reference

```
src/
├── agent/
│   ├── graph.py          # LangGraph state machine [EXISTS]
│   ├── state.py          # Data models [EXISTS]
│   ├── fallbacks.py      # Degradation responses [EXISTS]
│   └── tools/
│       ├── registry.py   # Tool registration [EXISTS]
│       ├── stylist.py    # AI Stylist [EXISTS]
│       ├── places.py     # Fitness discovery [EXISTS]
│       ├── preferences.py # User prefs [EXISTS]
│       ├── travel.py     # Travel tools [B.3]
│       └── wardrobe_engine.py # Wardrobe intelligence [B.4]
├── bot/
│   ├── app.py            # Application factory [EXISTS]
│   └── handlers.py       # Telegram handlers [EXISTS, needs A.1]
├── cache/
│   ├── redis.py          # Redis client [EXISTS]
│   └── session.py        # Session management [EXISTS]
├── config/
│   ├── settings.py       # Pydantic settings [EXISTS, expand A.3]
│   └── logging.py        # Structlog config [EXISTS]
├── data/
│   └── body_types.py     # Body type reference [B.1]
├── services/
│   ├── resilience.py     # Circuit breakers, retry [A.2]
│   ├── vision.py         # GPT-4V [EXISTS]
│   ├── places.py         # Google Places [EXISTS]
│   ├── flights.py        # Amadeus [B.2]
│   └── hotels.py         # Hotelbeds [B.2]
├── cli.py                # CLI commands [A.3]
└── __main__.py           # Entry point [A.3]

tests/
├── conftest.py           # Fixtures [EXISTS]
├── test_*.py             # Various [EXISTS + C.1]

docs/
├── DEPLOYMENT.md         # Deployment guide [C.3]
└── TAILORED_BUILD_PLAN.md # This document
```

---

*This plan provides structure without constraining implementation. Each prompt defines outcomes and constraints, leaving tactical decisions to the implementer's expertise.*
