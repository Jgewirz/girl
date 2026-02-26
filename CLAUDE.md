# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GirlBot is a Telegram-based agentic chatbot for fitness and lifestyle assistance. It uses LangGraph for agent orchestration, python-telegram-bot for Telegram integration, and connects to various booking/affiliate APIs (Mindbody, Google Places, affiliate networks).

## Tech Stack

- **Python 3.12+** with async throughout
- **python-telegram-bot v22** - Telegram bot framework with message handlers
- **LangGraph** - Agent orchestration as graph-based state machines
- **LangChain (OpenAI/Anthropic)** - LLM providers (GPT-4o-mini primary, GPT-4o for complex tasks)
- **Supabase** - PostgreSQL + pgvector + Auth (planned)
- **Redis** - Session state, rate limiting (30 req/min), API response caching
- **LangSmith** - Agent tracing and debugging

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the bot
python main.py

# Run tests
pytest
pytest tests/test_specific.py -v          # Single test file
pytest -k "test_name" -v                   # Single test by name

# Type checking
mypy .

# Linting and formatting
ruff check .
ruff check . --fix
black .
```

## Project Structure

```
src/
├── config/
│   ├── settings.py      # Pydantic settings from .env
│   └── logging.py       # Structlog configuration
├── agent/
│   ├── state.py         # AgentState + UserContext dataclasses
│   ├── graph.py         # LangGraph agent with tool support
│   └── tools/
│       ├── registry.py  # Tool registry pattern
│       ├── places.py    # Google Places discovery tool
│       └── preferences.py # User preference learning tools
├── bot/
│   ├── app.py           # Telegram Application factory
│   └── handlers.py      # Command and message handlers
├── cache/
│   ├── redis.py         # Connection pool and RedisClient wrapper
│   └── session.py       # ConversationSession + SessionManager
└── services/
    └── places.py        # Google Places API client
```

## Architecture

```
Telegram Update → Rate Limiter (Redis) → LangGraph Agent Controller
    → LLM classifies intent + selects tool(s)
    → Tool execution (with circuit breaker + cache check)
    → LLM synthesizes response
    → Telegram sends message with inline keyboard
```

### Key Patterns

**Single agent with tool registry** - Not multiple specialized agents. Each API integration registers as a tool with name, description, JSON schema, rate limit config, and fallback behavior. Adding new capabilities means registering one function in `src/agent/tools/registry.py`.

**Redis session management** - Conversation history persisted in Redis keyed by `session:{telegram_id}`. Sessions auto-expire after 7 days. Rate limiting (30 req/min) uses `rate:{telegram_id}` keys with 60-second windows.

**Error handling chain**: Primary API → Cached result → Alternative API → Graceful degradation message

**Context management**: Sliding window (last 20 messages) stored in session. User preferences (location, fitness_goals) persist separately from conversation.

**LLM routing**: GPT-4o-mini for 95% of interactions. Route only complex multi-step planning to GPT-4o/Claude Sonnet.

## Redis Keys

| Pattern | TTL | Purpose |
|---------|-----|---------|
| `session:{telegram_id}` | 7 days | Conversation history + user preferences |
| `rate:{telegram_id}` | 60 seconds | Rate limiting counter (max 30 requests) |
| `places:nearby:{hash}` | 30 minutes | Cached Places search results |
| `places:details:{hash}` | 1 hour | Cached place details |

## Agent Tools

| Tool | Description | API |
|------|-------------|-----|
| `search_fitness_studios` | Search for gyms, yoga studios, fitness centers | Google Places |
| `get_studio_details` | Get details (phone, website, hours) for a specific studio | Google Places |
| `update_user_location` | Save user's location for future searches | Redis Session |
| `update_fitness_goals` | Save fitness goals (lose weight, build strength, etc.) | Redis Session |
| `update_workout_preferences` | Save preferred activities (yoga, pilates, etc.) | Redis Session |

### Adding New Tools

1. Create service client in `src/services/` with caching support
2. Create tool in `src/agent/tools/` with Pydantic input schema
3. Add `register_*_tools()` function and export from `src/agent/tools/__init__.py`
4. Import and call registration in `register_all_tools()` in `src/agent/tools/__init__.py`
5. Add tests in `tests/test_tools_*.py`

## External APIs

| Domain | Primary API | Status | Fallback |
|--------|-------------|--------|----------|
| Discovery | Google Places API | ✅ Implemented | - |
| Fitness booking | Mindbody API (v6 REST) | Planned | Redirect links |
| Fashion/Beauty | CJ Affiliate, Amazon Associates | Planned | Redirect to retailer |
| Flights | Amadeus Self-Service | Planned | Kiwi.com Tequila |
| Restaurants | Google Places (discovery) | ✅ Implemented | OpenTable/Resy deep links |

## Resilience

- **pybreaker** for circuit breakers (open after 5 failures, 30-second reset)
- **aiolimiter** for client-side rate limiting per API
- **tenacity** for retry logic with exponential backoff

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Initialize session, show welcome |
| `/help` | Show capabilities |
| `/settings` | Show user preferences |
| `/clear` | Clear conversation history |

## Environment Variables

See `.env.example` for all required variables. Key ones:
- `TELEGRAM_BOT_TOKEN` - From @BotFather (required)
- `OPENAI_API_KEY` - LLM provider (required)
- `REDIS_URL` - Session cache (defaults to `redis://localhost:6379`)
- `ANTHROPIC_API_KEY` - Alternative LLM (optional)
- `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` - Database (optional, for future persistence)
- `MINDBODY_API_KEY` / `MINDBODY_SITE_ID` - Fitness booking (-99 for sandbox)
- `LANGSMITH_API_KEY` - Tracing (set `LANGCHAIN_TRACING_V2=true`)
