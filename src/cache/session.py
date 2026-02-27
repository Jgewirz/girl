"""Session management for conversation state persistence."""

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.config.logging import get_logger

from .redis import CacheClient, get_cache

logger = get_logger(__name__)

# Key prefixes for Redis
SESSION_PREFIX = "session:"
RATE_LIMIT_PREFIX = "rate:"

# Default TTLs
SESSION_TTL = 60 * 60 * 24 * 7  # 7 days
RATE_LIMIT_WINDOW = 60  # 1 minute


@dataclass
class ConversationSession:
    """User conversation session stored in Redis."""

    telegram_id: int
    messages: list[dict[str, Any]] = field(default_factory=list)

    # User preferences (learned over time)
    first_name: str | None = None
    username: str | None = None
    location: str | None = None
    fitness_goals: list[str] = field(default_factory=list)
    preferred_workout_types: list[str] = field(default_factory=list)

    # Style profile (AI Stylist)
    style_profile: dict[str, Any] = field(default_factory=dict)
    wardrobe: list[dict[str, Any]] = field(default_factory=list)

    # Conversation summary (for context compression)
    conversation_summary: str | None = None

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    message_count: int = 0

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(UTC).isoformat(),
        })
        self.message_count += 1
        self.updated_at = datetime.now(UTC).isoformat()

        # Keep only last 20 messages in active history
        if len(self.messages) > 20:
            self.messages = self.messages[-20:]

    def to_langchain_messages(self) -> list[BaseMessage]:
        """Convert stored messages to LangChain message format."""
        lc_messages: list[BaseMessage] = []
        for msg in self.messages:
            role = msg.get("role", "human")
            content = msg.get("content", "")

            if role == "human":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            elif role == "system":
                lc_messages.append(SystemMessage(content=content))

        return lc_messages

    def to_json(self) -> str:
        """Serialize session to JSON."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "ConversationSession":
        """Deserialize session from JSON."""
        parsed = json.loads(data)
        return cls(**parsed)


class SessionManager:
    """Manages user sessions with rate limiting.

    Works with Redis or in-memory cache transparently.
    """

    def __init__(self, cache_client: CacheClient):
        self._cache = cache_client

    @classmethod
    async def create(cls) -> "SessionManager":
        """Create a SessionManager with appropriate cache backend."""
        client = await get_cache()
        return cls(client)

    def _session_key(self, telegram_id: int) -> str:
        """Generate Redis key for a session."""
        return f"{SESSION_PREFIX}{telegram_id}"

    def _rate_key(self, telegram_id: int) -> str:
        """Generate Redis key for rate limiting."""
        return f"{RATE_LIMIT_PREFIX}{telegram_id}"

    async def get_session(self, telegram_id: int) -> ConversationSession:
        """Get or create a session for a user."""
        key = self._session_key(telegram_id)
        data = await self._cache.get(key)

        if data:
            try:
                session = ConversationSession.from_json(data)
                logger.debug("session_loaded", telegram_id=telegram_id)
                return session
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    "session_parse_error",
                    telegram_id=telegram_id,
                    error=str(e),
                )

        # Create new session
        session = ConversationSession(telegram_id=telegram_id)
        logger.info("session_created", telegram_id=telegram_id)
        return session

    async def save_session(self, session: ConversationSession) -> bool:
        """Save a session to Redis."""
        key = self._session_key(session.telegram_id)
        session.updated_at = datetime.now(UTC).isoformat()

        success = await self._cache.set(
            key,
            session.to_json(),
            ttl_seconds=SESSION_TTL,
        )

        if success:
            logger.debug(
                "session_saved",
                telegram_id=session.telegram_id,
                message_count=session.message_count,
            )
        else:
            logger.error("session_save_failed", telegram_id=session.telegram_id)

        return success

    async def delete_session(self, telegram_id: int) -> bool:
        """Delete a user's session."""
        key = self._session_key(telegram_id)
        success = await self._cache.delete(key)
        if success:
            logger.info("session_deleted", telegram_id=telegram_id)
        return success

    async def update_user_info(
        self,
        telegram_id: int,
        first_name: str | None = None,
        username: str | None = None,
        location: str | None = None,
    ) -> ConversationSession:
        """Update user info in session."""
        session = await self.get_session(telegram_id)

        if first_name:
            session.first_name = first_name
        if username:
            session.username = username
        if location:
            session.location = location

        await self.save_session(session)
        return session

    async def check_rate_limit(
        self,
        telegram_id: int,
        max_requests: int = 30,
        window_seconds: int = RATE_LIMIT_WINDOW,
    ) -> tuple[bool, int]:
        """
        Check if user is within rate limits.

        Returns:
            Tuple of (is_allowed, current_count)
        """
        key = self._rate_key(telegram_id)

        # Increment counter
        count = await self._cache.incr(key)

        if count is None:
            # Redis error - allow request but log warning
            logger.warning("rate_limit_check_failed", telegram_id=telegram_id)
            return True, 0

        # Set expiry on first request in window
        if count == 1:
            await self._cache.expire(key, window_seconds)

        is_allowed = count <= max_requests

        if not is_allowed:
            logger.warning(
                "rate_limit_exceeded",
                telegram_id=telegram_id,
                count=count,
                max_requests=max_requests,
            )

        return is_allowed, count

    async def close(self) -> None:
        """Close the Redis client."""
        await self._cache.close()


# Convenience function for getting a session manager
_session_manager: SessionManager | None = None


async def get_session_manager() -> SessionManager:
    """Get or create the global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = await SessionManager.create()
    return _session_manager
