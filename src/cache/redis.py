"""Redis connection management with connection pooling."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import redis.asyncio as redis
from redis.asyncio import ConnectionPool, Redis

from src.config import settings
from src.config.logging import get_logger

logger = get_logger(__name__)

# Global connection pool (initialized lazily)
_pool: ConnectionPool | None = None
_pool_lock = asyncio.Lock()


async def _get_pool() -> ConnectionPool:
    """Get or create the Redis connection pool."""
    global _pool

    if _pool is None:
        async with _pool_lock:
            # Double-check after acquiring lock
            if _pool is None:
                redis_url = settings.redis_url.get_secret_value()
                _pool = ConnectionPool.from_url(
                    redis_url,
                    max_connections=20,
                    decode_responses=True,
                    socket_timeout=5.0,
                    socket_connect_timeout=5.0,
                    retry_on_timeout=True,
                )
                logger.info("redis_pool_created", max_connections=20)

    return _pool


async def get_redis() -> Redis:
    """Get a Redis client from the connection pool."""
    pool = await _get_pool()
    return Redis(connection_pool=pool)


@asynccontextmanager
async def redis_client() -> AsyncIterator[Redis]:
    """Context manager for Redis client with automatic cleanup."""
    client = await get_redis()
    try:
        yield client
    finally:
        await client.aclose()


class RedisClient:
    """Redis client wrapper with common operations and error handling."""

    def __init__(self, client: Redis):
        self._client = client

    @classmethod
    async def create(cls) -> "RedisClient":
        """Create a new RedisClient instance."""
        client = await get_redis()
        return cls(client)

    async def get(self, key: str) -> str | None:
        """Get a value by key."""
        try:
            return await self._client.get(key)
        except redis.RedisError as e:
            logger.error("redis_get_error", key=key, error=str(e))
            return None

    async def set(
        self,
        key: str,
        value: str,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Set a value with optional TTL."""
        try:
            if ttl_seconds:
                await self._client.setex(key, ttl_seconds, value)
            else:
                await self._client.set(key, value)
            return True
        except redis.RedisError as e:
            logger.error("redis_set_error", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """Delete a key."""
        try:
            await self._client.delete(key)
            return True
        except redis.RedisError as e:
            logger.error("redis_delete_error", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        try:
            return bool(await self._client.exists(key))
        except redis.RedisError as e:
            logger.error("redis_exists_error", key=key, error=str(e))
            return False

    async def incr(self, key: str) -> int | None:
        """Increment a counter."""
        try:
            return await self._client.incr(key)
        except redis.RedisError as e:
            logger.error("redis_incr_error", key=key, error=str(e))
            return None

    async def expire(self, key: str, ttl_seconds: int) -> bool:
        """Set expiration on a key."""
        try:
            return bool(await self._client.expire(key, ttl_seconds))
        except redis.RedisError as e:
            logger.error("redis_expire_error", key=key, error=str(e))
            return False

    async def ping(self) -> bool:
        """Check if Redis is reachable."""
        try:
            return await self._client.ping()
        except redis.RedisError as e:
            logger.error("redis_ping_error", error=str(e))
            return False

    async def close(self) -> None:
        """Close the client connection."""
        await self._client.aclose()


async def close_pool() -> None:
    """Close the global connection pool (call on shutdown)."""
    global _pool
    if _pool is not None:
        await _pool.disconnect()
        _pool = None
        logger.info("redis_pool_closed")
