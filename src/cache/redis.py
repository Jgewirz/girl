"""Redis connection management with connection pooling and in-memory fallback."""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import redis.asyncio as redis
from redis.asyncio import ConnectionPool, Redis

from src.config import settings
from src.config.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class CacheClient(Protocol):
    """Protocol for cache clients (Redis or Memory)."""

    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> bool: ...
    async def delete(self, key: str) -> bool: ...
    async def exists(self, key: str) -> bool: ...
    async def incr(self, key: str) -> int | None: ...
    async def expire(self, key: str, ttl_seconds: int) -> bool: ...
    async def ping(self) -> bool: ...
    async def close(self) -> None: ...


@dataclass
class CacheEntry:
    """Entry in the memory cache with optional expiration."""
    value: str
    expires_at: float | None = None

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class MemoryCache:
    """In-memory cache for single-user deployments when Redis is unavailable.

    This provides the same interface as RedisClient but stores everything in memory.
    Data is lost on restart. Suitable for development or single-user deployments.
    """

    def __init__(self):
        self._data: dict[str, CacheEntry] = {}
        self._counters: dict[str, int] = {}
        self._lock = asyncio.Lock()
        logger.warning("memory_cache_initialized", message="Using in-memory cache - data will not persist")

    async def get(self, key: str) -> str | None:
        """Get a value by key."""
        async with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry.is_expired():
                del self._data[key]
                return None
            return entry.value

    async def set(self, key: str, value: str, ttl_seconds: int | None = None) -> bool:
        """Set a value with optional TTL."""
        async with self._lock:
            expires_at = time.time() + ttl_seconds if ttl_seconds else None
            self._data[key] = CacheEntry(value=value, expires_at=expires_at)
            return True

    async def delete(self, key: str) -> bool:
        """Delete a key."""
        async with self._lock:
            if key in self._data:
                del self._data[key]
            if key in self._counters:
                del self._counters[key]
            return True

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        async with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._data[key]
                return False
            return True

    async def incr(self, key: str) -> int | None:
        """Increment a counter."""
        async with self._lock:
            self._counters[key] = self._counters.get(key, 0) + 1
            return self._counters[key]

    async def expire(self, key: str, ttl_seconds: int) -> bool:
        """Set expiration on a key."""
        async with self._lock:
            entry = self._data.get(key)
            if entry:
                entry.expires_at = time.time() + ttl_seconds
                return True
            if key in self._counters:
                # For counters, create a cache entry to track expiration
                self._data[key] = CacheEntry(
                    value=str(self._counters[key]),
                    expires_at=time.time() + ttl_seconds
                )
                return True
            return False

    async def ping(self) -> bool:
        """Always returns True for memory cache."""
        return True

    async def close(self) -> None:
        """Clear all data."""
        async with self._lock:
            self._data.clear()
            self._counters.clear()

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        now = time.time()
        expired_keys = [
            k for k, v in self._data.items()
            if v.expires_at and v.expires_at < now
        ]
        for key in expired_keys:
            del self._data[key]
        return len(expired_keys)

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


# Global cache instance (can be Redis or Memory)
_cache_instance: CacheClient | None = None
_cache_lock = asyncio.Lock()


async def get_cache() -> CacheClient:
    """Get a cache client, with automatic fallback to in-memory if Redis unavailable.

    Uses the following logic:
    1. If USE_MEMORY_CACHE=true, always use MemoryCache
    2. Otherwise, try to connect to Redis
    3. If Redis fails, fall back to MemoryCache

    Returns:
        CacheClient instance (RedisClient or MemoryCache)
    """
    global _cache_instance

    if _cache_instance is not None:
        return _cache_instance

    async with _cache_lock:
        # Double-check after acquiring lock
        if _cache_instance is not None:
            return _cache_instance

        # Check if user explicitly wants memory cache
        if settings.use_memory_cache:
            logger.info("using_memory_cache", reason="explicitly configured")
            _cache_instance = MemoryCache()
            return _cache_instance

        # Try to connect to Redis
        try:
            redis_client = await RedisClient.create()
            if await redis_client.ping():
                logger.info("redis_connected")
                _cache_instance = redis_client
                return _cache_instance
        except Exception as e:
            logger.warning("redis_connection_failed", error=str(e))

        # Fall back to memory cache
        logger.info("using_memory_cache", reason="redis unavailable")
        _cache_instance = MemoryCache()
        return _cache_instance


async def reset_cache() -> None:
    """Reset the global cache instance (for testing)."""
    global _cache_instance
    if _cache_instance is not None:
        await _cache_instance.close()
        _cache_instance = None
