"""Resilience layer for external service calls.

Provides circuit breakers, retry logic, rate limiting, and graceful degradation
patterns for all external API calls.

Usage:
    from src.services.resilience import resilient_call, create_service_config

    # Configure for a specific service
    config = create_service_config("openai")

    # Make a resilient call
    result, error = await resilient_call(
        func=api_client.some_method,
        args=(arg1, arg2),
        kwargs={"key": "value"},
        config=config,
        fallback=lambda: "Default response",
        cache_key="some:cache:key",
    )
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

import pybreaker
from aiolimiter import AsyncLimiter
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class ErrorCategory(str, Enum):
    """Categories of errors for handling decisions."""

    TRANSIENT = "transient"  # Retry-able (network, timeout)
    RATE_LIMITED = "rate_limited"  # Back off and retry
    CLIENT_ERROR = "client_error"  # Don't retry (bad request)
    SERVER_ERROR = "server_error"  # Retry with backoff
    CONFIGURATION = "configuration"  # Don't retry (missing keys)
    UNKNOWN = "unknown"  # Log and fail gracefully


@dataclass
class ServiceConfig:
    """Configuration for a specific service's resilience behavior."""

    name: str
    circuit_fail_max: int = 5
    circuit_reset_timeout: int = 30
    retry_attempts: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 10.0
    retry_multiplier: float = 2.0
    rate_limit_calls: int = 50
    rate_limit_period: float = 60.0
    timeout_seconds: float = 30.0
    cache_ttl_seconds: int = 300


def _get_default_configs() -> dict[str, ServiceConfig]:
    """Get default service configurations, incorporating settings if available."""
    try:
        from src.config import settings
        circuit_fail_max = settings.resilience_circuit_fail_max
        circuit_reset_timeout = settings.resilience_circuit_reset_timeout
        retry_attempts = settings.resilience_retry_attempts
        openai_rate_limit = settings.resilience_openai_rate_limit
        places_rate_limit = settings.resilience_places_rate_limit
    except Exception:
        # Use defaults if settings not available
        circuit_fail_max = 5
        circuit_reset_timeout = 30
        retry_attempts = 3
        openai_rate_limit = 50
        places_rate_limit = 10

    return {
        "openai": ServiceConfig(
            name="openai",
            circuit_fail_max=circuit_fail_max,
            circuit_reset_timeout=circuit_reset_timeout,
            retry_attempts=retry_attempts,
            retry_min_wait=1.0,
            retry_max_wait=10.0,
            rate_limit_calls=openai_rate_limit,
            rate_limit_period=60.0,
            timeout_seconds=60.0,
            cache_ttl_seconds=0,  # Vision results shouldn't be cached
        ),
        "google_places": ServiceConfig(
            name="google_places",
            circuit_fail_max=circuit_fail_max,
            circuit_reset_timeout=circuit_reset_timeout,
            retry_attempts=retry_attempts,
            retry_min_wait=0.5,
            retry_max_wait=5.0,
            rate_limit_calls=places_rate_limit,
            rate_limit_period=1.0,  # 10 per second
            timeout_seconds=10.0,
            cache_ttl_seconds=1800,  # 30 minutes
        ),
        "amadeus": ServiceConfig(
            name="amadeus",
            circuit_fail_max=circuit_fail_max,
            circuit_reset_timeout=circuit_reset_timeout,
            retry_attempts=retry_attempts,
            retry_min_wait=1.0,
            retry_max_wait=8.0,
            rate_limit_calls=5,
            rate_limit_period=1.0,  # 5 per second
            timeout_seconds=15.0,
            cache_ttl_seconds=300,  # 5 minutes
        ),
        "hotelbeds": ServiceConfig(
            name="hotelbeds",
            circuit_fail_max=circuit_fail_max,
            circuit_reset_timeout=circuit_reset_timeout,
            retry_attempts=retry_attempts,
            retry_min_wait=1.0,
            retry_max_wait=8.0,
            rate_limit_calls=5,
            rate_limit_period=1.0,
            timeout_seconds=15.0,
            cache_ttl_seconds=600,  # 10 minutes
        ),
        "telegram": ServiceConfig(
            name="telegram",
            circuit_fail_max=circuit_fail_max * 2,  # More forgiving for Telegram
            circuit_reset_timeout=circuit_reset_timeout * 2,
            retry_attempts=retry_attempts,
            retry_min_wait=0.5,
            retry_max_wait=5.0,
            rate_limit_calls=30,
            rate_limit_period=1.0,  # 30 per second
            timeout_seconds=30.0,
            cache_ttl_seconds=0,
        ),
    }


# Default configurations for known services (lazy-loaded from settings)
SERVICE_CONFIGS: dict[str, ServiceConfig] = _get_default_configs()


def create_service_config(
    service_name: str,
    **overrides: Any,
) -> ServiceConfig:
    """Create or get a service configuration with optional overrides."""
    base_config = SERVICE_CONFIGS.get(service_name)
    if base_config:
        # Create copy with overrides
        config_dict = {
            "name": base_config.name,
            "circuit_fail_max": base_config.circuit_fail_max,
            "circuit_reset_timeout": base_config.circuit_reset_timeout,
            "retry_attempts": base_config.retry_attempts,
            "retry_min_wait": base_config.retry_min_wait,
            "retry_max_wait": base_config.retry_max_wait,
            "retry_multiplier": base_config.retry_multiplier,
            "rate_limit_calls": base_config.rate_limit_calls,
            "rate_limit_period": base_config.rate_limit_period,
            "timeout_seconds": base_config.timeout_seconds,
            "cache_ttl_seconds": base_config.cache_ttl_seconds,
        }
        config_dict.update(overrides)
        return ServiceConfig(**config_dict)
    else:
        # Create new config with defaults
        return ServiceConfig(name=service_name, **overrides)


# Circuit breaker storage
_circuit_breakers: dict[str, pybreaker.CircuitBreaker] = {}
_rate_limiters: dict[str, AsyncLimiter] = {}


def get_circuit_breaker(config: ServiceConfig) -> pybreaker.CircuitBreaker:
    """Get or create a circuit breaker for a service."""
    if config.name not in _circuit_breakers:
        _circuit_breakers[config.name] = pybreaker.CircuitBreaker(
            fail_max=config.circuit_fail_max,
            reset_timeout=config.circuit_reset_timeout,
            name=config.name,
        )
    return _circuit_breakers[config.name]


def get_rate_limiter(config: ServiceConfig) -> AsyncLimiter:
    """Get or create a rate limiter for a service."""
    if config.name not in _rate_limiters:
        _rate_limiters[config.name] = AsyncLimiter(
            max_rate=config.rate_limit_calls,
            time_period=config.rate_limit_period,
        )
    return _rate_limiters[config.name]


def get_circuit_state(service_name: str) -> CircuitState:
    """Get the current circuit breaker state for a service."""
    if service_name not in _circuit_breakers:
        return CircuitState.CLOSED

    breaker = _circuit_breakers[service_name]
    if breaker.current_state == "closed":
        return CircuitState.CLOSED
    elif breaker.current_state == "open":
        return CircuitState.OPEN
    else:
        return CircuitState.HALF_OPEN


def get_all_circuit_states() -> dict[str, CircuitState]:
    """Get all circuit breaker states for health checks."""
    return {name: get_circuit_state(name) for name in _circuit_breakers}


def categorize_error(error: Exception) -> ErrorCategory:
    """Categorize an error for handling decisions."""
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    # Rate limiting
    if "429" in error_str or "rate limit" in error_str or "too many" in error_str:
        return ErrorCategory.RATE_LIMITED

    # Client errors (don't retry)
    if any(code in error_str for code in ["400", "401", "403", "404"]):
        return ErrorCategory.CLIENT_ERROR

    # Configuration errors
    if "api key" in error_str or "authentication" in error_str or "invalid key" in error_str:
        return ErrorCategory.CONFIGURATION

    # Server errors (retry with backoff)
    if any(code in error_str for code in ["500", "502", "503", "504"]):
        return ErrorCategory.SERVER_ERROR

    # Network/timeout errors (transient, retry)
    if any(
        term in error_str
        for term in ["timeout", "connection", "network", "reset", "refused"]
    ):
        return ErrorCategory.TRANSIENT

    if any(
        term in error_type
        for term in ["timeout", "connection", "network", "http"]
    ):
        return ErrorCategory.TRANSIENT

    return ErrorCategory.UNKNOWN


def should_retry(error: Exception) -> bool:
    """Determine if an error should be retried."""
    category = categorize_error(error)
    return category in (
        ErrorCategory.TRANSIENT,
        ErrorCategory.RATE_LIMITED,
        ErrorCategory.SERVER_ERROR,
    )


@dataclass
class ResilienceResult:
    """Result of a resilient call."""

    success: bool
    value: Any = None
    error: str | None = None
    error_category: ErrorCategory | None = None
    attempts: int = 1
    from_cache: bool = False
    from_fallback: bool = False
    circuit_state: CircuitState = CircuitState.CLOSED


async def resilient_call(
    func: Callable[..., Awaitable[T]],
    config: ServiceConfig,
    args: tuple = (),
    kwargs: dict | None = None,
    fallback: Callable[[], T] | T | None = None,
    cache_get: Callable[[str], Awaitable[T | None]] | None = None,
    cache_set: Callable[[str, T], Awaitable[None]] | None = None,
    cache_key: str | None = None,
) -> ResilienceResult:
    """
    Make a resilient call to an external service.

    Applies circuit breaker, rate limiting, retry logic, and fallback handling.

    Args:
        func: Async function to call
        config: Service configuration
        args: Positional arguments for func
        kwargs: Keyword arguments for func
        fallback: Fallback value or callable if all else fails
        cache_get: Async function to get cached value
        cache_set: Async function to set cached value
        cache_key: Key for caching

    Returns:
        ResilienceResult with success status, value/error, and metadata
    """
    kwargs = kwargs or {}
    breaker = get_circuit_breaker(config)
    limiter = get_rate_limiter(config)

    result = ResilienceResult(
        success=False,
        circuit_state=get_circuit_state(config.name),
    )

    # Check circuit breaker first
    if breaker.current_state == "open":
        logger.warning(
            "circuit_open",
            service=config.name,
            message="Circuit breaker is open, checking cache/fallback",
        )
        result.circuit_state = CircuitState.OPEN

        # Try cache when circuit is open
        if cache_get and cache_key:
            try:
                cached = await cache_get(cache_key)
                if cached is not None:
                    result.success = True
                    result.value = cached
                    result.from_cache = True
                    logger.info(
                        "circuit_open_cache_hit",
                        service=config.name,
                        cache_key=cache_key,
                    )
                    return result
            except Exception as e:
                logger.debug("cache_get_failed", error=str(e))

        # Fall through to fallback
        return _apply_fallback(result, fallback, config.name)

    # Try the actual call with retry logic
    attempts = 0
    last_error: Exception | None = None

    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(config.retry_attempts),
            wait=wait_exponential(
                multiplier=config.retry_multiplier,
                min=config.retry_min_wait,
                max=config.retry_max_wait,
            ),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                attempts += 1

                # Rate limiting
                await limiter.acquire()

                # Make the call with timeout
                try:
                    value = await asyncio.wait_for(
                        breaker.call_async(func, *args, **kwargs),
                        timeout=config.timeout_seconds,
                    )

                    # Success - cache if configured
                    if cache_set and cache_key and config.cache_ttl_seconds > 0:
                        try:
                            await cache_set(cache_key, value)
                        except Exception as e:
                            logger.debug("cache_set_failed", error=str(e))

                    result.success = True
                    result.value = value
                    result.attempts = attempts
                    result.circuit_state = get_circuit_state(config.name)

                    logger.debug(
                        "resilient_call_success",
                        service=config.name,
                        attempts=attempts,
                    )
                    return result

                except asyncio.TimeoutError:
                    raise TimeoutError(f"Call to {config.name} timed out after {config.timeout_seconds}s")

    except pybreaker.CircuitBreakerError:
        # Circuit opened during retries
        result.circuit_state = CircuitState.OPEN
        result.error = "Service temporarily unavailable"
        result.error_category = ErrorCategory.TRANSIENT
        logger.warning("circuit_breaker_tripped", service=config.name)

    except RetryError as e:
        last_error = e.last_attempt.exception()
        result.error = str(last_error)
        result.error_category = categorize_error(last_error)
        result.attempts = attempts
        logger.error(
            "resilient_call_exhausted",
            service=config.name,
            attempts=attempts,
            error=str(last_error),
            category=result.error_category.value,
        )

    except Exception as e:
        last_error = e
        result.error = str(e)
        result.error_category = categorize_error(e)
        result.attempts = attempts
        logger.error(
            "resilient_call_failed",
            service=config.name,
            attempts=attempts,
            error=str(e),
            category=result.error_category.value,
        )

    # Try cache as fallback
    if cache_get and cache_key:
        try:
            cached = await cache_get(cache_key)
            if cached is not None:
                result.success = True
                result.value = cached
                result.from_cache = True
                logger.info(
                    "fallback_to_cache",
                    service=config.name,
                    cache_key=cache_key,
                )
                return result
        except Exception as e:
            logger.debug("fallback_cache_failed", error=str(e))

    # Apply fallback
    return _apply_fallback(result, fallback, config.name)


def _apply_fallback(
    result: ResilienceResult,
    fallback: Callable[[], T] | T | None,
    service_name: str,
) -> ResilienceResult:
    """Apply fallback value or callable."""
    if fallback is not None:
        try:
            if callable(fallback):
                result.value = fallback()
            else:
                result.value = fallback
            result.success = True
            result.from_fallback = True
            logger.info("applied_fallback", service=service_name)
        except Exception as e:
            logger.error("fallback_failed", service=service_name, error=str(e))
            result.error = str(e)

    return result


async def error_chain(
    *callables: Callable[[], Awaitable[T]],
    fallback: T | None = None,
) -> tuple[T | None, str | None]:
    """
    Try a chain of async callables, returning the first success.

    Implements the pattern:
    Primary API -> Cached result -> Alternative API -> Graceful degradation

    Args:
        *callables: Async functions to try in order
        fallback: Final fallback value if all fail

    Returns:
        Tuple of (result, error_message)
    """
    errors: list[str] = []

    for i, call in enumerate(callables):
        try:
            result = await call()
            if result is not None:
                return result, None
        except Exception as e:
            errors.append(f"Attempt {i + 1}: {str(e)}")
            logger.debug(f"error_chain_attempt_{i + 1}", error=str(e))

    if fallback is not None:
        return fallback, None

    error_summary = "; ".join(errors) if errors else "All attempts failed"
    return None, error_summary


def resilient(
    config: ServiceConfig | str,
    fallback: Callable[[], T] | T | None = None,
) -> Callable:
    """
    Decorator to make a function resilient.

    Usage:
        @resilient("openai")
        async def call_openai():
            ...

        @resilient(create_service_config("custom", retry_attempts=5))
        async def call_custom():
            ...
    """
    if isinstance(config, str):
        config = create_service_config(config)

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            result = await resilient_call(
                func=func,
                config=config,
                args=args,
                kwargs=kwargs,
                fallback=fallback,
            )
            if result.success:
                return result.value
            raise RuntimeError(result.error or "Resilient call failed")

        return wrapper

    return decorator


# Health check utilities

@dataclass
class ServiceHealth:
    """Health status for a service."""

    name: str
    circuit_state: CircuitState
    is_healthy: bool
    last_failure: datetime | None = None
    failure_count: int = 0


def get_service_health(service_name: str) -> ServiceHealth:
    """Get health status for a specific service."""
    if service_name not in _circuit_breakers:
        return ServiceHealth(
            name=service_name,
            circuit_state=CircuitState.CLOSED,
            is_healthy=True,
        )

    breaker = _circuit_breakers[service_name]
    state = get_circuit_state(service_name)

    return ServiceHealth(
        name=service_name,
        circuit_state=state,
        is_healthy=state == CircuitState.CLOSED,
        failure_count=breaker.fail_counter if hasattr(breaker, 'fail_counter') else 0,
    )


def get_all_service_health() -> list[ServiceHealth]:
    """Get health status for all configured services."""
    # Include all known services plus any that have been used
    all_services = set(SERVICE_CONFIGS.keys()) | set(_circuit_breakers.keys())
    return [get_service_health(name) for name in sorted(all_services)]


def reset_circuit_breaker(service_name: str) -> bool:
    """Manually reset a circuit breaker (for recovery)."""
    if service_name in _circuit_breakers:
        # pybreaker doesn't have a direct reset, but we can recreate
        config = SERVICE_CONFIGS.get(service_name, ServiceConfig(name=service_name))
        _circuit_breakers[service_name] = pybreaker.CircuitBreaker(
            fail_max=config.circuit_fail_max,
            reset_timeout=config.circuit_reset_timeout,
            name=config.name,
        )
        logger.info("circuit_breaker_reset", service=service_name)
        return True
    return False
