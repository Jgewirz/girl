"""Tests for the resilience layer."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.services.resilience import (
    CircuitState,
    ErrorCategory,
    ResilienceResult,
    ServiceConfig,
    categorize_error,
    create_service_config,
    error_chain,
    get_all_circuit_states,
    get_all_service_health,
    get_circuit_breaker,
    get_circuit_state,
    get_rate_limiter,
    get_service_health,
    reset_circuit_breaker,
    resilient,
    resilient_call,
    should_retry,
)


class TestServiceConfig:
    """Tests for service configuration."""

    def test_create_default_config(self):
        """Test creating a config with defaults."""
        config = create_service_config("unknown_service")
        assert config.name == "unknown_service"
        assert config.circuit_fail_max == 5
        assert config.retry_attempts == 3

    def test_create_known_config(self):
        """Test creating a config for a known service."""
        config = create_service_config("openai")
        assert config.name == "openai"
        assert config.timeout_seconds == 60.0

    def test_create_config_with_overrides(self):
        """Test creating a config with custom overrides."""
        config = create_service_config(
            "openai",
            retry_attempts=5,
            timeout_seconds=120.0,
        )
        assert config.retry_attempts == 5
        assert config.timeout_seconds == 120.0
        # Other values from default
        assert config.circuit_fail_max == 5


class TestErrorCategorization:
    """Tests for error categorization."""

    def test_rate_limited_error(self):
        """Test categorizing rate limit errors."""
        assert categorize_error(Exception("429 Too Many Requests")) == ErrorCategory.RATE_LIMITED
        assert categorize_error(Exception("rate limit exceeded")) == ErrorCategory.RATE_LIMITED

    def test_client_error(self):
        """Test categorizing client errors."""
        assert categorize_error(Exception("400 Bad Request")) == ErrorCategory.CLIENT_ERROR
        assert categorize_error(Exception("401 Unauthorized")) == ErrorCategory.CLIENT_ERROR
        assert categorize_error(Exception("404 Not Found")) == ErrorCategory.CLIENT_ERROR

    def test_server_error(self):
        """Test categorizing server errors."""
        assert categorize_error(Exception("500 Internal Server Error")) == ErrorCategory.SERVER_ERROR
        assert categorize_error(Exception("503 Service Unavailable")) == ErrorCategory.SERVER_ERROR

    def test_transient_error(self):
        """Test categorizing transient errors."""
        assert categorize_error(Exception("Connection timeout")) == ErrorCategory.TRANSIENT
        assert categorize_error(Exception("Network error")) == ErrorCategory.TRANSIENT

    def test_configuration_error(self):
        """Test categorizing configuration errors."""
        assert categorize_error(Exception("Invalid API key")) == ErrorCategory.CONFIGURATION
        assert categorize_error(Exception("Authentication failed")) == ErrorCategory.CONFIGURATION

    def test_unknown_error(self):
        """Test categorizing unknown errors."""
        assert categorize_error(Exception("Something weird happened")) == ErrorCategory.UNKNOWN


class TestShouldRetry:
    """Tests for retry decision logic."""

    def test_should_retry_transient(self):
        """Test that transient errors should be retried."""
        assert should_retry(Exception("Connection timeout")) is True

    def test_should_retry_rate_limited(self):
        """Test that rate limit errors should be retried."""
        assert should_retry(Exception("429 Too Many Requests")) is True

    def test_should_retry_server_error(self):
        """Test that server errors should be retried."""
        assert should_retry(Exception("500 Internal Server Error")) is True

    def test_should_not_retry_client_error(self):
        """Test that client errors should not be retried."""
        assert should_retry(Exception("400 Bad Request")) is False

    def test_should_not_retry_config_error(self):
        """Test that configuration errors should not be retried."""
        assert should_retry(Exception("Invalid API key")) is False


class TestCircuitBreaker:
    """Tests for circuit breaker functionality."""

    def test_get_circuit_breaker(self):
        """Test getting a circuit breaker."""
        config = create_service_config("test_service_cb")
        breaker = get_circuit_breaker(config)
        assert breaker is not None
        assert breaker.name == "test_service_cb"

    def test_get_same_circuit_breaker(self):
        """Test that same config returns same breaker."""
        config = create_service_config("test_service_same")
        breaker1 = get_circuit_breaker(config)
        breaker2 = get_circuit_breaker(config)
        assert breaker1 is breaker2

    def test_get_circuit_state_initial(self):
        """Test that initial circuit state is closed."""
        state = get_circuit_state("nonexistent_service")
        assert state == CircuitState.CLOSED

    def test_reset_circuit_breaker(self):
        """Test resetting a circuit breaker."""
        config = create_service_config("test_reset")
        get_circuit_breaker(config)
        result = reset_circuit_breaker("test_reset")
        assert result is True

    def test_reset_nonexistent_breaker(self):
        """Test resetting a nonexistent breaker."""
        result = reset_circuit_breaker("never_existed_service")
        assert result is False


class TestRateLimiter:
    """Tests for rate limiter functionality."""

    def test_get_rate_limiter(self):
        """Test getting a rate limiter."""
        config = create_service_config("test_rate_limit")
        limiter = get_rate_limiter(config)
        assert limiter is not None

    def test_get_same_rate_limiter(self):
        """Test that same config returns same limiter."""
        config = create_service_config("test_rate_same")
        limiter1 = get_rate_limiter(config)
        limiter2 = get_rate_limiter(config)
        assert limiter1 is limiter2


class TestResilientCall:
    """Tests for the resilient_call function."""

    @pytest.mark.asyncio
    async def test_successful_call(self):
        """Test a successful resilient call."""
        async def success_func():
            return "success"

        config = create_service_config("test_success")
        result = await resilient_call(func=success_func, config=config)

        assert result.success is True
        assert result.value == "success"
        assert result.error is None
        assert result.from_cache is False
        assert result.from_fallback is False

    @pytest.mark.asyncio
    async def test_call_with_fallback(self):
        """Test that fallback is used on failure."""
        async def failing_func():
            raise Exception("Always fails")

        config = create_service_config("test_fallback", retry_attempts=1)
        result = await resilient_call(
            func=failing_func,
            config=config,
            fallback="fallback_value",
        )

        assert result.success is True
        assert result.value == "fallback_value"
        assert result.from_fallback is True

    @pytest.mark.asyncio
    async def test_call_with_callable_fallback(self):
        """Test that callable fallback is invoked."""
        async def failing_func():
            raise Exception("Always fails")

        config = create_service_config("test_callable_fb", retry_attempts=1)
        result = await resilient_call(
            func=failing_func,
            config=config,
            fallback=lambda: "computed_fallback",
        )

        assert result.success is True
        assert result.value == "computed_fallback"
        assert result.from_fallback is True

    @pytest.mark.asyncio
    async def test_call_with_cache_hit(self):
        """Test that cached values are returned."""
        async def api_func():
            return "from_api"

        async def cache_get(key):
            return "from_cache"

        async def cache_set(key, value):
            pass

        config = create_service_config("test_cache_hit")
        # Simulate circuit open to force cache check
        breaker = get_circuit_breaker(config)
        # Force circuit open by hitting fail threshold
        for _ in range(config.circuit_fail_max):
            try:
                breaker.call(lambda: (_ for _ in ()).throw(Exception("fail")))
            except Exception:
                pass

        result = await resilient_call(
            func=api_func,
            config=config,
            cache_get=cache_get,
            cache_set=cache_set,
            cache_key="test_key",
        )

        assert result.success is True
        assert result.value == "from_cache"
        assert result.from_cache is True

    @pytest.mark.asyncio
    async def test_call_caches_successful_result(self):
        """Test that successful results are cached."""
        cached_data = {}

        async def api_func():
            return "api_result"

        async def cache_get(key):
            return cached_data.get(key)

        async def cache_set(key, value):
            cached_data[key] = value

        config = create_service_config("test_cache_set", cache_ttl_seconds=300)
        result = await resilient_call(
            func=api_func,
            config=config,
            cache_get=cache_get,
            cache_set=cache_set,
            cache_key="test_key_set",
        )

        assert result.success is True
        assert cached_data.get("test_key_set") == "api_result"

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self):
        """Test that transient failures trigger retries."""
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Connection timeout")
            return "success_after_retry"

        config = create_service_config("test_retry", retry_attempts=3)
        result = await resilient_call(func=flaky_func, config=config)

        assert result.success is True
        assert result.value == "success_after_retry"
        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_exhausted_retries(self):
        """Test behavior when all retries are exhausted."""
        async def always_fails():
            raise Exception("Persistent failure")

        config = create_service_config("test_exhausted", retry_attempts=2)
        result = await resilient_call(func=always_fails, config=config)

        assert result.success is False
        assert "Persistent failure" in (result.error or "")
        assert result.attempts == 2


class TestErrorChain:
    """Tests for the error_chain function."""

    @pytest.mark.asyncio
    async def test_first_succeeds(self):
        """Test that first successful call wins."""
        async def first():
            return "first_result"

        async def second():
            return "second_result"

        result, error = await error_chain(first, second)
        assert result == "first_result"
        assert error is None

    @pytest.mark.asyncio
    async def test_fallback_on_first_failure(self):
        """Test fallback when first fails."""
        async def first():
            raise Exception("First failed")

        async def second():
            return "second_result"

        result, error = await error_chain(first, second)
        assert result == "second_result"
        assert error is None

    @pytest.mark.asyncio
    async def test_all_fail_with_fallback(self):
        """Test fallback value when all fail."""
        async def first():
            raise Exception("First failed")

        async def second():
            raise Exception("Second failed")

        result, error = await error_chain(first, second, fallback="final_fallback")
        assert result == "final_fallback"
        assert error is None

    @pytest.mark.asyncio
    async def test_all_fail_no_fallback(self):
        """Test error message when all fail with no fallback."""
        async def first():
            raise Exception("First failed")

        async def second():
            raise Exception("Second failed")

        result, error = await error_chain(first, second)
        assert result is None
        assert error is not None
        assert "Attempt 1" in error
        assert "Attempt 2" in error


class TestResilientDecorator:
    """Tests for the @resilient decorator."""

    @pytest.mark.asyncio
    async def test_decorator_success(self):
        """Test decorator with successful function."""
        @resilient("test_decorator_success")
        async def my_func():
            return "decorated_result"

        result = await my_func()
        assert result == "decorated_result"

    @pytest.mark.asyncio
    async def test_decorator_with_fallback(self):
        """Test decorator with fallback on failure."""
        call_count = 0

        @resilient(
            create_service_config("test_decorator_fallback", retry_attempts=1),
            fallback="fallback_value",
        )
        async def failing_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Always fails")

        result = await failing_func()
        assert result == "fallback_value"

    @pytest.mark.asyncio
    async def test_decorator_raises_on_failure_no_fallback(self):
        """Test decorator raises when no fallback provided."""
        @resilient(create_service_config("test_decorator_raise", retry_attempts=1))
        async def failing_func():
            raise Exception("Always fails")

        with pytest.raises(RuntimeError):
            await failing_func()


class TestHealthChecks:
    """Tests for health check functionality."""

    def test_get_service_health_unknown(self):
        """Test health check for unknown service."""
        health = get_service_health("unknown_health_service")
        assert health.name == "unknown_health_service"
        assert health.is_healthy is True
        assert health.circuit_state == CircuitState.CLOSED

    def test_get_service_health_known(self):
        """Test health check for a known service."""
        config = create_service_config("health_check_known")
        get_circuit_breaker(config)

        health = get_service_health("health_check_known")
        assert health.name == "health_check_known"
        assert health.circuit_state == CircuitState.CLOSED

    def test_get_all_service_health(self):
        """Test getting health for all services."""
        config = create_service_config("health_all_test")
        get_circuit_breaker(config)

        all_health = get_all_service_health()
        assert isinstance(all_health, list)
        # Should include at least the service we just created
        names = [h.name for h in all_health]
        assert "health_all_test" in names

    def test_get_all_circuit_states(self):
        """Test getting all circuit states."""
        config = create_service_config("circuit_states_test")
        get_circuit_breaker(config)

        states = get_all_circuit_states()
        assert isinstance(states, dict)
        assert "circuit_states_test" in states
        assert states["circuit_states_test"] == CircuitState.CLOSED


class TestResilienceResult:
    """Tests for ResilienceResult dataclass."""

    def test_default_values(self):
        """Test default values of ResilienceResult."""
        result = ResilienceResult(success=False)
        assert result.value is None
        assert result.error is None
        assert result.error_category is None
        assert result.attempts == 1
        assert result.from_cache is False
        assert result.from_fallback is False
        assert result.circuit_state == CircuitState.CLOSED

    def test_with_all_values(self):
        """Test ResilienceResult with all values set."""
        result = ResilienceResult(
            success=True,
            value="test_value",
            error=None,
            error_category=None,
            attempts=3,
            from_cache=True,
            from_fallback=False,
            circuit_state=CircuitState.HALF_OPEN,
        )
        assert result.success is True
        assert result.value == "test_value"
        assert result.attempts == 3
        assert result.from_cache is True
        assert result.circuit_state == CircuitState.HALF_OPEN
