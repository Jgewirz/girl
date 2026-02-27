"""Tests for graceful degradation and fallback responses."""

import pytest

from src.agent.fallbacks import (
    FALLBACKS,
    ErrorCategory,
    FallbackResponse,
    FallbackType,
    categorize_exception,
    format_with_alternative,
    get_fallback,
    get_fallback_response,
    places_fallback,
    tool_fallback,
    unknown_fallback,
    vision_fallback,
)


class TestFallbackTypes:
    """Tests for fallback type definitions."""

    def test_all_types_have_fallbacks(self):
        """Test that all FallbackType values have corresponding fallback messages."""
        for fallback_type in FallbackType:
            assert fallback_type in FALLBACKS, f"Missing fallback for {fallback_type.name}"

    def test_fallback_responses_have_required_fields(self):
        """Test that all fallback responses have required fields."""
        for fallback_type, response in FALLBACKS.items():
            assert isinstance(response, FallbackResponse), f"{fallback_type.name} is not a FallbackResponse"
            assert isinstance(response.message, str), f"{fallback_type.name} message is not a string"
            assert isinstance(response.category, ErrorCategory), f"{fallback_type.name} category is invalid"
            assert isinstance(response.suggest_retry, bool), f"{fallback_type.name} suggest_retry is not a bool"


class TestErrorCategories:
    """Tests for error categorization."""

    def test_transient_errors(self):
        """Test that transient errors are properly categorized."""
        transient_types = [
            FallbackType.VISION_UNAVAILABLE,
            FallbackType.PLACES_UNAVAILABLE,
            FallbackType.TIMEOUT,
            FallbackType.SESSION_UNAVAILABLE,
        ]
        for ft in transient_types:
            assert FALLBACKS[ft].category == ErrorCategory.TRANSIENT, f"{ft.name} should be transient"

    def test_permanent_errors(self):
        """Test that permanent errors are properly categorized."""
        permanent_types = [
            FallbackType.PHOTO_TOO_LARGE,
            FallbackType.PHOTO_INVALID_FORMAT,
            FallbackType.FEATURE_DISABLED,
            FallbackType.API_KEY_MISSING,
        ]
        for ft in permanent_types:
            assert FALLBACKS[ft].category == ErrorCategory.PERMANENT, f"{ft.name} should be permanent"

    def test_resource_errors(self):
        """Test that resource errors are properly categorized."""
        resource_types = [
            FallbackType.VISION_RATE_LIMITED,
            FallbackType.PLACES_RATE_LIMITED,
            FallbackType.RATE_LIMITED,
        ]
        for ft in resource_types:
            assert FALLBACKS[ft].category == ErrorCategory.RESOURCE, f"{ft.name} should be resource"


class TestCategorizeException:
    """Tests for exception categorization."""

    def test_timeout_error(self):
        """Test categorization of timeout errors."""
        assert categorize_exception(Exception("Connection timeout")) == FallbackType.TIMEOUT
        assert categorize_exception(TimeoutError("timeout")) == FallbackType.TIMEOUT

    def test_rate_limit_error(self):
        """Test categorization of rate limit errors."""
        assert categorize_exception(Exception("429 Too Many Requests")) == FallbackType.RATE_LIMITED
        assert categorize_exception(Exception("Rate limit exceeded")) == FallbackType.RATE_LIMITED

    def test_auth_error(self):
        """Test categorization of authentication errors."""
        assert categorize_exception(Exception("401 Unauthorized")) == FallbackType.API_KEY_INVALID
        assert categorize_exception(Exception("403 Forbidden")) == FallbackType.API_KEY_INVALID

    def test_api_key_error(self):
        """Test categorization of API key errors."""
        assert categorize_exception(Exception("API key not found")) == FallbackType.API_KEY_MISSING
        assert categorize_exception(Exception("Invalid api_key")) == FallbackType.API_KEY_MISSING

    def test_service_unavailable(self):
        """Test categorization of service unavailable errors."""
        assert categorize_exception(Exception("503 Service Unavailable")) == FallbackType.SERVICE_OVERLOADED
        assert categorize_exception(Exception("502 Bad Gateway")) == FallbackType.SERVICE_OVERLOADED

    def test_unknown_error(self):
        """Test that unknown errors are categorized as UNKNOWN."""
        assert categorize_exception(Exception("Something weird")) == FallbackType.UNKNOWN_ERROR


class TestGetFallback:
    """Tests for getting fallback messages."""

    def test_get_vision_fallback(self):
        """Test getting vision unavailable fallback."""
        message = get_fallback(FallbackType.VISION_UNAVAILABLE)
        assert "can't analyze photos" in message.lower()
        assert "text" in message.lower()  # Should suggest text alternative

    def test_get_places_fallback(self):
        """Test getting places unavailable fallback."""
        message = get_fallback(FallbackType.PLACES_UNAVAILABLE)
        assert "google maps" in message.lower()

    def test_get_unknown_fallback(self):
        """Test getting unknown error fallback."""
        message = get_fallback(FallbackType.UNKNOWN_ERROR)
        assert "unexpected" in message.lower()
        assert "different approach" in message.lower()

    def test_get_fallback_with_context(self):
        """Test getting fallback with context."""
        message = get_fallback(
            FallbackType.FEATURE_DISABLED,
            context={"feature": "travel"}
        )
        assert message  # Should return a message

    def test_get_fallback_with_error(self):
        """Test getting fallback with error for logging."""
        error = Exception("Test error")
        message = get_fallback(FallbackType.TIMEOUT, error=error)
        assert message  # Should return a message


class TestGetFallbackResponse:
    """Tests for getting full fallback response objects."""

    def test_get_full_response(self):
        """Test getting full fallback response."""
        response = get_fallback_response(FallbackType.VISION_UNAVAILABLE)
        assert isinstance(response, FallbackResponse)
        assert response.message
        assert response.category == ErrorCategory.TRANSIENT
        assert response.suggest_retry is True

    def test_response_has_alternative(self):
        """Test that some responses have alternative actions."""
        response = get_fallback_response(FallbackType.VISION_UNAVAILABLE)
        assert response.alternative_action is not None

    def test_response_has_retry_after(self):
        """Test that rate limit responses have retry_after."""
        response = get_fallback_response(FallbackType.VISION_RATE_LIMITED)
        assert response.retry_after_seconds is not None
        assert response.retry_after_seconds > 0


class TestConvenienceFunctions:
    """Tests for convenience fallback functions."""

    def test_vision_fallback(self):
        """Test vision_fallback convenience function."""
        message = vision_fallback()
        assert "can't analyze photos" in message.lower()

    def test_vision_fallback_with_error(self):
        """Test vision_fallback with error."""
        message = vision_fallback(error=Exception("API error"))
        assert message

    def test_places_fallback(self):
        """Test places_fallback convenience function."""
        message = places_fallback()
        assert "google maps" in message.lower()

    def test_places_fallback_with_query(self):
        """Test places_fallback with query substitution."""
        message = places_fallback(query="yoga studios")
        # Should either include the query or suggest Google Maps as alternative
        assert "yoga studios" in message.lower() or "google maps" in message.lower()

    def test_tool_fallback(self):
        """Test tool_fallback convenience function."""
        message = tool_fallback("some_tool")
        assert message
        assert "problem" in message.lower() or "different" in message.lower()

    def test_unknown_fallback(self):
        """Test unknown_fallback convenience function."""
        message = unknown_fallback()
        assert "unexpected" in message.lower()


class TestFormatWithAlternative:
    """Tests for formatting messages with alternatives."""

    def test_format_with_alternative(self):
        """Test formatting with alternative action."""
        result = format_with_alternative("Main message.", "Try this instead")
        assert "Main message." in result
        assert "Alternatively:" in result
        assert "Try this instead" in result

    def test_format_without_alternative(self):
        """Test formatting without alternative action."""
        result = format_with_alternative("Main message.", None)
        assert result == "Main message."
        assert "Alternatively" not in result


class TestFallbackMessageQuality:
    """Tests for the quality of fallback messages."""

    def test_no_technical_jargon(self):
        """Test that fallback messages avoid technical jargon."""
        technical_terms = [
            "exception", "error code", "stack trace", "traceback",
            "null", "undefined", "HTTP", "API", "timeout error",
            "connection refused", "circuit breaker"
        ]

        for fallback_type, response in FALLBACKS.items():
            message_lower = response.message.lower()
            for term in technical_terms:
                assert term not in message_lower, (
                    f"'{term}' found in {fallback_type.name} message"
                )

    def test_messages_are_friendly(self):
        """Test that messages use friendly language."""
        unfriendly_patterns = [
            "error:", "failed:", "exception:", "invalid:",
            "you did something wrong", "your fault"
        ]

        for fallback_type, response in FALLBACKS.items():
            message_lower = response.message.lower()
            for pattern in unfriendly_patterns:
                assert pattern not in message_lower, (
                    f"Unfriendly pattern '{pattern}' in {fallback_type.name}"
                )

    def test_messages_suggest_action(self):
        """Test that most messages suggest what to do next."""
        # Messages that should suggest next steps
        action_types = [
            FallbackType.VISION_UNAVAILABLE,
            FallbackType.PLACES_UNAVAILABLE,
            FallbackType.PHOTO_TOO_LARGE,
            FallbackType.UNKNOWN_ERROR,
        ]

        action_words = [
            "try", "send", "describe", "search", "ask",
            "tell me", "let me", "could you", "please"
        ]

        for ft in action_types:
            message = FALLBACKS[ft].message.lower()
            has_action = any(word in message for word in action_words)
            assert has_action, f"{ft.name} should suggest an action"

    def test_silent_fallbacks_are_empty(self):
        """Test that intentionally silent fallbacks have empty messages."""
        # Cache unavailable should be silent
        assert FALLBACKS[FallbackType.CACHE_UNAVAILABLE].message == ""
