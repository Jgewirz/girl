"""Tests for photo message handling."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bot.photo_handler import (
    MAX_MESSAGE_LENGTH,
    PhotoAnalysisError,
    PhotoDownloadError,
    PhotoIntent,
    detect_photo_intent,
    download_photo_with_retry,
    split_long_message,
)


class TestDetectPhotoIntent:
    """Tests for intent detection from photo captions."""

    def test_color_analysis_keywords(self):
        """Test detection of color analysis intent."""
        test_cases = [
            ("analyze my colors", PhotoIntent.COLOR_ANALYSIS),
            ("What colors suit me?", PhotoIntent.COLOR_ANALYSIS),
            ("What's my color season?", PhotoIntent.COLOR_ANALYSIS),
            ("Selfie - tell me my undertone", PhotoIntent.COLOR_ANALYSIS),
            ("Find my COLOUR season", PhotoIntent.COLOR_ANALYSIS),
            ("What is my colouring?", PhotoIntent.COLOR_ANALYSIS),
        ]
        for caption, expected_intent in test_cases:
            intent, _ = detect_photo_intent(caption)
            assert intent == expected_intent, f"Failed for caption: {caption}"

    def test_wardrobe_catalog_keywords(self):
        """Test detection of wardrobe cataloging intent."""
        test_cases = [
            ("add to wardrobe", PhotoIntent.WARDROBE_CATALOG),
            ("Add this to my wardrobe", PhotoIntent.WARDROBE_CATALOG),
            ("Catalog this item", PhotoIntent.WARDROBE_CATALOG),
            ("Save this dress", PhotoIntent.WARDROBE_CATALOG),
            ("New item for my closet", PhotoIntent.WARDROBE_CATALOG),
        ]
        for caption, expected_intent in test_cases:
            intent, _ = detect_photo_intent(caption)
            assert intent == expected_intent, f"Failed for caption: {caption}"

    def test_outfit_feedback_default(self):
        """Test that captions default to outfit feedback."""
        test_cases = [
            ("How does this look?", PhotoIntent.OUTFIT_FEEDBACK, None),
            ("Rate my outfit", PhotoIntent.OUTFIT_FEEDBACK, None),
            ("Is this okay?", PhotoIntent.OUTFIT_FEEDBACK, None),
            ("Thoughts?", PhotoIntent.OUTFIT_FEEDBACK, None),
        ]
        for caption, expected_intent, expected_occasion in test_cases:
            intent, occasion = detect_photo_intent(caption)
            assert intent == expected_intent, f"Failed for caption: {caption}"

    def test_outfit_with_occasion_detection(self):
        """Test occasion extraction from captions."""
        test_cases = [
            ("Is this good for work?", "work"),
            ("Date night outfit", "date"),
            ("Going to a formal event", "formal"),
            ("Casual weekend look", "casual"),
            ("Wedding guest outfit", "formal"),
            ("Party tonight!", "party"),
            ("Gym outfit check", "athletic"),
        ]
        for caption, expected_occasion in test_cases:
            intent, occasion = detect_photo_intent(caption)
            assert intent == PhotoIntent.OUTFIT_FEEDBACK
            assert occasion == expected_occasion, f"Failed for caption: {caption}"

    def test_empty_caption_unknown(self):
        """Test that empty captions result in unknown intent."""
        intent, occasion = detect_photo_intent("")
        assert intent == PhotoIntent.UNKNOWN
        assert occasion is None

    def test_whitespace_only_unknown(self):
        """Test that whitespace-only captions result in unknown intent."""
        intent, _ = detect_photo_intent("   ")
        assert intent == PhotoIntent.UNKNOWN


class TestSplitLongMessage:
    """Tests for message splitting logic."""

    def test_short_message_not_split(self):
        """Test that short messages are not split."""
        text = "This is a short message."
        chunks = split_long_message(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_message_split(self):
        """Test that long messages are split."""
        # Create a message longer than the limit
        text = "A" * (MAX_MESSAGE_LENGTH + 100)
        chunks = split_long_message(text)
        assert len(chunks) >= 2
        # Verify all content is preserved
        assert "".join(chunks) == text

    def test_split_at_paragraph(self):
        """Test that messages are split at paragraph boundaries when possible."""
        para1 = "First paragraph. " * 50
        para2 = "Second paragraph. " * 50
        text = f"{para1}\n\n{para2}"

        chunks = split_long_message(text, max_length=len(para1) + 50)
        assert len(chunks) >= 2

    def test_split_at_newline(self):
        """Test that messages are split at newlines when no paragraph break."""
        line1 = "Line one. " * 50
        line2 = "Line two. " * 50
        text = f"{line1}\n{line2}"

        chunks = split_long_message(text, max_length=len(line1) + 50)
        assert len(chunks) >= 2

    def test_split_preserves_content(self):
        """Test that split preserves all content."""
        text = "Hello world. " * 500
        chunks = split_long_message(text, max_length=200)

        # Remove whitespace that might be trimmed
        original_stripped = text.replace(" ", "").replace("\n", "")
        joined_stripped = "".join(chunks).replace(" ", "").replace("\n", "")

        assert original_stripped == joined_stripped


class TestPhotoDownload:
    """Tests for photo download with retry logic."""

    @pytest.mark.asyncio
    async def test_successful_download(self):
        """Test successful photo download."""
        # Mock the update with a photo
        mock_photo = MagicMock()
        mock_file = AsyncMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"fake_image_data"))
        mock_photo.get_file = AsyncMock(return_value=mock_file)

        mock_message = MagicMock()
        mock_message.photo = [mock_photo]

        mock_update = MagicMock()
        mock_update.message = mock_message

        result = await download_photo_with_retry(mock_update, max_retries=1, timeout=5)

        assert result == b"fake_image_data"
        mock_photo.get_file.assert_called_once()
        mock_file.download_as_bytearray.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_photo_raises_error(self):
        """Test that missing photo raises PhotoDownloadError."""
        mock_update = MagicMock()
        mock_update.message = MagicMock()
        mock_update.message.photo = None

        with pytest.raises(PhotoDownloadError, match="No photo"):
            await download_photo_with_retry(mock_update)

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test that download retries on failure."""
        mock_photo = MagicMock()
        mock_file = AsyncMock()

        # Fail first, succeed second
        mock_file.download_as_bytearray = AsyncMock(
            side_effect=[Exception("Network error"), bytearray(b"success")]
        )
        mock_photo.get_file = AsyncMock(return_value=mock_file)

        mock_message = MagicMock()
        mock_message.photo = [mock_photo]

        mock_update = MagicMock()
        mock_update.message = mock_message

        result = await download_photo_with_retry(mock_update, max_retries=1, timeout=5)

        assert result == b"success"
        assert mock_file.download_as_bytearray.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test that error is raised after max retries."""
        mock_photo = MagicMock()
        mock_file = AsyncMock()
        mock_file.download_as_bytearray = AsyncMock(side_effect=Exception("Always fails"))
        mock_photo.get_file = AsyncMock(return_value=mock_file)

        mock_message = MagicMock()
        mock_message.photo = [mock_photo]

        mock_update = MagicMock()
        mock_update.message = mock_message

        with pytest.raises(PhotoDownloadError, match="Failed after"):
            await download_photo_with_retry(mock_update, max_retries=1, timeout=5)


class TestPhotoAnalysisError:
    """Tests for PhotoAnalysisError exception."""

    def test_recoverable_error(self):
        """Test recoverable error flag."""
        error = PhotoAnalysisError("Try again", recoverable=True)
        assert str(error) == "Try again"
        assert error.recoverable is True

    def test_non_recoverable_error(self):
        """Test non-recoverable error flag."""
        error = PhotoAnalysisError("Service down", recoverable=False)
        assert error.recoverable is False


class TestIntentKeyboard:
    """Tests for intent clarification keyboard."""

    def test_keyboard_has_all_options(self):
        """Test that keyboard includes all photo intent options."""
        from src.bot.photo_handler import build_intent_keyboard

        keyboard = build_intent_keyboard()

        # Flatten all buttons
        all_buttons = []
        for row in keyboard.inline_keyboard:
            all_buttons.extend(row)

        callback_data = [btn.callback_data for btn in all_buttons]

        assert "photo_intent:colors" in callback_data
        assert "photo_intent:outfit" in callback_data
        assert "photo_intent:wardrobe" in callback_data


class TestHandlePhotoIntegration:
    """Integration tests for the full photo handler flow."""

    @pytest.fixture
    def mock_update(self):
        """Create a mock Telegram update with photo."""
        mock_photo = MagicMock()
        mock_file = AsyncMock()
        mock_file.download_as_bytearray = AsyncMock(
            return_value=bytearray(b"fake_image_data")
        )
        mock_photo.get_file = AsyncMock(return_value=mock_file)

        mock_message = MagicMock()
        mock_message.photo = [mock_photo]
        mock_message.caption = None
        mock_message.reply_text = AsyncMock()
        mock_message.chat = MagicMock()
        mock_message.chat.send_action = AsyncMock()

        mock_user = MagicMock()
        mock_user.id = 12345
        mock_user.username = "testuser"

        mock_update = MagicMock()
        mock_update.message = mock_message
        mock_update.effective_user = mock_user

        return mock_update

    @pytest.fixture
    def mock_context(self):
        """Create a mock context."""
        context = MagicMock()
        context.user_data = {}
        return context

    @pytest.mark.asyncio
    async def test_photo_without_caption_asks_intent(self, mock_update, mock_context):
        """Test that photo without caption prompts for intent."""
        from src.bot.photo_handler import handle_photo

        mock_update.message.caption = None

        # Mock session manager
        with patch("src.bot.photo_handler.get_session_manager") as mock_mgr:
            mock_session_mgr = AsyncMock()
            mock_session_mgr.check_rate_limit = AsyncMock(return_value=(True, 1))
            mock_mgr.return_value = mock_session_mgr

            await handle_photo(mock_update, mock_context)

        # Should have stored photo in context
        assert "pending_photo" in mock_context.user_data

        # Should have asked for intent
        mock_update.message.reply_text.assert_called()
        call_args = mock_update.message.reply_text.call_args
        assert "What would you like me to do" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_photo_with_color_caption_analyzes_colors(self, mock_update, mock_context):
        """Test that photo with color keywords triggers color analysis."""
        from src.bot.photo_handler import handle_photo, ConversationSession

        mock_update.message.caption = "analyze my colors"

        # Mock session manager and stylist
        with patch("src.bot.photo_handler.get_session_manager") as mock_mgr:
            mock_session_mgr = AsyncMock()
            mock_session_mgr.check_rate_limit = AsyncMock(return_value=(True, 1))
            mock_session = ConversationSession(telegram_id=12345)
            mock_session_mgr.get_session = AsyncMock(return_value=mock_session)
            mock_session_mgr.save_session = AsyncMock()
            mock_mgr.return_value = mock_session_mgr

            with patch("src.bot.photo_handler.analyze_colors_photo") as mock_analyze:
                mock_analyze.return_value = ("You're a True Autumn!", {"color_season": "true_autumn"})

                await handle_photo(mock_update, mock_context)

        mock_analyze.assert_called_once()
        mock_update.message.reply_text.assert_called()

    @pytest.mark.asyncio
    async def test_rate_limited_user_blocked(self, mock_update, mock_context):
        """Test that rate-limited users are blocked."""
        from src.bot.photo_handler import handle_photo

        with patch("src.bot.photo_handler.get_session_manager") as mock_mgr:
            mock_session_mgr = AsyncMock()
            mock_session_mgr.check_rate_limit = AsyncMock(return_value=(False, 31))
            mock_mgr.return_value = mock_session_mgr

            await handle_photo(mock_update, mock_context)

        # Should inform about rate limiting
        mock_update.message.reply_text.assert_called()
        call_args = mock_update.message.reply_text.call_args
        assert "too quickly" in call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_download_error_handled_gracefully(self, mock_update, mock_context):
        """Test that download errors are handled gracefully."""
        from src.bot.photo_handler import handle_photo

        # Make download fail
        mock_update.message.photo[-1].get_file = AsyncMock(
            side_effect=Exception("Network error")
        )

        with patch("src.bot.photo_handler.get_session_manager") as mock_mgr:
            mock_session_mgr = AsyncMock()
            mock_session_mgr.check_rate_limit = AsyncMock(return_value=(True, 1))
            mock_mgr.return_value = mock_session_mgr

            await handle_photo(mock_update, mock_context)

        # Should send friendly error
        mock_update.message.reply_text.assert_called()
        call_args = mock_update.message.reply_text.call_args
        assert "trouble" in call_args[0][0].lower() or "try" in call_args[0][0].lower()
