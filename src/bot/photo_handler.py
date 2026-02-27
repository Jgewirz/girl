"""Photo message handling with robust error handling and graceful degradation."""

import asyncio
import base64
from dataclasses import dataclass
from enum import Enum
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from src.agent.fallbacks import (
    FallbackType,
    get_fallback,
    unknown_fallback,
    vision_fallback,
)
from src.cache.session import ConversationSession, get_session_manager
from src.config.logging import get_logger

logger = get_logger(__name__)

# Telegram message size limit
MAX_MESSAGE_LENGTH = 4096

# Photo download settings
MAX_DOWNLOAD_RETRIES = 2
DOWNLOAD_TIMEOUT_SECONDS = 30


class PhotoIntent(Enum):
    """Types of photo analysis the user might want."""

    COLOR_ANALYSIS = "color_analysis"
    OUTFIT_FEEDBACK = "outfit_feedback"
    WARDROBE_CATALOG = "wardrobe_catalog"
    UNKNOWN = "unknown"


@dataclass
class PhotoContext:
    """Context for processing a photo message."""

    telegram_id: int
    photo_bytes: bytes
    caption: str
    intent: PhotoIntent
    session: ConversationSession
    occasion: str | None = None


class PhotoDownloadError(Exception):
    """Raised when photo download fails after retries."""
    pass


class PhotoAnalysisError(Exception):
    """Raised when photo analysis fails."""

    def __init__(self, message: str, recoverable: bool = True):
        super().__init__(message)
        self.recoverable = recoverable


# === INTENT DETECTION ===

def detect_photo_intent(caption: str) -> tuple[PhotoIntent, str | None]:
    """
    Detect user intent from photo caption.

    Returns tuple of (intent, occasion if detected).
    """
    caption_lower = caption.lower().strip()

    # Color analysis keywords
    color_keywords = [
        "color", "colors", "colour", "colours",
        "season", "undertone", "analyze my", "analyse my",
        "selfie", "what colors suit", "my coloring", "my colouring",
    ]
    if any(kw in caption_lower for kw in color_keywords):
        return PhotoIntent.COLOR_ANALYSIS, None

    # Wardrobe cataloging keywords
    wardrobe_keywords = [
        "add to wardrobe", "wardrobe", "catalog", "catalogue",
        "save this", "add this", "add item", "new item",
    ]
    if any(kw in caption_lower for kw in wardrobe_keywords):
        return PhotoIntent.WARDROBE_CATALOG, None

    # Outfit feedback (default for most photos)
    # Check for occasion context
    occasion_map = {
        "work": ["work", "office", "professional", "meeting", "interview"],
        "casual": ["casual", "weekend", "everyday", "running errands"],
        "date": ["date", "dinner", "romantic", "date night"],
        "formal": ["formal", "wedding", "gala", "black tie", "event"],
        "party": ["party", "club", "night out", "celebration"],
        "athletic": ["gym", "workout", "athletic", "yoga", "running"],
    }

    occasion = None
    for occ_name, keywords in occasion_map.items():
        if any(kw in caption_lower for kw in keywords):
            occasion = occ_name
            break

    # If caption exists, assume outfit feedback
    if caption_lower:
        return PhotoIntent.OUTFIT_FEEDBACK, occasion

    # No caption - unknown intent
    return PhotoIntent.UNKNOWN, None


# === PHOTO DOWNLOAD ===

async def download_photo_with_retry(
    update: Update,
    max_retries: int = MAX_DOWNLOAD_RETRIES,
    timeout: int = DOWNLOAD_TIMEOUT_SECONDS,
) -> bytes:
    """
    Download photo from Telegram with retry logic.

    Raises PhotoDownloadError if all retries fail.
    """
    if not update.message or not update.message.photo:
        raise PhotoDownloadError("No photo in message")

    # Get highest resolution photo
    photo = update.message.photo[-1]

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            # Get file with timeout
            photo_file = await asyncio.wait_for(
                photo.get_file(),
                timeout=timeout,
            )

            # Download with timeout
            photo_bytes = await asyncio.wait_for(
                photo_file.download_as_bytearray(),
                timeout=timeout,
            )

            logger.info(
                "photo_downloaded",
                attempt=attempt + 1,
                size_bytes=len(photo_bytes),
            )
            return bytes(photo_bytes)

        except asyncio.TimeoutError:
            last_error = "Download timed out"
            logger.warning(
                "photo_download_timeout",
                attempt=attempt + 1,
                max_retries=max_retries,
            )
        except TelegramError as e:
            last_error = str(e)
            logger.warning(
                "photo_download_telegram_error",
                attempt=attempt + 1,
                error=str(e),
            )
        except Exception as e:
            last_error = str(e)
            logger.warning(
                "photo_download_error",
                attempt=attempt + 1,
                error=str(e),
            )

        # Wait before retry (exponential backoff)
        if attempt < max_retries:
            await asyncio.sleep(1 * (attempt + 1))

    raise PhotoDownloadError(f"Failed after {max_retries + 1} attempts: {last_error}")


# === RESPONSE UTILITIES ===

def split_long_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """
    Split a long message into chunks that fit Telegram's limit.

    Tries to split at natural boundaries (newlines, sentences).
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        # Find best split point
        chunk = remaining[:max_length]

        # Try to split at paragraph
        split_at = chunk.rfind("\n\n")
        if split_at == -1 or split_at < max_length // 2:
            # Try to split at newline
            split_at = chunk.rfind("\n")
        if split_at == -1 or split_at < max_length // 2:
            # Try to split at sentence
            for punct in [". ", "! ", "? "]:
                pos = chunk.rfind(punct)
                if pos > max_length // 2:
                    split_at = pos + 1
                    break
        if split_at == -1 or split_at < max_length // 2:
            # Force split at space
            split_at = chunk.rfind(" ")
        if split_at == -1:
            # Absolute last resort: hard split
            split_at = max_length

        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    return chunks


async def send_response(
    update: Update,
    text: str,
    parse_mode: str | None = "Markdown",
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    """Send response, handling long messages and formatting errors."""

    chunks = split_long_message(text)

    for i, chunk in enumerate(chunks):
        # Only add reply markup to last chunk
        markup = reply_markup if i == len(chunks) - 1 else None

        try:
            await update.message.reply_text(
                chunk,
                parse_mode=parse_mode,
                reply_markup=markup,
            )
        except TelegramError as e:
            # If Markdown fails, try without formatting
            if parse_mode and "parse" in str(e).lower():
                logger.warning("markdown_parse_failed", error=str(e))
                await update.message.reply_text(
                    chunk,
                    parse_mode=None,
                    reply_markup=markup,
                )
            else:
                raise


# === ANALYSIS FUNCTIONS ===

async def analyze_colors_photo(
    photo_bytes: bytes,
    session: ConversationSession,
) -> tuple[str, dict | None]:
    """
    Perform color analysis on a selfie.

    Returns (response_text, profile_update_dict or None on error).
    """
    from src.agent.tools.stylist import analyze_my_colors

    try:
        result = await analyze_my_colors(
            image_data=photo_bytes,
            user_context={"style_profile": session.style_profile},
        )

        if "error" in result:
            return result["error"], None

        return result["response"], result.get("profile_update")

    except Exception as e:
        logger.error("color_analysis_exception", error=str(e))
        raise PhotoAnalysisError(
            "I had trouble analyzing your colors. "
            "Try a photo in natural light, facing a window, with minimal makeup.",
            recoverable=True,
        )


async def analyze_outfit_feedback(
    photo_bytes: bytes,
    session: ConversationSession,
    occasion: str | None = None,
    question: str | None = None,
) -> str:
    """
    Analyze an outfit photo and return feedback.
    """
    from src.agent.tools.stylist import analyze_outfit_photo

    try:
        response = await analyze_outfit_photo(
            image_data=photo_bytes,
            occasion=occasion,
            question=question,
            user_context={
                "style_profile": session.style_profile,
                "wardrobe": session.wardrobe,
            },
        )
        return response

    except Exception as e:
        logger.error("outfit_analysis_exception", error=str(e))
        raise PhotoAnalysisError(
            "I couldn't analyze that outfit photo. "
            "Could you try sending it again with better lighting?",
            recoverable=True,
        )


async def catalog_wardrobe_item(
    photo_bytes: bytes,
    session: ConversationSession,
    notes: str | None = None,
) -> tuple[str, dict | None]:
    """
    Catalog a clothing item to the wardrobe.

    Returns (response_text, item_dict or None on error).
    """
    from src.agent.tools.stylist import add_wardrobe_item

    try:
        result = await add_wardrobe_item(
            image_data=photo_bytes,
            notes=notes,
            user_context={"wardrobe": session.wardrobe},
        )

        if "error" in result:
            return result["error"], None

        return result["response"], result.get("item")

    except Exception as e:
        logger.error("wardrobe_catalog_exception", error=str(e))
        raise PhotoAnalysisError(
            "I couldn't catalog that item. "
            "Try a clearer photo of just the clothing item.",
            recoverable=True,
        )


# === INTENT CLARIFICATION ===

def build_intent_keyboard() -> InlineKeyboardMarkup:
    """Build keyboard for asking what user wants to do with photo."""
    keyboard = [
        [
            InlineKeyboardButton("Analyze My Colors", callback_data="photo_intent:colors"),
            InlineKeyboardButton("Rate My Outfit", callback_data="photo_intent:outfit"),
        ],
        [
            InlineKeyboardButton("Add to Wardrobe", callback_data="photo_intent:wardrobe"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


async def ask_intent(update: Update) -> None:
    """Ask user what they want to do with the photo."""
    await update.message.reply_text(
        "Nice photo! What would you like me to do with it?",
        reply_markup=build_intent_keyboard(),
    )


# === MAIN HANDLER ===

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle incoming photo messages with full error handling and graceful degradation.
    """
    user = update.effective_user

    logger.info(
        "photo_received",
        user_id=user.id,
        username=user.username,
        has_caption=bool(update.message.caption),
    )

    # === Rate Limiting ===
    try:
        session_mgr = await get_session_manager()
        is_allowed, count = await session_mgr.check_rate_limit(
            telegram_id=user.id,
            max_requests=30,
            window_seconds=60,
        )
        if not is_allowed:
            await update.message.reply_text(
                "You're sending photos too quickly! Please wait a moment."
            )
            logger.warning("photo_rate_limited", user_id=user.id, count=count)
            return
    except Exception as e:
        # Fail open on rate limit errors
        logger.warning("rate_limit_check_error", user_id=user.id, error=str(e))

    # === Show typing indicator ===
    await update.message.chat.send_action("typing")

    # === Download Photo ===
    try:
        photo_bytes = await download_photo_with_retry(update)
    except PhotoDownloadError as e:
        logger.error("photo_download_failed", user_id=user.id, error=str(e))
        fallback_msg = get_fallback(FallbackType.PHOTO_DOWNLOAD_FAILED, error=e)
        await update.message.reply_text(fallback_msg)
        return

    # === Get Session ===
    session_mgr = None
    try:
        session_mgr = await get_session_manager()
        session = await session_mgr.get_session(user.id)
    except Exception as e:
        logger.error("session_load_error", user_id=user.id, error=str(e))
        # Create ephemeral session and warn user
        from src.cache.session import ConversationSession
        session = ConversationSession(telegram_id=user.id)
        logger.warning("using_ephemeral_session", user_id=user.id)
        # Warn about session persistence
        session_warning = get_fallback(FallbackType.SESSION_UNAVAILABLE, error=e)
        if session_warning:  # Only send if there's a message
            await update.message.reply_text(session_warning)

    # === Detect Intent ===
    caption = update.message.caption or ""
    intent, occasion = detect_photo_intent(caption)

    logger.info(
        "photo_intent_detected",
        user_id=user.id,
        intent=intent.value,
        occasion=occasion,
        caption_preview=caption[:50] if caption else None,
    )

    # If intent unclear, ask the user
    if intent == PhotoIntent.UNKNOWN:
        # Store photo in context for callback handler
        context.user_data["pending_photo"] = base64.b64encode(photo_bytes).decode("utf-8")
        context.user_data["pending_photo_time"] = asyncio.get_event_loop().time()
        await ask_intent(update)
        return

    # === Process by Intent ===
    try:
        if intent == PhotoIntent.COLOR_ANALYSIS:
            response, profile_update = await analyze_colors_photo(photo_bytes, session)

            # Save profile update
            if profile_update:
                session.style_profile.update(profile_update)
                await session_mgr.save_session(session)
                logger.info("style_profile_updated", user_id=user.id)

            await send_response(update, response)

        elif intent == PhotoIntent.WARDROBE_CATALOG:
            # Extract notes from caption (remove trigger words)
            notes = None
            if caption:
                cleaned = caption.lower()
                for kw in ["add to wardrobe", "wardrobe", "catalog", "add this", "save this"]:
                    cleaned = cleaned.replace(kw, "")
                cleaned = cleaned.strip()
                if cleaned:
                    notes = cleaned

            response, item = await catalog_wardrobe_item(photo_bytes, session, notes)

            # Save to wardrobe
            if item:
                session.wardrobe.append(item)
                await session_mgr.save_session(session)
                logger.info(
                    "wardrobe_item_added",
                    user_id=user.id,
                    item_id=item.get("item_id"),
                    category=item.get("category"),
                )

            await send_response(update, response)

        elif intent == PhotoIntent.OUTFIT_FEEDBACK:
            # Use caption as question if it's not just an occasion keyword
            question = caption if caption and len(caption) > 15 else None

            response = await analyze_outfit_feedback(
                photo_bytes, session, occasion, question
            )

            await send_response(update, response)

        # Record photo interaction in session
        session.add_message("human", f"[Photo: {intent.value}] {caption}")
        await session_mgr.save_session(session)

    except PhotoAnalysisError as e:
        logger.warning(
            "photo_analysis_error",
            user_id=user.id,
            intent=intent.value,
            error=str(e),
            recoverable=e.recoverable,
        )
        # Use fallback for vision-related errors
        fallback_msg = vision_fallback(error=e)
        await update.message.reply_text(fallback_msg)

    except Exception as e:
        logger.error(
            "photo_handler_unexpected_error",
            user_id=user.id,
            intent=intent.value,
            error=str(e),
        )
        # Use the unknown error fallback
        fallback_msg = unknown_fallback(error=e)
        await update.message.reply_text(fallback_msg)


async def handle_photo_intent_callback(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Handle callback when user clicks intent button after sending photo without caption.
    """
    query = update.callback_query
    await query.answer()

    user = update.effective_user

    # Check for pending photo
    pending_photo = context.user_data.get("pending_photo")
    pending_time = context.user_data.get("pending_photo_time", 0)

    # Photos expire after 5 minutes
    current_time = asyncio.get_event_loop().time()
    if not pending_photo or (current_time - pending_time) > 300:
        await query.edit_message_text(
            "That photo expired. Please send it again!"
        )
        return

    # Decode photo
    try:
        photo_bytes = base64.b64decode(pending_photo)
    except Exception:
        await query.edit_message_text(
            "Something went wrong. Please send the photo again."
        )
        return

    # Clear pending photo
    context.user_data.pop("pending_photo", None)
    context.user_data.pop("pending_photo_time", None)

    # Parse intent from callback data
    callback_data = query.data  # e.g., "photo_intent:colors"
    intent_str = callback_data.split(":")[-1]

    intent_map = {
        "colors": PhotoIntent.COLOR_ANALYSIS,
        "outfit": PhotoIntent.OUTFIT_FEEDBACK,
        "wardrobe": PhotoIntent.WARDROBE_CATALOG,
    }
    intent = intent_map.get(intent_str, PhotoIntent.OUTFIT_FEEDBACK)

    logger.info(
        "photo_intent_selected",
        user_id=user.id,
        intent=intent.value,
    )

    # Update message to show processing
    await query.edit_message_text(f"Analyzing your photo...")

    # Get session
    try:
        session_mgr = await get_session_manager()
        session = await session_mgr.get_session(user.id)
    except Exception as e:
        logger.error("session_load_error", user_id=user.id, error=str(e))
        from src.cache.session import ConversationSession
        session = ConversationSession(telegram_id=user.id)

    # Process photo
    try:
        if intent == PhotoIntent.COLOR_ANALYSIS:
            response, profile_update = await analyze_colors_photo(photo_bytes, session)
            if profile_update:
                session.style_profile.update(profile_update)
                await session_mgr.save_session(session)

        elif intent == PhotoIntent.WARDROBE_CATALOG:
            response, item = await catalog_wardrobe_item(photo_bytes, session, None)
            if item:
                session.wardrobe.append(item)
                await session_mgr.save_session(session)

        elif intent == PhotoIntent.OUTFIT_FEEDBACK:
            response = await analyze_outfit_feedback(photo_bytes, session)

        else:
            response = "I'm not sure what to do with that. Try sending another photo!"

        # Send response (as new message since we can't edit with Markdown reliably)
        await query.message.reply_text(response, parse_mode="Markdown")

        # Update original message
        await query.edit_message_text(
            f"Analysis complete! See my response below."
        )

    except PhotoAnalysisError as e:
        fallback_msg = vision_fallback(error=e)
        await query.edit_message_text(fallback_msg)

    except Exception as e:
        logger.error(
            "photo_callback_error",
            user_id=user.id,
            intent=intent.value,
            error=str(e),
        )
        fallback_msg = unknown_fallback(error=e)
        await query.edit_message_text(fallback_msg)
