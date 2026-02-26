"""AI Stylist tools for outfit and makeup analysis."""

import uuid
from typing import Any

from pydantic import BaseModel, Field

from src.config.logging import get_logger
from src.services.vision import get_vision_service

from .registry import tool_registry

logger = get_logger(__name__)


# === COLOR PALETTES BY SEASON ===

SEASON_PALETTES = {
    "true_spring": {
        "best": ["coral", "turquoise", "sunny yellow", "mint", "salmon", "peach"],
        "neutrals": ["ivory", "camel", "warm brown"],
        "avoid": ["black", "cool gray", "burgundy"],
        "metals": ["gold", "rose gold"],
        "lips": ["coral", "peach", "warm pink", "poppy red"],
        "eyes": ["warm brown", "peach", "gold", "turquoise"],
    },
    "light_spring": {
        "best": ["light peach", "aqua", "light coral", "butter yellow", "mint"],
        "neutrals": ["cream", "light camel", "soft brown"],
        "avoid": ["black", "dark colors", "muted tones"],
        "metals": ["light gold", "rose gold"],
        "lips": ["peach", "light coral", "soft pink"],
        "eyes": ["soft browns", "peach", "light aqua"],
    },
    "true_summer": {
        "best": ["soft rose", "lavender", "powder blue", "sage", "mauve"],
        "neutrals": ["soft white", "gray", "taupe"],
        "avoid": ["orange", "black", "bright yellow"],
        "metals": ["silver", "white gold", "rose gold"],
        "lips": ["rose", "mauve", "soft berry", "dusty pink"],
        "eyes": ["taupe", "soft plum", "gray", "soft blue"],
    },
    "soft_summer": {
        "best": ["dusty rose", "soft teal", "lavender", "sage green", "powder blue"],
        "neutrals": ["soft gray", "taupe", "cocoa"],
        "avoid": ["bright orange", "black", "stark white"],
        "metals": ["silver", "pewter", "rose gold"],
        "lips": ["dusty rose", "mauve", "soft berry"],
        "eyes": ["taupe", "soft gray", "muted plum"],
    },
    "true_autumn": {
        "best": ["burnt orange", "olive", "mustard", "teal", "rust", "terracotta"],
        "neutrals": ["cream", "camel", "chocolate brown", "khaki"],
        "avoid": ["black", "cool pink", "icy blue"],
        "metals": ["gold", "brass", "copper"],
        "lips": ["terracotta", "brick red", "warm nude", "coral"],
        "eyes": ["copper", "bronze", "olive", "warm brown"],
    },
    "soft_autumn": {
        "best": ["soft teal", "dusty coral", "olive", "warm taupe", "soft rust"],
        "neutrals": ["soft camel", "warm gray", "cream"],
        "avoid": ["black", "bright colors", "cool tones"],
        "metals": ["soft gold", "antique brass"],
        "lips": ["warm nude", "soft coral", "muted terracotta"],
        "eyes": ["soft brown", "taupe", "muted olive"],
    },
    "deep_autumn": {
        "best": ["deep teal", "burnt orange", "olive", "burgundy", "terracotta"],
        "neutrals": ["chocolate", "dark camel", "cream"],
        "avoid": ["pastels", "cool pink", "icy colors"],
        "metals": ["gold", "bronze", "copper"],
        "lips": ["deep coral", "burgundy", "brick red", "deep nude"],
        "eyes": ["bronze", "deep brown", "forest green"],
    },
    "true_winter": {
        "best": ["true red", "royal blue", "hot pink", "emerald", "black", "white"],
        "neutrals": ["pure white", "black", "charcoal", "navy"],
        "avoid": ["orange", "muted colors", "warm beige"],
        "metals": ["silver", "platinum", "white gold"],
        "lips": ["true red", "berry", "fuchsia", "wine"],
        "eyes": ["charcoal", "plum", "navy", "silver"],
    },
    "cool_winter": {
        "best": ["icy pink", "royal purple", "true red", "emerald", "icy blue"],
        "neutrals": ["white", "black", "charcoal", "cool gray"],
        "avoid": ["orange", "warm brown", "gold tones"],
        "metals": ["silver", "platinum"],
        "lips": ["cool pink", "berry", "true red", "plum"],
        "eyes": ["cool gray", "plum", "icy colors"],
    },
    "deep_winter": {
        "best": ["burgundy", "deep purple", "forest green", "true red", "black"],
        "neutrals": ["black", "charcoal", "dark navy"],
        "avoid": ["pastels", "warm muted colors"],
        "metals": ["silver", "gunmetal"],
        "lips": ["deep berry", "burgundy", "deep red"],
        "eyes": ["charcoal", "deep plum", "black"],
    },
}


# === TOOL FUNCTIONS ===


async def analyze_outfit_photo(
    image_data: bytes,
    occasion: str | None = None,
    question: str | None = None,
    user_context: dict | None = None,
) -> str:
    """Analyze an outfit photo and provide personalized feedback."""

    vision = await get_vision_service()
    if not vision:
        return "I can't analyze photos right now. Please make sure the service is configured."

    style_profile = None
    if user_context and user_context.get("style_profile"):
        style_profile = user_context["style_profile"]

    try:
        analysis = await vision.analyze_outfit(
            image_data=image_data,
            style_profile=style_profile,
            occasion=occasion,
            question=question,
        )

        # Format response
        parts = []
        parts.append(f"**Style:** {analysis.overall_style.title()}")
        parts.append(f"**Rating:** {'⭐' * analysis.rating}/10")

        if analysis.whats_working:
            parts.append("\n**What's Working:**")
            for point in analysis.whats_working[:4]:
                parts.append(f"  ✓ {point}")

        if analysis.suggestions:
            parts.append("\n**Suggestions:**")
            for suggestion in analysis.suggestions[:3]:
                parts.append(f"  → {suggestion}")

        if analysis.color_notes and style_profile:
            parts.append("\n**Color Notes:**")
            for note in analysis.color_notes[:2]:
                parts.append(f"  🎨 {note}")

        if analysis.elevate_with:
            parts.append("\n**To Elevate:**")
            for idea in analysis.elevate_with[:3]:
                parts.append(f"  + {idea}")

        if occasion and analysis.occasion_scores:
            score = analysis.occasion_scores.get(occasion.lower(), 0.7)
            if score >= 0.8:
                parts.append(f"\n✅ Great choice for {occasion}!")
            elif score >= 0.6:
                parts.append(f"\n👍 Works for {occasion} with minor tweaks")
            else:
                parts.append(f"\n⚠️ Consider adjustments for {occasion}")

        return "\n".join(parts)

    except Exception as e:
        logger.error("analyze_outfit_failed", error=str(e))
        return "I had trouble analyzing that photo. Could you try sending it again with good lighting?"


async def analyze_my_colors(
    image_data: bytes,
    lighting: str | None = None,
    user_context: dict | None = None,
) -> dict[str, Any]:
    """Analyze coloring to determine color season. Returns data to save to profile."""

    vision = await get_vision_service()
    if not vision:
        return {"error": "Vision service not configured"}

    try:
        analysis = await vision.analyze_colors(
            image_data=image_data,
            lighting=lighting,
        )

        # Build response text
        season_display = analysis.color_season.replace("_", " ").title()

        response = f"""**Your Color Analysis** ✨

**Undertone:** {analysis.undertone.title()} ({int(analysis.undertone_confidence * 100)}% confidence)

**Color Season:** {season_display}
{analysis.season_reasoning}

**Your Features:**
- Skin: {analysis.features.get('skin_tone', 'N/A')}
- Hair: {analysis.features.get('hair_color', 'N/A')}
- Eyes: {analysis.features.get('eye_color', 'N/A')}

**Your Best Colors:**
"""
        for color in analysis.best_colors[:6]:
            response += f"  • {color.get('name', 'Unknown')}\n"

        response += "\n**Colors to Avoid:**\n"
        for color in analysis.avoid_colors[:3]:
            response += f"  • {color.get('name', 'Unknown')}"
            if color.get("reason"):
                response += f" ({color['reason']})"
            response += "\n"

        makeup = analysis.makeup_recommendations
        if makeup:
            response += f"""
**Makeup Tips:**
- Foundation: Look for {makeup.get('foundation_undertone', 'neutral')} undertones
- Lips: {', '.join(makeup.get('lip_colors', ['nude'])[:3])}
- Eyes: {', '.join(makeup.get('eye_colors', ['neutral'])[:3])}
"""

        response += "\n*I've saved this to your profile! All future recommendations will be personalized.*"

        # Return both response and data to save
        return {
            "response": response,
            "profile_update": {
                "skin_undertone": analysis.undertone,
                "color_season": analysis.color_season,
                "best_colors": [c.get("name", "") for c in analysis.best_colors],
                "avoid_colors": [c.get("name", "") for c in analysis.avoid_colors],
                "hair_color": analysis.features.get("hair_color"),
                "eye_color": analysis.features.get("eye_color"),
            },
        }

    except Exception as e:
        logger.error("color_analysis_failed", error=str(e))
        return {"error": "Analysis failed. Try a photo in natural light with minimal makeup."}


async def add_wardrobe_item(
    image_data: bytes,
    notes: str | None = None,
    user_context: dict | None = None,
) -> dict[str, Any]:
    """Catalog a clothing item and add to wardrobe."""

    vision = await get_vision_service()
    if not vision:
        return {"error": "Vision service not configured"}

    try:
        analysis = await vision.catalog_item(image_data)

        item = {
            "item_id": str(uuid.uuid4())[:8],
            "category": analysis.category,
            "subcategory": analysis.subcategory,
            "colors": [c.get("name", "") for c in analysis.colors],
            "patterns": analysis.patterns,
            "occasions": analysis.occasions,
            "seasons": analysis.seasons,
            "notes": notes,
        }

        response = f"""**Added to Wardrobe!** 👗

**{analysis.subcategory.title()}** ({analysis.category})
Colors: {', '.join(item['colors'])}
Good for: {', '.join(analysis.occasions)}
Seasons: {', '.join(analysis.seasons)}
"""

        if notes:
            response += f"Notes: {notes}\n"

        response += "\nI'll use this when suggesting outfits!"

        return {"response": response, "item": item}

    except Exception as e:
        logger.error("wardrobe_add_failed", error=str(e))
        return {"error": "Couldn't catalog that item. Try a clearer photo."}


async def get_makeup_recommendations(
    occasion: str | None = None,
    outfit_colors: list[str] | None = None,
    style: str | None = None,
    user_context: dict | None = None,
) -> str:
    """Get personalized makeup recommendations based on color season."""

    style_profile = user_context.get("style_profile") if user_context else None

    if not style_profile or not style_profile.get("color_season"):
        return """I don't know your color season yet!

Send me a selfie in natural light (minimal makeup) and say "analyze my colors" - then I can give you personalized makeup recommendations that make you glow! ✨"""

    season = style_profile["color_season"]
    palette = SEASON_PALETTES.get(season, SEASON_PALETTES["true_autumn"])
    season_display = season.replace("_", " ").title()

    response = f"**Makeup for {season_display}** 💄\n\n"

    if occasion:
        response += f"*For: {occasion}*\n\n"

    response += "**Lips:**\n"
    for color in palette.get("lips", [])[:4]:
        response += f"  💋 {color.title()}\n"

    response += "\n**Eyes:**\n"
    for color in palette.get("eyes", [])[:4]:
        response += f"  👁️ {color.title()}\n"

    response += f"\n**Your metals:** {', '.join(palette.get('metals', ['silver']))}\n"

    if outfit_colors:
        response += f"\n**With your outfit:** Consider a neutral eye to let your {', '.join(outfit_colors)} outfit shine, or pick up an accent in your lip color."

    if style == "natural":
        response += "\n\n*For a natural look:* Stick to your best neutrals - a hint of your lip color and mascara goes a long way!"
    elif style == "glam":
        response += "\n\n*For glam:* Don't be afraid to go bold with your lips or a smoky eye in your season's colors!"

    return response


async def get_outfit_suggestion(
    occasion: str,
    weather: str | None = None,
    mood: str | None = None,
    user_context: dict | None = None,
) -> str:
    """Generate outfit suggestions based on occasion and profile."""

    style_profile = user_context.get("style_profile") if user_context else None
    wardrobe = user_context.get("wardrobe", []) if user_context else []

    if not style_profile or not style_profile.get("color_season"):
        # Generic suggestions
        return f"""**Outfit Ideas for {occasion.title()}** 👗

Since I don't know your colors yet, here are some classic combinations:

**Option 1:** Navy + white + cognac accessories
**Option 2:** Black + cream + gold jewelry
**Option 3:** Olive + white + tan leather

*Send me a selfie and say "analyze my colors" for personalized recommendations!*"""

    season = style_profile["color_season"]
    palette = SEASON_PALETTES.get(season, SEASON_PALETTES["true_autumn"])
    season_display = season.replace("_", " ").title()

    response = f"**{occasion.title()} Outfit Ideas** ({season_display})\n\n"

    neutrals = palette.get("neutrals", ["cream", "gray"])
    accents = palette.get("best", ["coral", "teal"])
    metals = palette.get("metals", ["gold"])

    # Generate combinations
    response += f"**Classic Combo:**\n"
    response += f"  {neutrals[0].title()} base + {accents[0].title()} accent + {metals[0].title()} jewelry\n\n"

    response += f"**Bold Option:**\n"
    response += f"  {accents[1].title() if len(accents) > 1 else accents[0].title()} statement piece + {neutrals[-1].title()} to ground it\n\n"

    if weather:
        response += f"*For {weather} weather:* "
        if "cold" in weather.lower() or "winter" in weather.lower():
            response += "Layer with your best neutrals and add texture!\n"
        elif "hot" in weather.lower() or "summer" in weather.lower():
            response += "Lighter fabrics in your brighter colors will keep you cool and glowing!\n"

    if mood:
        response += f"\n*Feeling {mood}?* "
        if mood.lower() in ["powerful", "confident"]:
            response += f"Go for high contrast - {palette['best'][0]} with {neutrals[-1]}!"
        elif mood.lower() in ["cozy", "relaxed"]:
            response += f"Soft layers in your neutrals - {neutrals[0]} and {neutrals[1]}."

    # If wardrobe items, suggest from what they have
    if wardrobe:
        response += f"\n\n**From Your Wardrobe:**\n"
        matching = [
            item
            for item in wardrobe
            if occasion.lower() in [o.lower() for o in item.get("occasions", [])]
        ][:3]
        if matching:
            for item in matching:
                response += f"  • Your {item.get('subcategory', 'item')} ({', '.join(item.get('colors', []))})\n"
        else:
            response += "  (Add more items with photos to get personalized suggestions!)"

    return response


# === REGISTRATION ===


def register_stylist_tools() -> None:
    """Register all stylist tools with the registry."""

    tool_registry.register(
        name="analyze_outfit",
        description="""Analyze an outfit from a photo. Use when user sends a photo and wants feedback on what they're wearing, asks "how does this look?", or asks if something works for an occasion.""",
        func=analyze_outfit_photo,
    )

    tool_registry.register(
        name="analyze_colors",
        description="""Analyze a user's natural coloring from a selfie to determine their color season. Use when user wants to know what colors suit them or says "analyze my colors".""",
        func=analyze_my_colors,
    )

    tool_registry.register(
        name="add_to_wardrobe",
        description="""Add a clothing item to the user's wardrobe. Use when user sends a photo of a clothing item (not worn) and wants to catalog it.""",
        func=add_wardrobe_item,
    )

    tool_registry.register(
        name="get_makeup_recommendations",
        description="""Get personalized makeup recommendations based on user's color season. Use when user asks about makeup colors or wants makeup suggestions.""",
        func=get_makeup_recommendations,
    )

    tool_registry.register(
        name="suggest_outfit",
        description="""Generate outfit suggestions for an occasion using the user's color palette and wardrobe. Use when user asks what to wear or needs outfit ideas.""",
        func=get_outfit_suggestion,
    )
