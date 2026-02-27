"""Vision service for style analysis using GPT-4V."""

import base64
import json
from dataclasses import dataclass
from typing import Any

import httpx
from openai import AsyncOpenAI

from src.config import settings
from src.config.logging import get_logger
from src.services.resilience import (
    ResilienceResult,
    create_service_config,
    resilient_call,
)

logger = get_logger(__name__)

# Resilience configuration for OpenAI
_openai_config = create_service_config("openai")


@dataclass
class OutfitAnalysis:
    """Structured analysis of an outfit photo."""

    items_detected: list[dict]
    overall_style: str
    color_harmony: float
    occasion_scores: dict[str, float]
    whats_working: list[str]
    suggestions: list[str]
    color_notes: list[str]
    elevate_with: list[str]
    rating: int
    confidence: float


@dataclass
class ColorAnalysis:
    """Result of analyzing someone's coloring."""

    undertone: str
    undertone_confidence: float
    color_season: str
    season_confidence: float
    season_reasoning: str
    features: dict[str, str]
    best_colors: list[dict]
    avoid_colors: list[dict]
    makeup_recommendations: dict


@dataclass
class WardrobeItemAnalysis:
    """Analysis of a single wardrobe item."""

    category: str
    subcategory: str
    colors: list[dict]
    patterns: list[str]
    occasions: list[str]
    seasons: list[str]
    style_descriptors: list[str]


class VisionService:
    """Service for analyzing images using GPT-4V."""

    def __init__(self, api_key: str):
        self._client = AsyncOpenAI(api_key=api_key)

    async def analyze_outfit(
        self,
        image_data: bytes,
        style_profile: dict | None = None,
        occasion: str | None = None,
        question: str | None = None,
    ) -> OutfitAnalysis:
        """Analyze an outfit photo with personalized feedback."""

        system_prompt = self._build_outfit_prompt(style_profile)
        user_prompt = self._build_outfit_question(occasion, question, style_profile)

        base64_image = base64.b64encode(image_data).decode("utf-8")

        async def _call_api():
            response = await self._client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high",
                                },
                            },
                        ],
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=1500,
            )
            return json.loads(response.choices[0].message.content)

        result = await resilient_call(
            func=_call_api,
            config=_openai_config,
        )

        if not result.success:
            logger.error("outfit_analysis_failed", error=result.error)
            raise RuntimeError(result.error or "Failed to analyze outfit")

        data = result.value
        logger.info("outfit_analysis_complete", attempts=result.attempts)

        return OutfitAnalysis(
            items_detected=data.get("items_detected", []),
            overall_style=data.get("overall_style", "casual"),
            color_harmony=data.get("color_harmony", 0.5),
            occasion_scores=data.get("occasion_scores", {}),
            whats_working=data.get("whats_working", []),
            suggestions=data.get("suggestions", []),
            color_notes=data.get("color_notes", []),
            elevate_with=data.get("elevate_with", []),
            rating=data.get("rating", 7),
            confidence=data.get("confidence", 0.8),
        )

    async def analyze_colors(
        self,
        image_data: bytes,
        lighting: str | None = None,
    ) -> ColorAnalysis:
        """Analyze a person's coloring to determine their color season."""

        system_prompt = """You are an expert color analyst trained in the 12-season color analysis system.
        Analyze the person's natural coloring to determine their undertone and color season.

        The 12 seasons are:
        - SPRING (warm + light/bright): light_spring, true_spring, clear_spring
        - SUMMER (cool + muted/light): light_summer, true_summer, soft_summer
        - AUTUMN (warm + muted/deep): soft_autumn, true_autumn, deep_autumn
        - WINTER (cool + bright/deep): cool_winter, true_winter, deep_winter

        Provide specific, actionable color recommendations."""

        user_prompt = f"""Analyze this person's coloring. {f'Photo taken in {lighting} lighting.' if lighting else ''}

        Return JSON:
        {{
            "undertone": "warm|cool|neutral",
            "undertone_confidence": 0.0-1.0,
            "color_season": "one of the 12 seasons (e.g., true_autumn)",
            "season_confidence": 0.0-1.0,
            "season_reasoning": "explanation of why this season",
            "features": {{
                "skin_tone": "description",
                "hair_color": "description",
                "eye_color": "description"
            }},
            "best_colors": [{{"hex": "#XXXXXX", "name": "Color Name"}}],
            "avoid_colors": [{{"hex": "#XXXXXX", "name": "Color Name", "reason": "why"}}],
            "makeup_recommendations": {{
                "foundation_undertone": "warm|cool|neutral",
                "lip_colors": ["color1", "color2"],
                "eye_colors": ["color1", "color2"],
                "blush": ["color1"]
            }}
        }}"""

        base64_image = base64.b64encode(image_data).decode("utf-8")

        async def _call_api():
            response = await self._client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high",
                                },
                            },
                        ],
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=1200,
            )
            return json.loads(response.choices[0].message.content)

        result = await resilient_call(
            func=_call_api,
            config=_openai_config,
        )

        if not result.success:
            logger.error("color_analysis_failed", error=result.error)
            raise RuntimeError(result.error or "Failed to analyze colors")

        data = result.value
        logger.info("color_analysis_complete", season=data.get("color_season"), attempts=result.attempts)

        return ColorAnalysis(
            undertone=data.get("undertone", "neutral"),
            undertone_confidence=data.get("undertone_confidence", 0.7),
            color_season=data.get("color_season", "true_autumn"),
            season_confidence=data.get("season_confidence", 0.7),
            season_reasoning=data.get("season_reasoning", ""),
            features=data.get("features", {}),
            best_colors=data.get("best_colors", []),
            avoid_colors=data.get("avoid_colors", []),
            makeup_recommendations=data.get("makeup_recommendations", {}),
        )

    async def catalog_item(self, image_data: bytes) -> WardrobeItemAnalysis:
        """Analyze a clothing item for wardrobe cataloging."""

        system_prompt = """You are cataloging a clothing item for a digital wardrobe.
        Analyze the image and extract detailed information about the item."""

        user_prompt = """Catalog this clothing item. Return JSON:
        {
            "category": "tops|bottoms|dresses|outerwear|shoes|bags|accessories|activewear",
            "subcategory": "specific type (e.g., blouse, jeans, sneakers)",
            "colors": [{"hex": "#XXXXXX", "name": "Color Name"}],
            "patterns": ["solid|stripes|floral|plaid|etc"],
            "occasions": ["work", "casual", "formal", "date", "athletic"],
            "seasons": ["spring", "summer", "fall", "winter"],
            "style_descriptors": ["classic", "trendy", "bohemian", "minimalist", etc]
        }"""

        base64_image = base64.b64encode(image_data).decode("utf-8")

        async def _call_api():
            response = await self._client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high",
                                },
                            },
                        ],
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=800,
            )
            return json.loads(response.choices[0].message.content)

        result = await resilient_call(
            func=_call_api,
            config=_openai_config,
        )

        if not result.success:
            logger.error("item_catalog_failed", error=result.error)
            raise RuntimeError(result.error or "Failed to catalog item")

        data = result.value
        logger.info("item_catalog_complete", category=data.get("category"), attempts=result.attempts)

        return WardrobeItemAnalysis(
            category=data.get("category", "tops"),
            subcategory=data.get("subcategory", "top"),
            colors=data.get("colors", []),
            patterns=data.get("patterns", ["solid"]),
            occasions=data.get("occasions", ["casual"]),
            seasons=data.get("seasons", ["spring", "summer", "fall", "winter"]),
            style_descriptors=data.get("style_descriptors", []),
        )

    def _build_outfit_prompt(self, profile: dict | None) -> str:
        """Build personalized system prompt."""

        base = """You are an expert personal stylist providing outfit feedback.
        Be specific, kind but honest, and actionable. Focus on what works."""

        if profile:
            if profile.get("color_season"):
                base += f"\n\nThis client's color season is {profile['color_season']}."
                if profile.get("best_colors"):
                    base += f" Their best colors include: {', '.join(profile['best_colors'][:5])}."
            if profile.get("body_type"):
                base += f" Body type: {profile['body_type']}."
            if profile.get("style_archetypes"):
                base += f" Style identity: {', '.join(profile['style_archetypes'])}."

        return base

    def _build_outfit_question(
        self, occasion: str | None, question: str | None, profile: dict | None
    ) -> str:
        """Build the user prompt for outfit analysis."""

        prompt = "Analyze this outfit. "
        if occasion:
            prompt += f"It's for: {occasion}. "
        if question:
            prompt += f"Specific question: {question} "

        prompt += """
        Return JSON:
        {
            "items_detected": [{"type": "top", "description": "white blouse", "color": "#FFFFFF"}],
            "overall_style": "smart casual",
            "color_harmony": 0.0-1.0,
            "occasion_scores": {"work": 0.8, "casual": 0.9, "formal": 0.3, "date": 0.7},
            "whats_working": ["list of positives"],
            "suggestions": ["specific improvements"],
            "color_notes": ["feedback on colors for their coloring"],
            "elevate_with": ["accessories or swaps to elevate"],
            "rating": 1-10,
            "confidence": 0.0-1.0
        }"""

        return prompt


# Singleton
_vision_service: VisionService | None = None


async def get_vision_service() -> VisionService | None:
    """Get or create the vision service singleton."""
    global _vision_service

    if not settings.openai_api_key:
        logger.warning("openai_not_configured")
        return None

    if _vision_service is None:
        _vision_service = VisionService(
            api_key=settings.openai_api_key.get_secret_value()
        )

    return _vision_service
