"""Agent state definition for LangGraph."""

from dataclasses import dataclass, field
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


@dataclass
class StyleProfile:
    """Deep personalization data for style recommendations."""

    # Color analysis
    skin_undertone: Literal["warm", "cool", "neutral"] | None = None
    color_season: str | None = None  # e.g., "true_autumn", "cool_winter"
    best_colors: list[str] = field(default_factory=list)
    avoid_colors: list[str] = field(default_factory=list)
    hair_color: str | None = None
    eye_color: str | None = None

    # Body & fit preferences
    body_type: str | None = None
    height_category: Literal["petite", "average", "tall"] | None = None
    preferred_fit: Literal["fitted", "relaxed", "oversized"] | None = None

    # Style identity
    style_archetypes: list[str] = field(default_factory=list)
    aesthetic_keywords: list[str] = field(default_factory=list)

    # Lifestyle context
    work_dress_code: str | None = None
    budget_range: Literal["budget", "mid", "premium", "luxury"] | None = None

    # Makeup preferences
    makeup_style: Literal["minimal", "natural", "glam", "bold"] | None = None
    lip_preferences: list[str] = field(default_factory=list)

    # Learning data
    liked_outfits: list[str] = field(default_factory=list)
    feedback_history: list[dict] = field(default_factory=list)

    # Profile completeness
    onboarding_complete: bool = False
    profile_confidence: float = 0.0


@dataclass
class WardrobeItem:
    """A single item in the user's wardrobe."""

    item_id: str
    category: str  # tops, bottoms, dresses, outerwear, shoes, accessories
    subcategory: str
    colors: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    occasions: list[str] = field(default_factory=list)
    seasons: list[str] = field(default_factory=list)
    image_data: str | None = None  # base64 encoded thumbnail
    notes: str | None = None
    wear_count: int = 0


@dataclass
class UserContext:
    """Persistent user context and preferences."""

    telegram_id: int
    username: str | None = None
    first_name: str | None = None

    # User preferences (populated over time)
    location: str | None = None
    fitness_goals: list[str] = field(default_factory=list)
    preferred_workout_types: list[str] = field(default_factory=list)
    dietary_preferences: list[str] = field(default_factory=list)

    # Style profile (AI Stylist)
    style_profile: StyleProfile | None = None
    wardrobe: list[WardrobeItem] = field(default_factory=list)

    # Session state
    conversation_summary: str | None = None


@dataclass
class AgentState:
    """State object passed through the LangGraph agent."""

    # Conversation messages (with LangGraph's add_messages reducer)
    messages: Annotated[list[BaseMessage], add_messages]

    # User context
    user: UserContext | None = None

    # Current turn metadata
    current_intent: str | None = None
    selected_tools: list[str] = field(default_factory=list)

    # Control flow
    next_action: Literal["continue", "respond", "end"] = "continue"
    error: str | None = None
