"""Agent state definition for LangGraph."""

from dataclasses import dataclass, field
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


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
