"""Tool registry for the agent."""

from .places import register_places_tools
from .preferences import get_user_preferences, register_preference_tools
from .registry import ToolRegistry, tool_registry
from .stylist import register_stylist_tools

__all__ = [
    "ToolRegistry",
    "tool_registry",
    "register_places_tools",
    "register_preference_tools",
    "register_stylist_tools",
    "get_user_preferences",
]


def register_all_tools() -> None:
    """Register all available tools with the registry."""
    register_places_tools()
    register_preference_tools()
    register_stylist_tools()
