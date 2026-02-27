"""Tool registry pattern for dynamic tool management."""

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool


@dataclass
class ToolConfig:
    """Configuration for a registered tool."""

    name: str
    description: str
    func: Callable[..., Any]
    args_schema: type | None = None

    # Rate limiting
    calls_per_minute: int = 60
    burst_limit: int = 10

    # Resilience
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    circuit_breaker_threshold: int = 5

    # Caching
    cache_ttl_seconds: int = 0  # 0 = no caching

    # Fallback
    fallback_func: Callable[..., Any] | None = None


class ToolRegistry:
    """Registry for managing agent tools with rate limiting and fallbacks."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolConfig] = {}

    def register(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        args_schema: type | None = None,
        **kwargs: Any,
    ) -> None:
        """Register a tool with the registry."""
        self._tools[name] = ToolConfig(
            name=name,
            description=description,
            func=func,
            args_schema=args_schema,
            **kwargs,
        )

    def get(self, name: str) -> ToolConfig | None:
        """Get a tool configuration by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def to_langchain_tools(self) -> list[BaseTool]:
        """Convert registered tools to LangChain tool format."""
        tools: list[BaseTool] = []
        for config in self._tools.values():
            # Check if function is async
            is_async = asyncio.iscoroutinefunction(config.func)

            tool = StructuredTool.from_function(
                func=config.func,
                name=config.name,
                description=config.description,
                args_schema=config.args_schema,
                coroutine=config.func if is_async else None,
            )
            tools.append(tool)
        return tools

    def __len__(self) -> int:
        return len(self._tools)


# Global tool registry instance
tool_registry = ToolRegistry()
