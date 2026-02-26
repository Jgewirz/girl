"""LangGraph agent definition."""

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from src.config import settings
from src.config.logging import get_logger

from .state import AgentState, UserContext
from .tools import tool_registry

logger = get_logger(__name__)

# System prompt that defines the assistant's personality and capabilities
SYSTEM_PROMPT = """You are GirlBot, a friendly and supportive fitness and lifestyle assistant on Telegram.

Your personality:
- Warm, encouraging, and body-positive
- Knowledgeable about fitness, wellness, and self-care
- Helpful without being pushy or judgmental
- Use a conversational, approachable tone

Your capabilities:
- Help users find fitness classes and studios (yoga, pilates, spin, gym, etc.)
- Provide workout suggestions and fitness tips
- Assist with wellness and self-care recommendations
- Answer questions about nutrition and healthy living
- Remember user preferences for personalized recommendations

IMPORTANT - Learning User Preferences:
When users share information about themselves, ALWAYS save it using the appropriate tool:

1. LOCATION - When users mention where they are or want to search:
   - "I'm in San Francisco" → call update_user_location
   - "Find gyms near Brooklyn" → call update_user_location with "Brooklyn"
   - "I live in Austin" → call update_user_location

2. FITNESS GOALS - When users mention what they want to achieve:
   - "I want to lose weight" → call update_fitness_goals
   - "I'm trying to build strength" → call update_fitness_goals
   - "My goal is better flexibility" → call update_fitness_goals

3. WORKOUT PREFERENCES - When users express activity preferences:
   - "I love yoga" → call update_workout_preferences
   - "I prefer pilates" → call update_workout_preferences
   - "I'm really into spinning" → call update_workout_preferences

When searching for studios, ALWAYS include the user's telegram_id in preference tool calls.
If the user has a saved location, use it automatically for searches unless they specify a different location.

Guidelines:
- Keep responses concise and actionable for mobile/chat
- Use emojis sparingly and naturally
- If location is needed but not known, ask the user
- Never provide medical advice - recommend consulting professionals
- Acknowledge when you save preferences ("Got it, I'll remember that!")"""


def _get_llm(complex_task: bool = False) -> ChatOpenAI:
    """Get the appropriate LLM based on task complexity."""
    if not settings.has_openai:
        raise RuntimeError("OpenAI API key not configured")

    model = settings.openai_model_complex if complex_task else settings.openai_model_primary
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key.get_secret_value(),  # type: ignore[union-attr]
        temperature=0.7,
    )


def _build_messages(state: AgentState) -> list[BaseMessage]:
    """Build message list with system prompt."""
    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    # Add user context if available
    if state.user:
        context_parts = [
            f"Current user's telegram_id: {state.user.telegram_id} (use this for all preference tool calls)"
        ]

        if state.user.first_name:
            context_parts.append(f"User's name: {state.user.first_name}")

        if state.user.location:
            context_parts.append(
                f"Saved location: {state.user.location} (use this for searches if no other location specified)"
            )
        else:
            context_parts.append("No location saved yet - ask the user where they are if needed for search")

        if state.user.fitness_goals:
            context_parts.append(f"Fitness goals: {', '.join(state.user.fitness_goals)}")

        if state.user.preferred_workout_types:
            context_parts.append(f"Preferred workouts: {', '.join(state.user.preferred_workout_types)}")

        context_msg = "=== USER CONTEXT ===\n" + "\n".join(context_parts) + "\n==================="
        messages.append(SystemMessage(content=context_msg))

    # Add conversation history
    messages.extend(state.messages)
    return messages


async def process_message(state: AgentState) -> dict[str, Any]:
    """Process user message and generate response."""
    logger.info(
        "processing_message",
        user_id=state.user.telegram_id if state.user else None,
        message_count=len(state.messages),
    )

    try:
        llm = _get_llm(complex_task=False)
        tools = tool_registry.to_langchain_tools()

        if tools:
            llm_with_tools = llm.bind_tools(tools)
        else:
            llm_with_tools = llm

        messages = _build_messages(state)
        response = await llm_with_tools.ainvoke(messages)

        logger.info(
            "llm_response_generated",
            has_tool_calls=bool(response.tool_calls) if hasattr(response, "tool_calls") else False,
        )

        return {"messages": [response], "next_action": "respond"}

    except Exception as e:
        logger.error("llm_error", error=str(e))
        error_response = AIMessage(
            content="I'm having trouble processing that right now. Could you try again?"
        )
        return {"messages": [error_response], "error": str(e), "next_action": "respond"}


async def handle_tool_response(state: AgentState) -> dict[str, Any]:
    """Handle tool execution results and generate final response."""
    # This node handles post-tool-execution response generation
    llm = _get_llm(complex_task=False)
    messages = _build_messages(state)
    response = await llm.ainvoke(messages)
    return {"messages": [response], "next_action": "respond"}


def should_use_tools(state: AgentState) -> str:
    """Determine if we should execute tools or respond directly."""
    last_message = state.messages[-1] if state.messages else None

    if last_message and isinstance(last_message, AIMessage):
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

    return "respond"


def create_agent() -> CompiledStateGraph:
    """Create and compile the LangGraph agent."""
    logger.info("creating_agent", tool_count=len(tool_registry))

    # Build the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("process", process_message)

    # Add tool node if we have tools registered
    tools = tool_registry.to_langchain_tools()
    if tools:
        graph.add_node("tools", ToolNode(tools))
        graph.add_node("post_tools", handle_tool_response)

    # Define edges
    graph.add_edge(START, "process")

    if tools:
        # Conditional routing based on tool calls
        graph.add_conditional_edges(
            "process",
            should_use_tools,
            {"tools": "tools", "respond": END},
        )
        graph.add_edge("tools", "post_tools")
        graph.add_edge("post_tools", END)
    else:
        # No tools, go directly to end
        graph.add_edge("process", END)

    # Compile
    compiled = graph.compile()
    logger.info("agent_compiled")

    return compiled


# Export state for type hints
__all__ = ["create_agent", "AgentState", "UserContext"]
