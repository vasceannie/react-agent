"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Sequence, cast

from langchain_core.messages import AIMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.base import Runnable
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools.tavily import TOOLS
from react_agent.utils.llm import LLMClient

llm_client = LLMClient(default_model="openai/gpt-4")
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the two nodes we will cycle between
builder.add_node("llm_chat", llm_client.llm_chat)
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as `llm_chat`
# This means that this node is the first one called
builder.add_edge("__start__", "llm_chat")


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    return "tools" if last_message.tool_calls else "__end__"


# Add a conditional edge to determine the next step after `llm_chat`
builder.add_conditional_edges(
    "llm_chat",
    # After llm_chat finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `llm_chat`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "llm_chat")

# Compile the builder into an executable graph
# You can customize this by adding interrupt points for state updates
graph = builder.compile(
    interrupt_before=[],  # Add node names here to update state before they're called
    interrupt_after=[],  # Add node names here to update state after they're called
)
graph.name = "ReAct Agent"  # This customizes the name in LangSmith
