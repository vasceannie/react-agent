"""Utility & helper functions."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.base import Runnable

from react_agent.configuration import Configuration
from react_agent.state import State
from react_agent.tools.tavily import TOOLS


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    provider: str = fully_specified_name.split("/", maxsplit=1)[0]
    model: str = fully_specified_name.split("/", maxsplit=1)[1]
    return init_chat_model(model, model_provider=provider)


async def call_model(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration: Configuration = Configuration.from_runnable_config(config)

    # Initialize the model with tool binding. Change the model or add more tools here.
    model: Runnable[PromptValue | str | Sequence[BaseMessage | list[str] | tuple[str, str] | str | dict[str, Any]], BaseMessage] = load_chat_model(configuration.model).bind_tools(TOOLS)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=timezone.utc).isoformat()
    )

    # Get the model's response
    response: AIMessage = cast(
        AIMessage,
        await model.ainvoke(
            input=[{"role": "system", "content": system_message}, *state.messages],
            config=config,
        ),
    )

    return {"messages": [response]}


async def call_model_json(
    messages: List[Dict[str, str]],
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Call the LLM and get a JSON response.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        config: Optional configuration for the model run
        
    Returns:
        Dict containing parsed JSON response
    """
    base_config: RunnableConfig = config or {}
    configuration: Configuration = Configuration.from_runnable_config(base_config)
    model: BaseChatModel = load_chat_model(configuration.model)
    
    # Create a new config dictionary with response format
    model_config = dict(base_config)
    model_config["response_format"] = {"type": "json_object"}
    
    response = await model.ainvoke(
        messages,
        config=cast(RunnableConfig, model_config)
    )
    
    content = get_message_text(response)
    return json.loads(content)