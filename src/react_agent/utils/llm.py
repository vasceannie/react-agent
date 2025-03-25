"""LLM utility functions for handling model calls and content processing.

This module provides utilities for interacting with language models,
including content length management, error handling, and JSON parsing.
The LLMClient class consolidates chat, JSON, and embedding calls into a
single interface, minimizing breaking changes while providing a strictly typed
interface for LangGraph nodes.

Examples:
    Basic usage of call_model:

    ```python
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    response = await _call_model(messages)
    # Returns: {"content": "The capital of France is Paris."}
    ```

    Using _call_model_json with chunking:

    ```python
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Analyze this long text: " + large_document}
    ]
    config = {"configurable": {"model": "openai/gpt-4"}}
    response = await _call_model_json(
        messages,
        config=config,
        chunk_size=1000,  # Process in 1000-token chunks
        overlap=100       # 100-token overlap between chunks
    )
    # Returns a merged JSON response.
    ```

    Using the LLMClient for LangGraph nodes:

    ```python
    # Initialize the client
    llm_client = LLMClient(default_model="openai/gpt-4")

    # Use in an async function or LangGraph node
    async def my_node(state: dict) -> dict:
        # Simple chat completion
        response = await llm_client.llm_chat(
            prompt="What is the capital of France?",
            system_prompt="You are a helpful assistant."
        )

        # Get structured JSON output
        data = await llm_client.llm_json(
            prompt="List the top 3 largest countries by area in JSON format",
            system_prompt="You are a helpful assistant that outputs valid JSON."
        )

        # Get embeddings
        embedding = await llm_client.llm_embed("This is a text to embed")

        return {"response": response, "data": data, "embedding": embedding}
    ```
"""

import asyncio
import json
import os
import re
from datetime import UTC, datetime
from typing import Any, Dict, Iterable, List, TypedDict, Union, cast

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from langchain_core.runnables import RunnableConfig
from openai import AsyncClient
from openai.types.chat import ChatCompletionMessageParam

from react_agent.configuration import Configuration
from react_agent.utils.content import (
    chunk_text,
    estimate_tokens,
    merge_chunk_results,
    preprocess_content,
    validate_content,
)
from react_agent.utils.defaults import ChunkConfig
from react_agent.utils.extraction import safe_json_parse
from react_agent.utils.logging import (
    error_highlight,
    get_logger,
    info_highlight,
    warning_highlight,
)

logger = get_logger(__name__)

# Constants for token limits
MAX_TOKENS: int = 16000
MAX_SUMMARY_TOKENS: int = 2000

# Initialize API clients
openai_client: AsyncClient = AsyncClient(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE")
)
anthropic_client: AsyncAnthropic = AsyncAnthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


class Message(TypedDict):
    role: str
    content: str


async def _summarize_content(
    input_content: str, max_tokens: int = MAX_SUMMARY_TOKENS
) -> str:
    """Summarize content using a more efficient model.

    Args:
        input_content: The content to summarize.
        max_tokens: Maximum tokens for the summary.

    Returns:
        A concise summary of the input content.
    """
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that creates concise summaries. "
                        "Focus on key points and maintain factual accuracy."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Please summarize the following content concisely:\n\n{input_content}",
                },
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        response_content: str | None = cast(
            str | None, response.choices[0].message.content
        )
        return response_content if response_content is not None else ""
    except Exception as e:
        error_highlight(f"Error in _summarize_content: {str(e)}")
        return input_content


async def _format_openai_messages(
    messages: List[Message], system_prompt: str, max_tokens: int | None = None
) -> List[ChatCompletionMessageParam]:
    """Format messages for the OpenAI API with content handling.

    Args:
        messages: List of message dictionaries.
        system_prompt: System prompt to include.
        max_tokens: Optional maximum tokens per message.

    Returns:
        A list of formatted messages for the API.
    """
    if not messages:
        return [{"role": "system", "content": system_prompt}]

    formatted_messages: List[ChatCompletionMessageParam] = []
    max_tokens = max_tokens or MAX_TOKENS

    for msg in messages:
        if msg["role"] == "system":
            formatted_messages.append({"role": "system", "content": msg["content"]})
        else:
            content: str = msg["content"]
            if estimate_tokens(content) > max_tokens:
                info_highlight("Content too long, summarizing...")
                content = await _summarize_content(content, max_tokens)
            formatted_messages.append({"role": "user", "content": content})

    if all(msg["role"] != "system" for msg in formatted_messages):
        formatted_messages.insert(0, {"role": "system", "content": system_prompt})

    return formatted_messages


async def _call_openai_api(
    model: str, messages: List[ChatCompletionMessageParam]
) -> Dict[str, Any]:
    """Call the OpenAI API using formatted messages.

    Args:
        model: The model identifier.
        messages: Formatted messages for the API.

    Returns:
        A dictionary containing the model's response.
    """
    try:
        response = await openai_client.chat.completions.create(
            model=model, messages=messages, max_tokens=MAX_TOKENS, temperature=0.7
        )
        content: str | None = response.choices[0].message.content
        return {"content": content} if content else {}
    except Exception as e:
        error_highlight(f"Error in _call_openai_api: {str(e)}")
        return {}


async def _call_model(
    messages: List[Message], config: RunnableConfig | None = None
) -> Dict[str, Any]:
    """Call the language model with the provided messages.

    Args:
        messages: List of messages (each with 'role' and 'content').
        config: Optional configuration for the model call.

    Returns:
        A dictionary with the model's output under the key 'content'.
    """
    if not messages:
        error_highlight("No messages provided to _call_model")
        return {}

    try:
        config = config or {}
        configurable: Dict[str, Any] = config.get("configurable", {})
        configurable["timestamp"] = datetime.now(UTC).isoformat()
        config = {**config, "configurable": configurable}
        configuration: Configuration = Configuration.from_runnable_config(config)
        logger.info(f"Calling model with {len(messages)} messages")
        logger.debug(f"Config: {config}")
        provider, model = configuration.model.split("/", 1)
        if provider == "openai":
            openai_formatted_messages = await _format_openai_messages(
                messages, configuration.system_prompt
            )
            return await _call_openai_api(model, openai_formatted_messages)
        elif provider == "anthropic":
            # Check if a system message is already present
            has_system_message: bool = any(msg["role"] == "system" for msg in messages)

            # Create properly typed messages for Anthropic
            anthropic_formatted_messages: List[Dict[str, str]] = []

            # Add system message if not already present
            if not has_system_message and configuration.system_prompt:
                # For Anthropic, we need to handle system messages differently
                # as they have specific role requirements
                anthropic_formatted_messages.append({
                    "role": "user",  # Using user role as a workaround for system
                    "content": f"System instruction: {configuration.system_prompt}",
                })

            for msg in messages:
                # Anthropic only accepts user and assistant roles
                if msg["role"] == "system":
                    # Convert system messages to user messages with a prefix
                    anthropic_formatted_messages.append({
                        "role": "user",
                        "content": f"System instruction: {msg['content']}",
                    })
                elif msg["role"] in ["user", "assistant"]:
                    anthropic_formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"],
                    })

            # Make the API call
            try:
                # Cast the messages to the expected type for Anthropic
                typed_messages = cast(
                    Iterable[MessageParam], anthropic_formatted_messages
                )
                response = await anthropic_client.messages.create(
                    model=model,
                    messages=typed_messages,
                    max_tokens=MAX_TOKENS,
                    temperature=0.7,
                )
                # Access the content correctly based on the response structure
                content_block = response.content[0]
                # Check the type of content block and extract text appropriately
                if hasattr(content_block, "text"):
                    return {"content": content_block.text}
                else:
                    return {"content": str(content_block)}
            except Exception as e:
                error_highlight(f"Error in Anthropic API call: {str(e)}")
                raise
        else:
            error_highlight(f"Unsupported model provider: {provider}")
            return {}
    except Exception as e:
        error_highlight(f"Error in _call_model: {str(e)}")
        raise


async def _process_chunk(
    chunk: str,
    previous_messages: List[Message],
    config: RunnableConfig | None = None,
    model: str | None = None,
) -> Dict[str, Any]:
    """Process a single chunk of content with error handling.

    Args:
        chunk: The content chunk to process.
        previous_messages: Previous conversation messages.
        config: Optional configuration.
        model: Optional model override.

    Returns:
        A dictionary with the processed chunk result.
    """
    if not chunk or not previous_messages:
        return {}
    try:
        messages: List[Message] = previous_messages + [
            {"role": "human", "content": chunk}
        ]
        max_retries: int = 3
        retry_delay: int = 1
        for attempt in range(max_retries):
            try:
                # Convert dict to RunnableConfig if needed
                runnable_config: RunnableConfig | None = config
                if (
                    config
                    and not isinstance(config, dict)
                    and not isinstance(config, RunnableConfig)
                ):
                    runnable_config = cast(RunnableConfig, config)

                response = await _call_model(messages, runnable_config)
                if not response or not response.get("content"):
                    error_highlight("Empty response from model")
                    return {}
                parsed: Dict[str, Any] = safe_json_parse(
                    response["content"], "model_response"
                )
                return parsed if parsed is not None else {}
            except Exception as e:
                if attempt < max_retries - 1:
                    warning_highlight(
                        f"Attempt {attempt + 1} failed: {str(e)}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    error_highlight(f"All retry attempts failed: {str(e)}")
                    return {}
    except Exception as e:
        error_highlight(f"Error processing chunk: {str(e)}")
        return {}
    return {}


async def _call_model_json(
    messages: List[Message],
    config: RunnableConfig | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> Dict[str, Any]:
    """Call the model for JSON output, handling chunking if necessary.

    Args:
        messages: List of message dictionaries.
        config: Optional configuration for the model call.
        chunk_size: Optional custom chunk size.
        overlap: Optional token overlap between chunks.

    Returns:
        A dictionary containing the merged JSON response.
    """
    try:
        if not messages:
            error_highlight("No messages provided to _call_model_json")
            return {}
        content: str = messages[-1]["content"]
        if estimate_tokens(content) <= MAX_TOKENS:
            return await _process_chunk(content, messages[:-1], config)
        info_highlight(
            f"Content too large ({estimate_tokens(content)} tokens), chunking..."
        )
        chunks: List[str] = chunk_text(
            content, chunk_size=chunk_size, overlap=overlap, use_large_chunks=True
        )
        if len(chunks) <= 1:
            return await _process_chunk(content, messages[:-1], config)
        chunk_results: List[Dict[str, Any]] = []
        for chunk in chunks:
            result: Dict[str, Any] = await _process_chunk(chunk, messages[:-1], config)
            if result:
                chunk_results.append(result)
        if not chunk_results:
            error_highlight("No valid results from chunks")
            return {}
        return merge_chunk_results(chunk_results, "model_response")
    except Exception as e:
        error_highlight(f"Error in _call_model_json: {str(e)}")
        return {}


def _parse_json_response(response: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Parse and clean a JSON response from the model.

    Args:
        response: The raw response as a string or dictionary.

    Returns:
        A dictionary parsed from the JSON response.
    """
    if isinstance(response, dict):
        return response
    try:
        cleaned: str = re.sub(r"```json\s*|\s*```", "", response)
        return json.loads(cleaned)
    except Exception:
        return {}


class LLMClient:
    """Asynchronous LLM utility for chat, JSON output, and embeddings.

    Provides a consolidated interface for calling language models,
    abstracting away the details of message formatting, chunking, and error handling.

    Attributes:
        default_model: An optional default model to use for calls.
    """

    def __init__(self, default_model: str | None = None) -> None:
        """Initialize the LLMClient.

        Args:
            default_model: Optional default model name (e.g., "openai/gpt-4").
                If not provided, defaults from configuration are used.
        """
        self.default_model: str | None = default_model

    async def llm_chat(
        self, prompt: str, system_prompt: str | None = None, **kwargs: Any
    ) -> str:
        """Get a chat completion as plain text.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            **kwargs: Additional parameters (e.g., model, temperature).

        Returns:
            The model's response as a string.
        """
        messages: List[Message] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        config: Dict[str, Any] = kwargs.pop("config", {})
        configurable: Dict[str, Any] = config.get("configurable", {})
        if self.default_model and "model" not in configurable:
            configurable["model"] = self.default_model
        for key, value in kwargs.items():
            configurable[key] = value
        config = {**config, "configurable": configurable}

        # Convert dict to RunnableConfig
        runnable_config: RunnableConfig | None = (
            cast(RunnableConfig | None, config) if config else None
        )

        response: Dict[str, Any] = await _call_model(messages, runnable_config)
        return response.get("content", "")

    async def llm_json(
        self, prompt: str, system_prompt: str | None = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Get a structured JSON response as a Python dictionary.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            **kwargs: Additional parameters (e.g., model, temperature, chunk_size, overlap).

        Returns:
            A dictionary containing the parsed JSON response.
        """
        chunk_size: int | None = kwargs.pop("chunk_size", None)
        overlap: int | None = kwargs.pop("overlap", None)
        messages: List[Message] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        config: Dict[str, Any] = kwargs.pop("config", {})
        configurable: Dict[str, Any] = config.get("configurable", {})
        if self.default_model and "model" not in configurable:
            configurable["model"] = self.default_model
        for key, value in kwargs.items():
            configurable[key] = value
        config = {**config, "configurable": configurable}

        # Convert dict to RunnableConfig
        runnable_config: RunnableConfig | None = (
            cast(RunnableConfig | None, config) if config else None
        )

        return await _call_model_json(
            messages=messages,
            config=runnable_config,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    async def llm_embed(self, text: str, **kwargs: Any) -> List[float]:
        """Get embeddings for a given text.

        Args:
            text: The text to embed.
            **kwargs: Additional parameters (e.g., model).

        Returns:
            A list of floats representing the embedding vector.
        """
        config: Dict[str, Any] = kwargs.pop("config", {})
        configurable: Dict[str, Any] = config.get("configurable", {})
        embedding_model: str | None = kwargs.pop("embedding_model", None)
        if embedding_model:
            configurable["model"] = embedding_model
        elif self.default_model and "model" not in configurable:
            configurable["model"] = self.default_model
        for key, value in kwargs.items():
            configurable[key] = value
        config = {**config, "configurable": configurable}

        try:
            # Ensure config is properly typed for from_runnable_config
            runnable_config = cast(RunnableConfig | None, config)
            configuration: Configuration = Configuration.from_runnable_config(
                runnable_config
            )
            provider, model = configuration.model.split("/", 1)
            if provider == "openai":
                response = await openai_client.embeddings.create(
                    model=model, input=text
                )
                return response.data[0].embedding
            else:
                error_highlight(f"Embedding not supported for provider: {provider}")
                return []
        except Exception as e:
            error_highlight(f"Error in llm_embed: {str(e)}")
            return []


__all__ = ["LLMClient"]
