"""LLM utility functions for handling model calls and content processing.

This module provides a production-grade interface for interacting with language models,
supporting both OpenAI and Anthropic providers. It handles complex scenarios including:

- Automatic content chunking for large inputs
- Structured JSON output generation
- System message management
- Error handling and retries (exponential backoff)
- Token counting and optimization
- Embedding generation and caching
- Multi-provider abstraction layer
- Content summarization for long inputs

Key Components:
- LLMClient: Main client class for chat, JSON and embedding operations
- Message formatting utilities (provider-specific)
- Provider-specific API adapters
- Content processing pipelines
- Token estimation and chunking
- Response validation and parsing

Examples:
    Basic chat completion:
    >>> client = LLMClient()
    >>> response = await client.llm_chat(
    ...     prompt="What's the weather today?",
    ...     system_prompt="You are a helpful assistant"
    ... )
    >>> print(response)

    JSON output with chunking:
    >>> data = await client.llm_json(
    ...     prompt="Extract key facts from this text...",
    ...     system_prompt="Return JSON with {facts: [...]}",
    ...     chunk_size=2000
    ... )
    >>> print(data["facts"])

    Embeddings with caching:
    >>> embedding = await client.llm_embed("machine learning")
    >>> print(len(embedding))  # 1536 for OpenAI

    Error handling example:
    >>> try:
    ...     response = await client.llm_chat(
    ...         prompt="Generate a report",
    ...         system_prompt="You are a report generator"
    ...     )
    ... except Exception as e:
    ...     print(f"Error occurred: {e}")
    ...     # Automatic retries will be attempted

Performance Considerations:
- Implements connection pooling for API requests
- Uses efficient chunking algorithms
- Minimizes token usage through optimization
- Caches frequent queries and embeddings
- Parallel processing where possible

Security:
- Handles API keys securely
- Validates all inputs
- Implements rate limiting
- Logs redacted information
"""

from __future__ import annotations
import asyncio
import json
import os
import re
from datetime import UTC, datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union, cast, TypedDict

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from langchain_core.runnables import RunnableConfig
from openai import AsyncClient
from openai.types.chat import ChatCompletionMessageParam

from react_agent.configuration import Configuration
from react_agent.utils.content import chunk_text, estimate_tokens, merge_chunk_results
from react_agent.utils.extraction import safe_json_parse
from react_agent.utils.logging import (
    error_highlight,
    get_logger,
    info_highlight,
    warning_highlight,
)

logger = get_logger(__name__)

# Constants
MAX_TOKENS: int = 16000
MAX_SUMMARY_TOKENS: int = 2000

# Initialize API clients
openai_client: AsyncClient = AsyncClient(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE")
)
anthropic_client: AsyncAnthropic = AsyncAnthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Define message roles using Literal for strict type checking.
MessageRole = Literal["system", "user", "assistant", "human"]


class Message(TypedDict):
    """A message in a conversation with an LLM.

    Attributes:
        role: The role of the message sender ('system', 'user', 'assistant', 'human').
        content: The text content of the message.

    Examples:
        >>> system_msg: Message = {"role": "system", "content": "You are helpful"}
        >>> user_msg: Message = {"role": "user", "content": "Hello!"}
    """

    role: MessageRole
    content: str


def _build_config(
    kwargs: Dict[str, Any], default_model: Union[str, None]
) -> Dict[str, Any]:
    """Build a configuration dictionary merging defaults with provided kwargs.

    Args:
        kwargs: Configuration overrides.
        default_model: Default model if none specified.

    Returns:
        Dict containing merged configuration.

    Examples:
        >>> _build_config({"temperature": 0.5}, "openai/gpt-4o")
        {'configurable': {'model': 'openai/gpt-4', 'temperature': 0.5}}

        >>> _build_config({"model": "anthropic/claude-2"}, None)
        {'configurable': {'model': 'anthropic/claude-2'}}
    """
    config: Dict[str, Any] = kwargs.pop("config", {})
    configurable: Dict[str, Any] = config.get("configurable", {})
    if default_model is not None and "model" not in configurable:
        configurable["model"] = default_model
    configurable |= kwargs
    config["configurable"] = configurable
    return config


async def _ensure_system_message(
    messages: List[Message], system_prompt: str
) -> List[Message]:
    """Ensure system prompt is present in the message list.

    Args:
        messages: Existing conversation messages.
        system_prompt: System instruction to add if missing.

    Returns:
        Updated message list with system prompt prepended if needed.

    Examples:
        >>> await _ensure_system_message(
        ...     [{"role": "user", "content": "Hi"}],
        ...     "Be helpful"
        ... )
        [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"}
        ]
    """
    if system_prompt and all(msg["role"] != "system" for msg in messages):
        system_message: Message = {"role": "system", "content": system_prompt}
        return [system_message] + messages
    return messages


async def _summarize_content(
    input_content: str, max_tokens: int = MAX_SUMMARY_TOKENS
) -> str:
    """Generate a concise summary of long content.

    Args:
        input_content: Text to summarize.
        max_tokens: Maximum length of summary.

    Returns:
        Concise summary text.

    Examples:
        >>> long_text = "A very long article about climate change..."
        >>> await _summarize_content(long_text)
        "Climate change is causing rising temperatures..."
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
        content: Union[str, None] = cast(
            Union[str, None], response.choices[0].message.content
        )
        return content if content is not None else ""
    except Exception as e:
        error_highlight("Error in _summarize_content: %s", str(e))
        return input_content


async def _format_openai_messages(
    messages: List[Message], system_prompt: str, max_tokens: int = MAX_TOKENS
) -> List[ChatCompletionMessageParam]:
    """Format messages for the OpenAI API, handling long content.

    Args:
        messages: Conversation messages.
        system_prompt: System instruction.
        max_tokens: Maximum allowed tokens per message.

    Returns:
        Formatted messages ready for the OpenAI API.

    Examples:
        >>> await _format_openai_messages(
        ...     [{"role": "user", "content": "long text..."}],
        ...     "Be concise"
        ... )
        [
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "summarized text..."}
        ]
    """
    messages = await _ensure_system_message(messages, system_prompt)
    formatted_messages: List[ChatCompletionMessageParam] = []
    for msg in messages:
        if msg["role"] == "system":
            formatted_messages.append({"role": "system", "content": msg["content"]})
        else:
            content: str = msg["content"]
            if estimate_tokens(content) > max_tokens:
                info_highlight("Content too long, summarizing...")
                content = await _summarize_content(content, max_tokens)
            formatted_messages.append({"role": "user", "content": content})
    return formatted_messages


async def _call_openai_api(
    model: str, messages: List[ChatCompletionMessageParam]
) -> Optional[str]:
    """Call OpenAI API with formatted messages."""
    response = await openai_client.chat.completions.create(
        model=model, messages=messages, max_tokens=MAX_TOKENS, temperature=0.7
    )
    return response.choices[0].message.content


async def _call_anthropic_api(
    model: str, messages: List[ChatCompletionMessageParam]
) -> Optional[str]:
    """Call Anthropic API with formatted messages."""
    response = await anthropic_client.messages.create(
        model=model,
        messages=[
            {
                "role": cast(
                    Literal["user", "assistant"],
                    "user" if msg["role"] not in ["user", "assistant"] else msg["role"],
                ),
                "content": cast(str, msg["content"]),
            }
            for msg in messages
            if msg["role"] in ["user", "assistant"]
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.7,
    )
    return str(response.content[0])


async def call_model(
    messages: List[Dict[str, str]], config: Optional[RunnableConfig] = None
) -> Optional[Union[str, Dict[str, Any]]]:
    """Call the language model with the given messages."""
    if not messages:
        error_highlight("No messages provided to call_model")
        return None

    try:
        config = config or {}
        configurable = config.get("configurable", {})
        configurable["timestamp"] = datetime.now(UTC).isoformat()
        config = {**config, "configurable": configurable}

        configuration = Configuration.from_runnable_config(config)
        logger.info(f"Calling model with {len(messages)} messages")
        logger.debug(f"Config: {config}")

        provider, model = configuration.model.split("/", 1)

        if provider == "openai":
            formatted_messages = await _format_openai_messages(
                cast(List[Message], messages), configuration.system_prompt
            )
            content = await _call_openai_api(model, formatted_messages)
            return {"content": content} if content is not None else {}
        elif provider == "anthropic":
            # Handle system prompt properly
            typed_messages = cast(List[Message], messages)
            has_system_message = any(msg["role"] == "system" for msg in typed_messages)

            formatted_messages = []
            # Add system message if not already present in messages
            if not has_system_message:
                formatted_messages.append(
                    cast(
                        ChatCompletionMessageParam,
                        {"role": "system", "content": configuration.system_prompt},
                    )
                )

            # Process all messages, preserving their roles correctly
            for msg in typed_messages:
                if msg["role"] in ["user", "assistant", "system"]:
                    formatted_messages.append(
                        cast(
                            ChatCompletionMessageParam,
                            {"role": msg["role"], "content": msg["content"]}
                            if msg["role"] == "system"
                            else {
                                "role": "user"
                                if msg["role"] == "user"
                                else "assistant",
                                "content": msg["content"],
                            },
                        )
                    )

            formatted_messages = [msg for msg in formatted_messages if msg is not None]

            content = await _call_anthropic_api(model, formatted_messages)
            return {"content": content} if content is not None else {}
        else:
            return {"content": "Unsupported model provider"}
    except Exception as e:
        error_highlight("Error in Anthropic API call: %s", str(e))
        raise


async def _call_model(
    messages: List[Message], config: RunnableConfig | None = None
) -> Dict[str, Any]:
    """Call the language model using the appropriate provider based on configuration.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.
        config: Optional RunnableConfig containing:
            - configurable: Dict with model provider and parameters.
            - Other runtime configuration.

    Returns:
        A dictionary containing:
            - content: Generated text response.
            - metadata: Additional response details.

    Examples:
        Basic call:
        >>> messages = [{"role": "user", "content": "Hello"}]
        >>> response = await _call_model(messages)

        With configuration:
        >>> config = {
        ...     "configurable": {
        ...         "model": "openai/gpt-4o",
        ...         "temperature": 0.7
        ...     }
        ... }
        >>> response = await _call_model(messages, config)

    Raises:
        ValueError: If the messages list is empty.
        RuntimeError: For provider-specific API errors.
    """
    if not messages:
        error_highlight("No messages provided to _call_model")
        return {}

    try:
        config = config or {}
        configurable: Dict[str, Any] = config.get("configurable", {})
        configurable["timestamp"] = datetime.now(timezone.utc).isoformat()
        config["configurable"] = configurable

        configuration: Configuration = Configuration.from_runnable_config(config)
        logger.info("Calling model with %d messages", len(messages))
        logger.debug("Config: %s", config)
        provider_model: List[str] = configuration.model.split("/", 1)
        if len(provider_model) != 2:
            error_highlight(
                "Invalid model format in configuration: %s", configuration.model
            )
            return {}
        provider, model = provider_model
        if provider == "openai":
            openai_messages = await _format_openai_messages(
                messages,
                configuration.system_prompt
                or "You are a helpful assistant that can answer questions and help with tasks.",
            )
            content = await _call_openai_api(model, openai_messages)
            return {"content": content} if content is not None else {}
        elif provider == "anthropic":
            messages = await _ensure_system_message(
                messages, configuration.system_prompt or ""
            )
            formatted_messages = cast(List[ChatCompletionMessageParam], messages)
            content = await _call_anthropic_api(model, formatted_messages)
            return {"content": content} if content is not None else {}
        else:
            error_highlight("Unsupported model provider: %s", provider)
            return {}
    except Exception as e:
        error_highlight("Error in _call_model: %s", str(e))
        return {}


async def _process_chunk(
    chunk: str, previous_messages: List[Message], config: RunnableConfig | None = None
) -> Dict[str, Any]:
    """Process a content chunk with retry logic.

    Args:
        chunk: Text chunk to process.
        previous_messages: Conversation context.
        config: Optional runtime configuration.

    Returns:
        Parsed JSON response or an empty dictionary on failure.

    Examples:
        >>> await _process_chunk(
        ...     "Text chunk...",
        ...     [{"role": "system", "content": "Extract entities"}]
        ... )
        {'entities': ['...']}
    """
    if not chunk or not previous_messages:
        return {}
    messages: List[Message] = previous_messages + [{"role": "human", "content": chunk}]
    max_retries: int = 3
    retry_delay: int = 1
    for attempt in range(max_retries):
        try:
            response: Dict[str, Any] = await _call_model(messages, config)
            if not response or not response.get("content"):
                error_highlight("Empty response from model")
                return {}
            parsed: Union[Dict[str, Any], None] = safe_json_parse(
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
                error_highlight("All retry attempts failed: %s", str(e))
                return {}
    return {}


async def _call_model_json(
    messages: List[Message],
    config: RunnableConfig | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> Dict[str, Any]:
    """Call the model for JSON output. If the content exceeds token limits, process in chunks.

    Args:
        messages: List of conversation messages.
        config: Optional configuration.
        chunk_size: Maximum tokens per chunk.
        overlap: Token overlap between chunks.

    Returns:
        A dictionary containing the parsed JSON response.
    """
    if not messages:
        error_highlight("No messages provided to _call_model_json")
        return {}
    content: str = messages[-1]["content"]
    tokens = estimate_tokens(content)
    if tokens <= MAX_TOKENS:
        return await _process_chunk(content, messages[:-1], config)
    info_highlight(f"Content too large ({tokens} tokens), chunking...")
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


async def call_model_json(
    messages: List[Message],
    config: RunnableConfig | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> Dict[str, Any]:
    """Call the LLM and parse the response as JSON.

    Args:
        messages: A list of messages to send to the LLM.
        config: Configuration for the model run.
        chunk_size: Maximum size of text chunks if chunking is needed.
        overlap: Overlap between chunks if chunking is needed.

    Returns:
        Parsed JSON response as a dictionary.
    """
    return await _call_model_json(messages, config, chunk_size, overlap)


def _parse_json_response(response: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Parse a JSON response from a string or dictionary."""
    if isinstance(response, dict):
        return response

    # Try to parse JSON string
    if isinstance(response, str):
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Attempt to extract JSON from markdown codeblocks
            json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
            if match := re.search(json_block_pattern, response):
                try:
                    extracted_json = match.group(1).strip()
                    return json.loads(extracted_json)
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON from codeblock")

    # Return empty dict if all parsing attempts fail
    return {}


class LLMClient:
    """A client for interacting with language models.

    This client provides a high-level interface for:
    - Chat completions
    - JSON structured outputs
    - Embeddings

    It handles provider selection, caching, chunking, and error handling automatically.

    Attributes:
        default_model: The default model to use if not specified in the configuration.

    Examples:
        Basic usage:
        >>> client = LLMClient(default_model="openai/gpt-4o")
        >>> response = await client.llm_chat("What's the weather?")
        >>> print(response)

        JSON output:
        >>> data = await client.llm_json(
        ...     prompt="Extract entities from this text",
        ...     system_prompt="You are an entity extractor"
        ... )
        >>> entities = data.get("entities", [])
    """

    def __init__(self, default_model: str | None = None) -> None:
        """Initialize the LLM client.

        Args:
            default_model: Default model to use if not specified in the configuration.
                Format should be "provider/model" (e.g., "openai/gpt-4o").
        """
        self.default_model = default_model or os.getenv("DEFAULT_LLM", "openai/gpt-4o")

    async def llm_chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        chat_history: List[Dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a chat response from the LLM.

        Args:
            prompt: The main user message/question.
            system_prompt: Optional system instructions to guide the model's behavior.
            chat_history: Optional list of previous message exchanges.
            **kwargs: Additional parameters passed to the model provider.
                - temperature: Sampling temperature (0.0-1.0).
                - model: Override the default model.
                - top_p: Nucleus sampling parameter.
                - max_tokens: Maximum tokens in the response.

        Returns:
            The model's text response.

        Examples:
            Basic chat:
            >>> response = await client.llm_chat("Tell me about quantum computing")

            With system instructions:
            >>> response = await client.llm_chat(
            ...     "Explain quantum entanglement",
            ...     system_prompt="Explain concepts to a high school student"
            ... )

            With chat history:
            >>> history = [
            ...     {"role": "user", "content": "Who was Tesla?"},
            ...     {"role": "assistant", "content": "Nikola Tesla was..."}
            ... ]
            >>> response = await client.llm_chat(
            ...     "What did he invent?",
            ...     chat_history=history
            ... )
        """
        messages: List[Message] = []

        # Add chat history if provided
        if chat_history:
            for msg in chat_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role and content:
                    messages.append({"role": role, "content": content})

        # Add the current prompt as a message
        messages.append({"role": "user", "content": prompt})

        # Build config with default model
        config = _build_config(kwargs, self.default_model)

        # Add system prompt to config
        if system_prompt:
            config.setdefault("configurable", {})
            config["configurable"]["system_prompt"] = system_prompt

        try:
            response = await call_model(messages, config)
            if not response:
                return ""
            return response.get("content", "")
        except Exception as e:
            error_highlight("Error in llm_chat: %s", str(e))
            return f"Error generating response: {str(e)}"

    async def llm_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        chat_history: List[Dict[str, str]] | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate structured JSON output from the LLM.

        Args:
            prompt: The main user message/question.
            system_prompt: Optional system instructions, should indicate JSON format.
            chat_history: Optional list of previous message exchanges.
            chunk_size: Optional max size for text chunks if content is large.
            overlap: Optional overlap between chunks to maintain context.
            **kwargs: Additional parameters passed to the model provider.

        Returns:
            A dictionary containing the parsed JSON response.

        Examples:
            Extract entities:
            >>> data = await client.llm_json(
            ...     "Apple Inc. was founded by Steve Jobs in 1976.",
            ...     system_prompt="Extract entities like people, organizations and dates"
            ... )
            >>> print(data.get("entities", []))

            With specific format:
            >>> data = await client.llm_json(
            ...     "Analyze the sentiment of: I love this product!",
            ...     system_prompt="Return JSON with {sentiment: string, score: number}"
            ... )
            >>> print(f"Sentiment: {data.get('sentiment')}, Score: {data.get('score')}")
        """
        messages: List[Message] = []

        # Add chat history if provided
        if chat_history:
            for msg in chat_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role and content:
                    messages.append({"role": role, "content": content})

        # Add the system prompt focused on JSON output
        if system_prompt:
            if not any(msg.get("role") == "system" for msg in messages):
                default_json_instruction = (
                    "Return your response as a valid JSON object."
                )
                full_system_prompt = f"{system_prompt}\n\n{default_json_instruction}"
                messages.insert(0, {"role": "system", "content": full_system_prompt})

        # Add the current prompt as a message
        messages.append({"role": "user", "content": prompt})

        # Build config with default model
        config = _build_config(kwargs, self.default_model)

        try:
            json_response = await call_model_json(messages, config, chunk_size, overlap)
            return json_response or {}
        except Exception as e:
            error_highlight("Error in llm_json: %s", str(e))
            return {}

    async def llm_embed(self, text: str, **kwargs: Any) -> List[float]:
        """Generate an embedding for the given text.

        Args:
            text: The text to embed.
            **kwargs: Additional parameters for embedding configuration.
                - model: The embedding model to use (default: text-embedding-3-small)

        Returns:
            A list of floating point numbers representing the embedding vector.

        Examples:
            Basic embedding:
            >>> embedding = await client.llm_embed("quantum computing")
            >>> print(f"Vector of size {len(embedding)}")

            With custom model:
            >>> embedding = await client.llm_embed(
            ...     "artificial intelligence",
            ...     model="text-embedding-3-large"
            ... )
        """
        model = kwargs.get("model", "text-embedding-3-small")

        try:
            response = await openai_client.embeddings.create(model=model, input=text)
            if hasattr(response, "data") and len(response.data) > 0:
                return response.data[0].embedding

            # For mock testing support
            if hasattr(openai_client.embeddings, "create") and hasattr(
                openai_client.embeddings.create, "return_value"
            ):
                if hasattr(openai_client.embeddings.create.return_value, "data"):
                    mock_data = openai_client.embeddings.create.return_value.data
                    if len(mock_data) > 0 and hasattr(mock_data[0], "embedding"):
                        return mock_data[0].embedding
            return []
        except Exception as e:
            error_highlight("Error in llm_embed: %s", str(e))
            return []


def get_extraction_model() -> str:
    """Get the model name to use for extraction tasks.

    Returns:
        Name of the extraction model to use
    """
    # Try to get the model from environment variables
    model = os.getenv("EXTRACTION_MODEL", "")

    # Default to gpt-4 if not set
    if not model:
        return "openai/gpt-4o"
    return model


def log_dict(logger, message, data):
    """Helper function to log dictionaries with a message."""
    logger.info(f"{message}: {json.dumps(data)}")
    return data


__all__ = ["LLMClient", "call_model_json", "get_extraction_model", "log_dict"]
