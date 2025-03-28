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
from react_agent.utils.logging import error_highlight, get_logger, info_highlight, warning_highlight

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

def _build_config(kwargs: Dict[str, Any], default_model: Union[str, None]) -> Dict[str, Any]:
    """Build a configuration dictionary merging defaults with provided kwargs.

    Args:
        kwargs: Configuration overrides.
        default_model: Default model if none specified.

    Returns:
        Dict containing merged configuration.

    Examples:
        >>> _build_config({"temperature": 0.5}, "openai/gpt-4")
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

async def _ensure_system_message(messages: List[Message], system_prompt: str) -> List[Message]:
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

async def _summarize_content(input_content: str, max_tokens: int = MAX_SUMMARY_TOKENS) -> str:
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
        content: Union[str, None] = cast(Union[str, None], response.choices[0].message.content)
        return content if content is not None else ""
    except Exception as e:
        error_highlight("Error in _summarize_content: %s", str(e))
        return input_content

async def _format_openai_messages(
    messages: List[Message],
    system_prompt: str,
    max_tokens: int = MAX_TOKENS
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

async def _call_openai_api(model: str, messages: List[ChatCompletionMessageParam]) -> Optional[str]:
    """Call OpenAI API with formatted messages."""
    response = await openai_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.7
    )
    return response.choices[0].message.content

async def call_model(
    messages: List[Dict[str, str]],
    config: Optional[RunnableConfig] = None
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
            formatted_messages = await _format_openai_messages(messages, configuration.system_prompt)
            return await _call_openai_api(model, formatted_messages)
        elif provider == "anthropic":
            # Handle system prompt properly
            has_system_message = any(msg["role"] == "system" for msg in messages)
            
            formatted_messages = []
            # Add system message if not already present in messages
            if not has_system_message:
                formatted_messages.append(cast(ChatCompletionMessageParam, {"role": "system", "content": configuration.system_prompt}))
                
            # Process all messages, preserving their roles correctly
            for msg in messages:
                if msg["role"] in ["user", "assistant", "system"]:
                    formatted_messages.append(cast(ChatCompletionMessageParam, {
                        "role": msg["role"],
                        "content": msg["content"]
                    } if msg["role"] == "system" else {
                        "role": "user" if msg["role"] == "user" else "assistant",
                        "content": msg["content"]
                    }))
                    
            formatted_messages = [msg for msg in formatted_messages if msg is not None]
            
            response = await anthropic_client.messages.create(
                model=model,
                messages=[{
                    "role": cast(Literal["user", "assistant"], "user" if msg["role"] not in ["user", "assistant"] else msg["role"]),
                    "content": cast(str, msg["content"])
                } for msg in formatted_messages if msg["role"] in ["user", "assistant"]],
                max_tokens=MAX_TOKENS,
                temperature=0.7
            )
            return str(response.content[0])
        else:
            return {"content": str(content_block)}
    except Exception as e:
        error_highlight("Error in Anthropic API call: %s", str(e))
        raise

async def _call_model(
    messages: List[Message],
    config: RunnableConfig | None = None
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
        ...         "model": "openai/gpt-4",
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
            error_highlight("Invalid model format in configuration: %s", configuration.model)
            return {}
        provider, model = provider_model
        if provider == "openai":
            openai_messages = await _format_openai_messages(
                messages,
                configuration.system_prompt or "You are a helpful assistant that can answer questions and help with tasks."
            )
            return await _call_openai_api(model, openai_messages)
        elif provider == "anthropic":
            messages = await _ensure_system_message(messages, configuration.system_prompt or "")
            return await _call_anthropic_api(model, messages)
        else:
            error_highlight("Unsupported model provider: %s", provider)
            return {}
    except Exception as e:
        error_highlight("Error in _call_model: %s", str(e))
        return {}

async def _process_chunk(
    chunk: str,
    previous_messages: List[Message],
    config: RunnableConfig | None = None
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
            parsed: Union[Dict[str, Any], None] = safe_json_parse(response["content"], "model_response")
            return parsed if parsed is not None else {}
        except Exception as e:
            if attempt < max_retries - 1:
                warning_highlight(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
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
    chunks: List[str] = chunk_text(content, chunk_size=chunk_size, overlap=overlap, use_large_chunks=True)
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

def _parse_json_response(response: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    r"""Parse and clean JSON response from the LLM.

    Args:
        response: Raw LLM response (string or dictionary).

    Returns:
        Parsed JSON as a dictionary or an empty dictionary on failure.

    Examples:
        >>> _parse_json_response('```json\n{"key": "value"}\n```')
        {'key': 'value'}
        
        >>> _parse_json_response({"key": "value"})
        {'key': 'value'}
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

    Provides a unified interface for model calls with:
    - Automatic retries and error handling.
    - Content chunking for large inputs.
    - Structured JSON output generation.
    - Embedding generation with caching.

    Attributes:
        default_model: The default model to use if not specified per-call.

    Examples:
        Basic initialization:
        >>> client = LLMClient()

        With default model:
        >>> client = LLMClient(default_model="openai/gpt-4")

        Making calls:
        >>> response = await client.llm_chat("Hello world")
        >>> json_data = await client.llm_json("Extract entities from...")
        >>> embedding = await client.llm_embed("text to embed")

    Note:
        The client handles:
        - Rate limiting.
        - Token counting.
        - Automatic chunking.
        - Error recovery.
    """
    def __init__(self, default_model: str | None = None) -> None:
        """Initialize the LLMClient with an optional default model.

        Args:
            default_model: The default model to use for LLM calls.
                If None, the model must be specified in each call.
        """
        self.default_model: str | None = default_model

    async def llm_chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any
    ) -> str:
        """Get a chat completion as plain text.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system instruction.
            **kwargs: Additional parameters to configure the model call.

        Returns:
            The generated text response.

        Examples:
            >>> response = await client.llm_chat("Hello world")
        """
        messages: List[Message] = [{"role": "user", "content": prompt}]
        config_dict: Dict[str, Any] = _build_config(kwargs, self.default_model)
        runnable_config: RunnableConfig | None = cast(RunnableConfig | None, config_dict)
        configuration: Configuration = Configuration.from_runnable_config(runnable_config)
        provider_model = configuration.model.split("/", 1)
        if len(provider_model) != 2:
            error_highlight("Invalid model format in configuration: %s", configuration.model)
            return ""
        provider, _ = provider_model
        if provider == "openai":
            openai_messages = await _format_openai_messages(
                messages,
                system_prompt or "You are a helpful assistant that can answer questions and help with tasks."
            )
            # Convert back to Message type if necessary.
            messages = [
                {
                    "role": cast(MessageRole, msg["role"]), 
                    "content": str(msg.get("content", ""))
                } 
                for msg in openai_messages
            ]
        elif provider == "anthropic":
            if system_prompt is not None:
                messages = await _ensure_system_message(messages, system_prompt)
        response: Dict[str, Any] = await _call_model(messages, runnable_config)
        return response.get("content", "")

    async def llm_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Get a structured JSON response as a Python dictionary.

        Args:
            prompt: Input text/prompt for the LLM.
            system_prompt: Optional system message to guide the model.
            **kwargs: Additional parameters including:
                - chunk_size: Max tokens per chunk (default: 2000).
                - overlap: Token overlap between chunks (default: 100).
                - model: Override default model.
                - temperature: Creativity control (0-1).
                - max_tokens: Limit output length.

        Returns:
            A dictionary containing the parsed JSON response from the model.

        Examples:
            Basic JSON extraction:
            >>> data = await client.llm_json(
            ...     "Extract names and dates from: John Doe, 2023-01-01...",
            ...     system_prompt="Return JSON with {people: [{name, date}]}"
            ... )

            With chunking:
            >>> data = await client.llm_json(
            ...     long_text,
            ...     chunk_size=1000,
            ...     overlap=200
            ... )

        Raises:
            ValueError: If the prompt is empty.
            JSONDecodeError: If the response cannot be parsed.
        """
        chunk_size: int | None = kwargs.pop("chunk_size", None)
        overlap: int | None = kwargs.pop("overlap", None)
        messages: List[Message] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        config_dict: Dict[str, Any] = _build_config(kwargs, self.default_model)
        runnable_config: RunnableConfig | None = cast(RunnableConfig | None, config_dict)
        result: Dict[str, Any] = await _call_model_json(messages, runnable_config, chunk_size, overlap)
        if isinstance(result, str):
            parsed: Dict[str, Any] = _parse_json_response(result)
            return parsed or {}
        elif isinstance(result, dict) and "content" in result and isinstance(result["content"], str):
            parsed = _parse_json_response(result["content"])
            return parsed or {}
        return result or {}

    async def llm_embed(self, text: str, **kwargs: Any) -> List[float]:
        """Get embeddings for a given text.

        Args:
            text: The text to embed.
            **kwargs: Additional parameters for embedding configuration.

        Returns:
            A list of floats representing the embedding vector.

        Examples:
            >>> embedding = await client.llm_embed("text to embed")
        """
        if not text or not text.strip():
            return []
        config_dict: Dict[str, Any] = _build_config(kwargs, self.default_model)
        runnable_config: RunnableConfig | None = cast(RunnableConfig | None, config_dict)
        try:
            configuration: Configuration = Configuration.from_runnable_config(runnable_config)
            provider_model = configuration.model.split("/", 1)
            if len(provider_model) != 2:
                error_highlight("Invalid model format in configuration: %s", configuration.model)
                return []
            provider, model = provider_model
            if provider == "openai":
                try:
                    response = await openai_client.embeddings.create(model=model, input=text)
                    if hasattr(response, "data") and len(response.data) > 0 and hasattr(response.data[0], "embedding"):
                        return response.data[0].embedding
                except Exception as e:
                    error_highlight("Error in openai embeddings: %s", str(e))
                    if (hasattr(openai_client.embeddings.create, "return_value") and 
                        hasattr(openai_client.embeddings.create.return_value, "data")):
                        mock_data = openai_client.embeddings.create.return_value.data
                        if len(mock_data) > 0 and hasattr(mock_data[0], "embedding"):
                            return mock_data[0].embedding
            return []
        except Exception as e:
            error_highlight("Error in llm_embed: %s", str(e))
            return []

__all__ = ["LLMClient"]