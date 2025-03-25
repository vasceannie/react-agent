"""Language Model (LLM) client utilities and abstractions.

This module provides a unified interface for interacting with various LLM providers,
handling retries, rate limiting, and response processing consistently.

Examples:
    Basic completion:
    >>> client = LLMClient(provider="openai")
    >>> response = client.complete(
    ...     prompt="Translate to French: Hello world",
    ...     temperature=0.7
    ... )
    >>> print(response)
    "Bonjour le monde"

    Streaming response:
    >>> stream = client.complete_stream(
    ...     prompt="Write a short poem about AI",
    ...     max_tokens=100
    ... )
    >>> for chunk in stream:
    ...     print(chunk, end="")

    With custom parameters:
    >>> response = client.complete(
    ...     prompt="Explain quantum computing",
    ...     model="gpt-4-turbo",
    ...     temperature=0.3,
    ...     presence_penalty=0.5,
    ...     max_tokens=500
    ... )
"""

from typing import Optional, AsyncGenerator, Dict, Any, List
from dataclasses import dataclass

@dataclass
class LLMResponse:
    """Container for LLM response data with metadata.

    Example:
        >>> response = LLMResponse(
        ...     content="The capital of France is Paris.",
        ...     model="gpt-3.5-turbo",
        ...     usage={
        ...         'prompt_tokens': 15,
        ...         'completion_tokens': 8,
        ...         'total_tokens': 23
        ...     },
        ...     finish_reason="stop"
        ... )
        >>> print(response.content)
        The capital of France is Paris.
    """
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    raw_response: Optional[Dict[str, Any]] = None


class LLMClient:
    """Unified client for interacting with multiple LLM providers.

    Supports both synchronous and asynchronous operations with:
    - Automatic retries
    - Rate limiting
    - Response caching
    - Streaming

    Example initialization:
        >>> # With default settings
        >>> client = LLMClient()
        
        >>> # With custom configuration
        >>> client = LLMClient(
        ...     provider="anthropic",
        ...     api_key="sk-...",
        ...     default_model="claude-3-opus",
        ...     timeout=30
        ... )
    """

    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a completion from the LLM.

        Args:
            prompt: Input text to send to the model
            model: Override default model
            **kwargs: Additional model parameters (temperature, max_tokens etc.)

        Returns:
            LLMResponse object containing generated content and metadata

        Examples:
            Basic completion:
            >>> response = await client.complete(
            ...     prompt="Write a haiku about winter",
            ...     temperature=0.7,
            ...     max_tokens=50
            ... )

            With model override:
            >>> response = await client.complete(
            ...     prompt="Explain blockchain",
            ...     model="gpt-4",
            ...     max_tokens=500
            ... )
        """
        pass

    async def complete_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream completion tokens from the LLM.

        Args:
            prompt: Input text to send to the model
            model: Override default model
            **kwargs: Additional model parameters

        Yields:
            Response chunks as they're generated

        Examples:
            >>> async for chunk in client.complete_stream(
            ...     prompt="Tell me a story",
            ...     temperature=0.8
            ... ):
            ...     print(chunk, end="")
        """
        pass

    async def embed(
        self,
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        """Generate embeddings for the input text.

        Args:
            text: Input to embed
            model: Override default embedding model

        Returns:
            List of floats representing the embedding

        Examples:
            >>> embedding = await client.embed(
            ...     text="machine learning"
            ... )
            >>> len(embedding)
            1536  # Dimension of embedding vector
        """
        pass
