"""LLM utility functions for handling model calls and content processing.

This module provides utilities for interacting with language models,
including content length management and error handling.

Examples:
    Basic usage of call_model:
    
    ```python
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    response = await call_model(messages)
    # Returns: {"content": "The capital of France is Paris."}
    ```

    Using call_model_json with chunking:
    
    ```python
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Analyze this long text: " + large_document}
    ]
    config = {"configurable": {"model": "openai/gpt-4"}}
    response = await call_model_json(
        messages, 
        config=config,
        chunk_size=1000,  # Process in 1000-token chunks
        overlap=100       # 100-token overlap between chunks
    )
    # Returns: {
    #     "analysis": "Key points from the document...",
    #     "summary": "Overall summary...",
    #     "topics": ["topic1", "topic2", ...]
    # }
    ```
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
import os
import asyncio
from typing import Any, Dict, List, Optional, Union, cast

from anthropic import AsyncAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, ensure_config
from openai import AsyncClient
from openai.types.chat import ChatCompletionMessageParam

from react_agent.configuration import Configuration
from react_agent.utils.content import (
    chunk_text,
    detect_content_type,
    estimate_tokens,
    merge_chunk_results,
    preprocess_content,
    validate_content,
)
from react_agent.configuration import Configuration
from react_agent.utils.extraction import safe_json_parse
from react_agent.utils.defaults import ChunkConfig
from react_agent.utils.logging import (
    get_logger,
    error_highlight,
    info_highlight,
    warning_highlight
)

# Initialize logger
logger = get_logger(__name__)

# Constants
MAX_TOKENS: int = 16000  # Reduced from 100000 to stay within model limits
MAX_SUMMARY_TOKENS: int = 2000  # New constant for summary model

# Initialize API clients
openai_client = AsyncClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)
anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

async def summarize_content(input_content: str, max_tokens: int = MAX_SUMMARY_TOKENS) -> str:
    """Summarize content using a more efficient model.
    
    Args:
        input_content: The content to summarize
        max_tokens: Maximum tokens for the summary
        
    Returns:
        Summarized content
    """
    try:
        # Use a more efficient model for summarization
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Use a more efficient model for summarization
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates concise summaries. Focus on key points and maintain factual accuracy."},
                {"role": "user", "content": f"Please summarize the following content concisely:\n\n{input_content}"}
            ],
            max_tokens=max_tokens,
            temperature=0.3  # Lower temperature for more focused summaries
        )
        response_content = cast(Optional[str], response.choices[0].message.content)
        return response_content if response_content is not None else ""
    except Exception as e:
        error_highlight(f"Error in summarize_content: {str(e)}")
        return input_content  # Return original content if summarization fails

async def _format_openai_messages(
    messages: List[Dict[str, str]], 
    system_prompt: str,
    max_tokens: Optional[int] = None
) -> List[ChatCompletionMessageParam]:
    """Format messages for OpenAI API with enhanced content handling.
    
    Args:
        messages: List of message dictionaries
        system_prompt: System prompt to use
        max_tokens: Optional maximum tokens per message
        
    Returns:
        Formatted messages for OpenAI API
    """
    if not messages:
        return [{"role": "system", "content": system_prompt}]
        
    formatted_messages: List[ChatCompletionMessageParam] = []
    max_tokens = max_tokens or MAX_TOKENS
    
    for msg in messages:
        if msg["role"] == "system":
            formatted_messages.append({"role": "system", "content": msg["content"]})
        else:
            content = msg["content"]
            if estimate_tokens(content) > max_tokens:
                info_highlight("Content too long, summarizing...")
                content = await summarize_content(content, max_tokens)
            formatted_messages.append({"role": "user", "content": content})

    if all(msg["role"] != "system" for msg in formatted_messages):
        formatted_messages.insert(0, {"role": "system", "content": system_prompt})
    
    return formatted_messages

async def _call_openai_api(model: str, messages: List[ChatCompletionMessageParam]) -> Dict[str, Any]:
    """Call OpenAI API with formatted messages."""
    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.7
        )
        content = response.choices[0].message.content
        return {"content": content} if content else {}
    except Exception as e:
        error_highlight(f"Error in _call_openai_api: {str(e)}")
        return {}

async def call_model(
    messages: List[Dict[str, str]],
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Call the language model with the given messages.
    
    Args:
        messages: List of message dictionaries. Each message should have 'role' 
                 (system/user/assistant) and 'content' keys.
        config: Optional configuration for the model call. Can include model selection,
               temperature, and other parameters.

    Returns:
        Dict containing the model's response. The response will always have a 'content'
        key with the model's output.

    Examples:
        ```python
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        response = await call_model(messages)
        # Returns: {"content": "2 + 2 = 4"}
        
        # With custom configuration
        config = {
            "configurable": {
                "model": "anthropic/claude-3",
                "temperature": 0.7
            }
        }
        response = await call_model(messages, config)
        ```
    """
    if not messages:
        error_highlight("No messages provided to call_model")
        return {}

    try:
        config = config or {}
        configurable = config.get("configurable", {})
        configurable["timestamp"] = datetime.now(timezone.utc).isoformat()
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
                formatted_messages.append({"role": "system", "content": configuration.system_prompt})

            # Process all messages, preserving their roles correctly
            formatted_messages.extend(
                (
                    {"role": msg["role"], "content": msg["content"]}
                    if msg["role"] == "system"
                    else {
                        "role": (
                            "user" if msg["role"] == "user" else "assistant"
                        ),
                        "content": msg["content"],
                    }
                )
                for msg in messages
                if msg["role"] in ["user", "assistant", "system"]
            )
            formatted_messages = [msg for msg in formatted_messages if msg is not None]

            response = await anthropic_client.messages.create(
                model=model,
                messages=[{"role": msg["role"], "content": msg["content"]} for msg in formatted_messages],
                max_tokens=MAX_TOKENS,
                temperature=0.7
            )
            return {"content": str(response.content[0])}
        else:
            error_highlight(f"Unsupported model provider: {provider}")
            return {}

    except Exception as e:
        error_highlight(f"Error in call_model: {str(e)}")
        return {}

async def _process_chunk(
    chunk: str,
    previous_messages: List[Dict[str, str]],
    config: Optional[RunnableConfig] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """Process a single chunk of content with enhanced error handling.
    
    Args:
        chunk: Content chunk to process
        previous_messages: Previous messages in the conversation
        config: Optional configuration
        model: Optional model to use for processing
        
    Returns:
        Processed chunk result
    """
    if not chunk or not previous_messages:
        return {}

    try:
        # Create messages with chunk
        messages = previous_messages + [{"role": "human", "content": chunk}]
        
        # Call model with retries
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = await call_model(messages, config)
                if not response or not response.get("content"):
                    error_highlight("Empty response from model")
                    return {}
                    
                # Parse JSON response with enhanced error handling
                parsed = safe_json_parse(response["content"], "model_response")
                return parsed or {}
                
            except Exception as e:
                if attempt < max_retries - 1:
                    warning_highlight(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    error_highlight(f"All retry attempts failed: {str(e)}")
                    return {}
                    
    except Exception as e:
        error_highlight(f"Error processing chunk: {str(e)}")
        return {}
        
    return {}  # Fallback return to satisfy type checker

async def call_model_json(
    messages: List[Dict[str, str]],
    config: Optional[RunnableConfig] = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None
) -> Dict[str, Any]:
    """Call the model with JSON output format and enhanced chunking.
    
    This function handles large inputs by automatically chunking them and
    merging the results. It expects and returns JSON-formatted data.
    
    Args:
        messages: List of message dictionaries. Each message should have 'role'
                 and 'content' keys.
        config: Optional configuration for the model call.
        chunk_size: Optional custom chunk size in tokens. If not provided,
                   uses default chunking configuration.
        overlap: Optional overlap size between chunks in tokens.

    Returns:
        Dict containing the merged JSON response from all chunks.

    Examples:
        ```python
        # Simple query expecting JSON response
        messages = [
            {"role": "user", "content": "List 3 capitals in JSON format"}
        ]
        response = await call_model_json(messages)
        # Returns: {
        #     "capitals": [
        #         {"city": "Paris", "country": "France"},
        #         {"city": "Tokyo", "country": "Japan"},
        #         {"city": "Rome", "country": "Italy"}
        #     ]
        # }

        # Processing a large document with custom chunking
        messages = [
            {"role": "user", "content": "Analyze this document: " + large_text}
        ]
        response = await call_model_json(
            messages,
            chunk_size=2000,  # Process in 2000-token chunks
            overlap=200       # 200-token overlap between chunks
        )
        # Returns merged analysis from all chunks:
        # {
        #     "main_topics": ["topic1", "topic2", ...],
        #     "key_points": ["point1", "point2", ...],
        #     "summary": "Overall summary of the document..."
        # }
        ```
    """
    try:
        if not messages:
            error_highlight("No messages provided to call_model_json")
            return {}

        # Process content in chunks if needed
        content = messages[-1]["content"]

        if estimate_tokens(content) <= MAX_TOKENS:
            return await _process_chunk(content, messages[:-1], config)

        info_highlight(f"Content too large ({estimate_tokens(content)} tokens), chunking...")
        chunks = chunk_text(
            content,
            chunk_size=chunk_size,
            overlap=overlap,
            use_large_chunks=True
        )

        if len(chunks) <= 1:
            return await _process_chunk(content, messages[:-1], config)
        # Process chunks in parallel with rate limiting
        chunk_results = []
        for chunk in chunks:
            result = await _process_chunk(chunk, messages[:-1], config)
            if result:
                chunk_results.append(result)

        if not chunk_results:
            error_highlight("No valid results from chunks")
            return {}

        # Merge results with enhanced merging
        return merge_chunk_results(chunk_results, "model_response")
    except Exception as e:
        error_highlight(f"Error in call_model_json: {str(e)}")
        return {}

async def _process_chunked_content(
    messages: List[Dict[str, str]],
    content: str,
    config: Optional[RunnableConfig]
) -> Dict[str, Any]:
    """Process content that exceeds token limit by chunking."""
    try:
        chunks = chunk_text(
            content,
            chunk_size=ChunkConfig.DEFAULT_CHUNK_SIZE,
            overlap=ChunkConfig.DEFAULT_OVERLAP
        )
        chunk_results = []
        
        for i, chunk in enumerate(chunks):
            chunk_messages = messages[:-1] + [{"role": "human", "content": chunk}]
            try:
                chunk_response = await call_model(chunk_messages, config)
                if chunk_response and chunk_response.get("content"):
                    chunk_results.append(chunk_response)
            except Exception as e:
                error_highlight(f"Error processing chunk {i+1}: {str(e)}")
                continue
        
        if not chunk_results:
            error_highlight("No valid results from chunks")
            return {}
            
        return merge_chunk_results(chunk_results, "general")
    except Exception as e:
        error_highlight(f"Error in _process_chunked_content: {str(e)}")
        return {}

def _parse_json_response(response: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Parse and clean JSON response from model."""
    try:
        if isinstance(response, dict):
            return response
        return safe_json_parse(response, "model_response")
    except Exception as e:
        error_highlight(f"Error in _parse_json_response: {str(e)}")
        return {}

__all__ = ["call_model_json", "call_model"]