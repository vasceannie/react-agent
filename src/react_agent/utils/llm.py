"""LLM utility functions for handling model calls and content processing.

This module provides utilities for interacting with language models,
including content length management and error handling.
"""

from typing import List, Dict, Any, Optional, Union, cast
import json
import logging
from datetime import datetime, timezone
import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, ensure_config
from openai import AsyncClient
from openai.types.chat import ChatCompletionMessageParam
from anthropic import AsyncAnthropic

from react_agent.utils.logging import get_logger, info_highlight, warning_highlight, error_highlight
from react_agent.utils.content import (
    chunk_text,
    preprocess_content,
    estimate_tokens,
    validate_content,
    detect_content_type,
    merge_chunk_results
)
from react_agent.configuration import Configuration

# Initialize logger
logger = get_logger(__name__)

# Constants
MAX_TOKENS: int = 16000  # Reduced from 100000 to stay within model limits
DEFAULT_CHUNK_SIZE: int = 4000  # Reduced chunk size
DEFAULT_OVERLAP: int = 500  # Reduced overlap
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

async def _format_openai_messages(messages: List[Dict[str, str]], system_prompt: str) -> List[ChatCompletionMessageParam]:
    """Format messages for OpenAI API."""
    formatted_messages: List[ChatCompletionMessageParam] = []
    for msg in messages:
        if msg["role"] == "system":
            formatted_messages.append({"role": "system", "content": msg["content"]})
        else:
            content = msg["content"]
            if estimate_tokens(content) > MAX_TOKENS:
                info_highlight("Content too long, summarizing...")
                content = await summarize_content(content)
            formatted_messages.append({"role": "user", "content": content})

    if all(msg["role"] != "system" for msg in formatted_messages):
        formatted_messages.insert(0, {"role": "system", "content": system_prompt})
    
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
            formatted_messages = [
                {"role": "user", "content": configuration.system_prompt} if messages[0]["role"] != "system" else None,
                *[{"role": "user" if msg["role"] == "system" else msg["role"], "content": msg["content"]} for msg in messages]
            ]
            formatted_messages = [msg for msg in formatted_messages if msg is not None]
            
            response = await anthropic_client.messages.create(
                model=model,
                messages=formatted_messages,
                max_tokens=MAX_TOKENS,
                temperature=0.7
            )
            return str(response.content[0])
        else:
            error_highlight(f"Unsupported model provider: {provider}")
            return None

    except Exception as e:
        error_highlight(f"Error in call_model: {str(e)}")
        return None

async def call_model_json(
    messages: List[Dict[str, str]],
    config: Optional[RunnableConfig] = None,
    max_tokens: int = MAX_TOKENS
) -> Dict[str, Any]:
    """Call the LLM and get a JSON response.
    
    Args:
        messages: List of message dictionaries with role and content
        config: Optional configuration dictionary
        max_tokens: Maximum number of tokens to process
        
    Returns:
        JSON response from the model
    """
    if not messages:
        error_highlight("No messages provided to call_model_json")
        return {}

    try:
        # Get and validate content from last message
        content = messages[-1].get("content", "")
        if not content or not validate_content(content):
            error_highlight("Invalid or empty content in last message")
            return {}

        # Preprocess and estimate tokens
        content = preprocess_content(content, "")
        estimated_tokens = estimate_tokens(content)
        
        # Handle large content with chunking
        if estimated_tokens > max_tokens:
            info_highlight(f"Content exceeds token limit ({estimated_tokens} > {max_tokens}), chunking")
            return await _process_chunked_content(messages, content, config)
            
        # Process single content
        response = await call_model(messages, config)
        if not response:
            error_highlight("Empty response from model")
            return {}
            
        return _parse_json_response(response)
                    
    except Exception as e:
        error_highlight(f"Error in call_model_json: {str(e)}")
        return {}

async def _process_chunked_content(
    messages: List[Dict[str, str]],
    content: str,
    config: Optional[RunnableConfig]
) -> Dict[str, Any]:
    """Process content that exceeds token limit by chunking."""
    chunks = chunk_text(content, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP)
    chunk_results = []
    
    for i, chunk in enumerate(chunks):
        chunk_messages = messages[:-1] + [{"role": "human", "content": chunk}]
        try:
            chunk_response = await call_model(chunk_messages, config)
            if chunk_response:
                chunk_results.append(chunk_response)
        except Exception as e:
            error_highlight(f"Error processing chunk {i+1}: {str(e)}")
            continue
    
    if not chunk_results:
        error_highlight("No valid results from chunks")
        return {}
        
    return merge_chunk_results(chunk_results, "general")

def _parse_json_response(response: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Parse and clean JSON response from model."""
    if isinstance(response, dict):
        return response
        
    if not isinstance(response, str):
        error_highlight(f"Unexpected response type: {type(response)}")
        return {}
        
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return _clean_and_parse_json(response)

def _clean_and_parse_json(response_str: str) -> Dict[str, Any]:
    """Clean and parse potentially malformed JSON response."""
    if "{" not in response_str or "}" not in response_str:
        return {}
        
    try:
        start_idx = response_str.find("{")
        end_idx = response_str.rfind("}") + 1
        cleaned = response_str[start_idx:end_idx]
        cleaned = cleaned.replace("'", '"').replace(",\n}", "\n}").replace(",\n]", "\n]")
        return json.loads(cleaned)
    except Exception as e:
        error_highlight(f"Failed to clean and parse JSON: {str(e)}")
        return {}

__all__ = ["call_model_json", "call_model"]