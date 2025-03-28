"""Jina AI Tools - Clean Implementation for LangGraph.

This module provides production-ready tool implementations for Jina AI's
Search Foundation APIs, designed to work seamlessly with LangGraph's
orchestration framework. The implementation follows best practices for
asynchronous programming, error handling, and type safety.

Each tool is structured to support both standalone usage and integration
with LangGraph's ToolNode, InjectedState, and checkpoint mechanisms.

Usage:
    1. Set the JINA_API_KEY environment variable
    2. Import specific tools or use the create_jina_toolnode() function
    3. Integrate with your LangGraph workflow

Example:
    ```python
    from langgraph.graph import StateGraph
    from react_agent.tools.jina import create_jina_toolnode
    from react_agent.tools.jina import search, reader, embeddings

    # Create a graph with all Jina tools
    workflow = StateGraph(YourStateType)
    workflow.add_node("jina_tools", create_jina_toolnode())
    
    # Or with selected tools
    search_tools = create_jina_toolnode(include_tools=["search", "grounding"])
    workflow.add_node("search_tools", search_tools)
    ```
"""

import asyncio
import os
from typing import Annotated, Any, Dict, List, Literal, TypedDict, Union

import aiohttp
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, ToolException, tool
from langgraph.prebuilt import InjectedState, InjectedStore, ToolNode
from langgraph.store.base import BaseStore
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from react_agent.configuration import Configuration
from react_agent.utils.cache import create_checkpoint, load_checkpoint

# Utilities from your existing project
from react_agent.utils.logging import (
    error_highlight,
    get_logger,
    info_highlight,
    warning_highlight,
)
from react_agent.utils.validations import is_valid_url

logger = get_logger(__name__)

# -------------------------------------------------------------------------
# Common Types and Configuration
# -------------------------------------------------------------------------


class RetryConfig(BaseModel):
    """Configuration for HTTP request retries with exponential backoff."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_retries: PositiveInt = Field(default=3, description="Maximum number of retries")
    base_delay: PositiveFloat = Field(default=1.0, description="Base delay in seconds")
    max_delay: PositiveFloat = Field(default=5.0, description="Maximum delay in seconds")

    @model_validator(mode="after")
    def validate_delays(self) -> "RetryConfig":
        """Ensure max_delay is greater than or equal to base_delay."""
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be greater than or equal to base_delay")
        return self


class JinaToolState(TypedDict, total=False):
    """State for LangGraph tools to access and modify."""
    messages: List[Any]  # Messages in the graph state
    jina_api_key: str | None  # Optional API key override
    retry_config: RetryConfig | None  # Optional retry configuration
    cache_results: bool  # Whether to cache results in store


# -------------------------------------------------------------------------
# Shared HTTP Client Logic 
# -------------------------------------------------------------------------

async def _make_request_with_retry(
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"],
    url: str,
    headers: Dict[str, str],
    json_data: Dict[str, Any] | None = None,
    retry_config: RetryConfig | None = None,
) -> Dict[str, Any]:
    """Make an HTTP request with exponential backoff retry logic.
    
    Args:
        method: HTTP method to use
        url: Target URL for the request
        headers: HTTP headers to include
        json_data: Optional JSON payload
        retry_config: Configuration for retries
        
    Returns:
        API response as a dictionary or error information
    """
    if retry_config is None:
        retry_config = RetryConfig()

    attempt = 0
    last_exception: Exception | None = None

    while attempt < retry_config.max_retries:
        attempt += 1
        try:
            info_highlight(f"Attempt {attempt} {method} {url}", "JinaTool")

            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json_data,
                    raise_for_status=False,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    status = resp.status
                    text = await resp.text()

                    # Special case for 422 (invalid input)
                    if status == 422:
                        warning_highlight(f"422 Unprocessable Entity: {text}", "JinaTool")
                        return {"error": {"status": status, "message": text}}

                    # Handle successful responses (status 2xx)
                    if 200 <= status < 300:
                        try:
                            return await resp.json()
                        except Exception:
                            # If not valid JSON, return the raw text
                            return {"raw_response": text, "status": status}

                    # Handle error responses
                    error_msg = f"HTTP {status}: {text}"
                    raise aiohttp.ClientError(error_msg)

        except aiohttp.ClientError as ce:
            last_exception = ce
            warning_highlight(
                f"ClientError on attempt {attempt}/{retry_config.max_retries}: {str(last_exception)}",
                "JinaTool",
            )
        except Exception as e:
            last_exception = e
            warning_highlight(
                f"Exception on attempt {attempt}/{retry_config.max_retries}: {str(last_exception)}",
                "JinaTool",
            )
        # Calculate exponential backoff delay
        delay = min(retry_config.max_delay, retry_config.base_delay * (2 ** (attempt - 1)))
        await asyncio.sleep(delay)

    # All retry attempts failed
    error_highlight(f"All retry attempts failed: {last_exception}", "JinaTool")
    return {"error": {"message": str(last_exception), "status": -1}}


def _get_jina_api_key(state: Dict[str, Any] | None = None, config: RunnableConfig | None = None) -> str:
    """Get the Jina API key from configuration, state, or environment.
    
    Args:
        state: Optional state dictionary that might contain the API key
        config: Optional RunnableConfig that might contain the API key
        
    Returns:
        The API key as a string
        
    Raises:
        ValueError: If the API key is not found
    """
    # First check if config exists and has the API key
    if config is not None:
        configuration = Configuration.from_runnable_config(config)
        if configuration.jina_api_key:
            return configuration.jina_api_key

    # Then check if key exists in state
    if state and state.get("jina_api_key"):
        return state["jina_api_key"]

    if key := os.environ.get("JINA_API_KEY"):
        return key
    else:
        raise ValueError(
            "JINA_API_KEY not found in configuration, state, or environment. "
            "Set the JINA_API_KEY environment variable or include it in the configuration."
        )


# -------------------------------------------------------------------------
# 1. Embeddings API
# -------------------------------------------------------------------------

class EmbeddingsRequest(BaseModel):
    """Request model for Jina's Embeddings API."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(
        ..., 
        description="Model identifier (e.g. 'jina-embeddings-v3', 'jina-clip-v2')"
    )
    input: List[str] = Field(
        ...,
        min_length=1,
        description="Text strings or base64-encoded images to embed"
    )
    embedding_type: Union[Literal["float", "base64"], List[Literal["float", "base64"]]] | None = Field(
        default="float",
        description="Format of returned embeddings"
    )
    task: str | None = Field(
        default=None,
        description="Intended downstream application (e.g. 'retrieval.query')"
    )
    dimensions: PositiveInt | None = Field(
        default=None,
        description="Truncates output embeddings to this size if set"
    )
    normalized: bool = Field(
        default=False,
        description="Normalize embeddings to unit L2 norm"
    )
    late_chunking: bool = Field(
        default=False,
        description="Concatenate sentences and treat as a single input"
    )


@tool
async def embeddings(
    model: str,
    input_texts: List[str],
    embedding_type: str = "float",
    task: str | None = None,
    dimensions: int | None = None,
    normalized: bool = False,
    late_chunking: bool = False,
    state: Annotated[JinaToolState, InjectedState] = None,
    store: Annotated[BaseStore, InjectedStore] = None,
    config: Annotated[RunnableConfig, InjectedToolArg] = None
) -> Dict[str, Any]:
    """Generate vector embeddings for text or images using Jina AI.
    
    Args:
        model: Identifier of model to use (e.g. 'jina-embeddings-v3', 'jina-clip-v2')
        input_texts: List of texts or base64-encoded images to embed
        embedding_type: Format of returned embeddings (default: 'float')
        task: Intended downstream application (e.g. 'retrieval.query')
        dimensions: Truncate output embeddings to this size if set
        normalized: Normalize embeddings to unit L2 norm
        late_chunking: Concatenate sentences as a single input
        state: Injected graph state with API key or retry config
        
    Returns:
        Dictionary containing embedding data for each input
    """
    # Create request object
    request = EmbeddingsRequest(
        model=model,
        input=input_texts,
        embedding_type=embedding_type,
        task=task,
        dimensions=dimensions,
        normalized=normalized,
        late_chunking=late_chunking
    )

    # Create cache key for store
    cache_key = f"jina_embeddings_{model}_{hash(str(input_texts))}"

    # Check cache if store is available and caching is enabled
    if store and state and state.get("cache_results", True):
        try:
            cached_data = store.get(["jina", cache_key])
            if cached_data and cached_data.value:
                info_highlight(f"Using cached embeddings result for {model}", "JinaTool")
                return cached_data.value
        except Exception as e:
            warning_highlight(f"Error accessing cache: {str(e)}", "JinaTool")

    # Get API key and retry configuration
    api_key = _get_jina_api_key(state, config)
    retry_config = state.get("retry_config") if state else None

    # Set up headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    # Make the API request
    response = await _make_request_with_retry(
        method="POST",
        url="https://api.jina.ai/v1/embeddings",
        headers=headers,
        json_data=request.model_dump(exclude_none=True),
        retry_config=retry_config,
    )

    # Check for errors
    if "error" in response:
        error_highlight(f"Embeddings API error: {response['error']}", "JinaTool")
        raise ToolException(f"Jina Embeddings API error: {response['error'].get('message', 'Unknown error')}")

    # Cache result if store is available and caching is enabled
    if store and state and state.get("cache_results", True):
        try:
            store.put(["jina", cache_key], response)
            info_highlight(f"Cached embeddings result for {model}", "JinaTool")
        except Exception as e:
            warning_highlight(f"Error caching embeddings result: {str(e)}", "JinaTool")

    return response


# -------------------------------------------------------------------------
# 2. Re-Ranker API
# -------------------------------------------------------------------------

class RerankerRequest(BaseModel):
    """Request model for Jina's Re-Ranker API."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(
        ...,
        description="Model identifier (e.g. 'jina-reranker-v2-base-multilingual')"
    )
    query: str = Field(
        ...,
        min_length=1,
        description="The search query for re-ranking"
    )
    documents: List[Union[str, Dict[str, Any]]] = Field(
        ...,
        min_length=1,
        description="List of documents to re-rank"
    )
    top_n: PositiveInt | None = Field(
        default=None,
        description="Number of top documents to return"
    )
    return_documents: bool = Field(
        default=True,
        description="Return documents in the response"
    )


@tool
async def rerank(
    model: str,
    query: str,
    documents: List[Union[str, Dict[str, Any]]],
    top_n: int | None = None,
    return_documents: bool = True,
    state: Annotated[JinaToolState, InjectedState] = None,
    store: Annotated[BaseStore, InjectedStore] = None,
    config: Annotated[RunnableConfig, InjectedToolArg] = None
) -> Dict[str, Any]:
    """Re-rank a list of documents based on relevance to a query.
    
    Args:
        model: Identifier of model to use (e.g. 'jina-reranker-v2-base-multilingual')
        query: The search query for re-ranking
        documents: List of documents (strings or dicts) to re-rank
        top_n: Number of top documents to return
        return_documents: Return documents in the response
        state: Injected graph state with API key or retry config
        
    Returns:
        Dictionary containing re-ranked documents with relevance scores
    """
    # Create request object
    request = RerankerRequest(
        model=model,
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=return_documents
    )
    
    # Create cache key for store
    doc_hash = str(hash(str(documents)))
    cache_key = f"jina_rerank_{model}_{query}_{doc_hash}_{top_n}_{return_documents}"
    
    # Check cache if store is available and caching is enabled
    if store and state and state.get("cache_results", True):
        try:
            cached_data = store.get(["jina", cache_key])
            if cached_data and cached_data.value:
                info_highlight(f"Using cached reranking result for {query}", "JinaTool")
                return cached_data.value
        except Exception as e:
            warning_highlight(f"Error accessing cache: {str(e)}", "JinaTool")
    
    # Get API key and retry configuration
    api_key = _get_jina_api_key(state, config)
    retry_config = state.get("retry_config") if state else None
    
    # Set up headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    
    # Make the API request
    response = await _make_request_with_retry(
        method="POST",
        url="https://api.jina.ai/v1/rerank",
        headers=headers,
        json_data=request.model_dump(exclude_none=True),
        retry_config=retry_config,
    )
    
    # Check for errors
    if "error" in response:
        error_highlight(f"Reranker API error: {response['error']}", "JinaTool")
        raise ToolException(f"Jina Reranker API error: {response['error'].get('message', 'Unknown error')}")
    
    # Cache result if store is available and caching is enabled
    if store and state and state.get("cache_results", True):
        try:
            store.put(["jina", cache_key], response)
            info_highlight(f"Cached reranking result for {query}", "JinaTool")
        except Exception as e:
            warning_highlight(f"Error caching reranking result: {str(e)}", "JinaTool")
    
    return response


# -------------------------------------------------------------------------
# 3. Reader API
# -------------------------------------------------------------------------

class ReaderHeaders(BaseModel):
    """Optional headers for the Reader API."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    x_engine: str | None = Field(default=None, alias="X-Engine")
    x_timeout: PositiveInt | None = Field(default=None, alias="X-Timeout")
    x_target_selector: str | None = Field(default=None, alias="X-Target-Selector")
    x_wait_for_selector: str | None = Field(default=None, alias="X-Wait-For-Selector")
    x_remove_selector: str | None = Field(default=None, alias="X-Remove-Selector")
    x_with_links_summary: bool | None = Field(default=None, alias="X-With-Links-Summary")
    x_with_images_summary: bool | None = Field(default=None, alias="X-With-Images-Summary")
    x_with_generated_alt: bool | None = Field(default=None, alias="X-With-Generated-Alt")
    x_no_cache: bool | None = Field(default=None, alias="X-No-Cache")
    x_with_iframe: bool | None = Field(default=None, alias="X-With-Iframe")
    x_return_format: Literal["markdown", "html", "text"] | None = Field(default=None, alias="X-Return-Format")
    x_token_budget: PositiveInt | None = Field(default=None, alias="X-Token-Budget")


class ReaderRequest(BaseModel):
    """Request model for Jina's Reader API."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    url: HttpUrl = Field(
        ...,
        description="The website URL to read/parse"
    )
    options: Literal["Default", "Markdown", "HTML", "Text", "Screenshot", "Pageshot"] | None = Field(
        default=None,
        description="Format options for response content"
    )
    headers: ReaderHeaders | None = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: HttpUrl) -> HttpUrl:
        """Validate that the URL is valid and not a placeholder."""
        str_url = str(v)
        if not is_valid_url(str_url):
            raise ValueError(f"Invalid or placeholder URL: {str_url}")
        return v


@tool
async def reader(
    url: str,
    options: str = "Default",
    with_links_summary: bool | None = None,
    with_images_summary: bool | None = None,
    with_generated_alt: bool | None = None,
    return_format: str | None = None,
    token_budget: int | None = None,
    no_cache: bool | None = None,
    state: Annotated[JinaToolState, InjectedState] = None,
    store: Annotated[BaseStore, InjectedStore] = None,
    config: Annotated[RunnableConfig, InjectedToolArg] = None
) -> Dict[str, Any]:
    """Parse and extract content from a webpage using Jina AI.
    
    Args:
        url: The URL of the webpage to read
        options: Format options ("Default", "Markdown", "HTML", "Text", "Screenshot", "Pageshot")
        with_links_summary: Include summary of links in the result
        with_images_summary: Include summary of images in the result
        with_generated_alt: Generate alt text for images
        return_format: Preferred format ("markdown", "html", "text")
        token_budget: Maximum number of tokens to return
        no_cache: Bypass API caching
        state: Injected graph state with API key or retry config
        
    Returns:
        Dictionary containing parsed webpage content
    """
    # Create headers if needed
    headers_dict = {}
    if any(v is not None for v in [with_links_summary, with_images_summary, with_generated_alt, return_format, token_budget, no_cache]):
        headers_dict = {
            "x_with_links_summary": with_links_summary,
            "x_with_images_summary": with_images_summary,
            "x_with_generated_alt": with_generated_alt,
            "x_return_format": return_format,
            "x_token_budget": token_budget,
            "x_no_cache": no_cache
        }
        # Remove None values
        headers_dict = {k: v for k, v in headers_dict.items() if v is not None}

    headers_model = ReaderHeaders(**headers_dict) if headers_dict else None

    # Create request object
    try:
        request = ReaderRequest(
            url=url,
            options=options,
            headers=headers_model
        )
    except Exception as e:
        raise ToolException(f"Invalid reader request: {str(e)}") from e

    # Create cache key for store - don't cache if no_cache is True
    should_use_cache = not no_cache and state and state.get("cache_results", True)
    cache_key = f"jina_reader_{url}_{options}"

    # Check cache if store is available and caching is enabled
    if store and should_use_cache:
        try:
            cached_data = store.get(["jina", cache_key])
            if cached_data and cached_data.value:
                info_highlight(f"Using cached reader result for {url}", "JinaTool")
                return cached_data.value
        except Exception as e:
            warning_highlight(f"Error accessing cache: {str(e)}", "JinaTool")

    # Get API key and retry configuration
    api_key = _get_jina_api_key(state, config)
    retry_config = state.get("retry_config") if state else None

    # Set up headers
    request_headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    # Add custom headers if provided
    if request.headers:
        for field_name, field_value in request.headers.model_dump(exclude_none=True, by_alias=True).items():
            if field_value is not None:
                # Convert bool to string
                if isinstance(field_value, bool):
                    field_value = "true" if field_value else "false"
                request_headers[field_name] = str(field_value)

    # Make the API request
    response = await _make_request_with_retry(
        method="POST",
        url="https://r.jina.ai/",
        headers=request_headers,
        json_data={"url": str(request.url), "options": request.options or "Default"},
        retry_config=retry_config,
    )

    # Check for errors
    if "error" in response:
        error_highlight(f"Reader API error: {response['error']}", "JinaTool")
        raise ToolException(f"Jina Reader API error: {response['error'].get('message', 'Unknown error')}")

    # Cache result if store is available and caching is enabled (and no_cache wasn't set)
    if store and should_use_cache:
        try:
            store.put(["jina", cache_key], response)
            info_highlight(f"Cached reader result for {url}", "JinaTool")
        except Exception as e:
            warning_highlight(f"Error caching reader result: {str(e)}", "JinaTool")

    return response


# -------------------------------------------------------------------------
# 4. Search API
# -------------------------------------------------------------------------

class SearchHeaders(BaseModel):
    """Optional headers for the Search API."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    x_site: str | None = Field(default=None, alias="X-Site")
    x_no_cache: bool | None = Field(default=None, alias="X-No-Cache")
    x_with_links_summary: bool | None = Field(default=None, alias="X-With-Links-Summary")
    x_with_images_summary: bool | None = Field(default=None, alias="X-With-Images-Summary")


class SearchRequest(BaseModel):
    """Request model for Jina's Search API."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    q: str = Field(
        ...,
        min_length=1,
        description="Search query text"
    )
    options: Literal["Default", "Markdown", "HTML", "Text"] | None = Field(
        default=None,
        description="Format options for response content"
    )
    headers: SearchHeaders | None = None

    @field_validator("q")
    @classmethod
    def no_placeholder_query(cls, v: str) -> str:
        """Validate that the search query is not empty."""
        if not v.strip():
            raise ValueError("Search query cannot be empty")
        return v


@tool
async def search(
    query: str,
    options: str = "Default",
    site: str | None = None,
    with_links_summary: bool | None = None,
    with_images_summary: bool | None = None,
    no_cache: bool | None = None,
    state: Annotated[JinaToolState, InjectedState] = None,
    store: Annotated[BaseStore, InjectedStore] = None,
    config: Annotated[RunnableConfig, InjectedToolArg] = None
) -> Dict[str, Any]:
    """Search the web using Jina AI.
    
    Args:
        query: The search query
        options: Format options ("Default", "Markdown", "HTML", "Text")
        site: Limit search to a specific site (e.g. "example.com")
        with_links_summary: Include summary of links in results
        with_images_summary: Include summary of images in results
        no_cache: Bypass API caching
        state: Injected graph state with API key or retry config
        
    Returns:
        Dictionary containing search results
    """
    # Create headers if needed
    headers_dict = {}
    if any(v is not None for v in [site, with_links_summary, with_images_summary, no_cache]):
        headers_dict = {
            "x_site": site,
            "x_with_links_summary": with_links_summary,
            "x_with_images_summary": with_images_summary,
            "x_no_cache": no_cache
        }
        # Remove None values
        headers_dict = {k: v for k, v in headers_dict.items() if v is not None}

    headers_model = SearchHeaders(**headers_dict) if headers_dict else None

    # Create request object
    try:
        request = SearchRequest(
            q=query,
            options=options,
            headers=headers_model
        )
    except Exception as e:
        raise ToolException(f"Invalid search request: {str(e)}") from e

    # Create cache key for store - don't cache if no_cache is True
    should_use_cache = not no_cache and state and state.get("cache_results", True)
    cache_key = f"jina_search_{query}_{options}_{site}"

    # Check cache if store is available and caching is enabled
    if store and should_use_cache:
        try:
            cached_data = store.get(["jina", cache_key])
            if cached_data and cached_data.value:
                info_highlight(f"Using cached search result for '{query}'", "JinaTool")
                return cached_data.value
        except Exception as e:
            warning_highlight(f"Error accessing cache: {str(e)}", "JinaTool")

    # Get API key and retry configuration
    api_key = _get_jina_api_key(state, config)
    retry_config = state.get("retry_config") if state else None

    # Set up headers
    request_headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    # Add custom headers if provided
    if request.headers:
        for field_name, field_value in request.headers.model_dump(exclude_none=True, by_alias=True).items():
            if field_value is not None:
                # Convert bool to string
                if isinstance(field_value, bool):
                    field_value = "true" if field_value else "false"
                request_headers[field_name] = str(field_value)

    # Make the API request
    response = await _make_request_with_retry(
        method="POST",
        url="https://s.jina.ai/",
        headers=request_headers,
        json_data={"q": request.q, "options": request.options or "Default"},
        retry_config=retry_config,
    )

    # Check for errors
    if "error" in response:
        error_highlight(f"Search API error: {response['error']}", "JinaTool")
        raise ToolException(f"Jina Search API error: {response['error'].get('message', 'Unknown error')}")

    # Cache result if store is available and caching is enabled (and no_cache wasn't set)
    if store and should_use_cache:
        try:
            store.put(["jina", cache_key], response)
            info_highlight(f"Cached search result for '{query}'", "JinaTool")
        except Exception as e:
            warning_highlight(f"Error caching search result: {str(e)}", "JinaTool")

    return response


# -------------------------------------------------------------------------
# 5. Grounding API
# -------------------------------------------------------------------------

class GroundingHeaders(BaseModel):
    """Optional headers for the Grounding API."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    x_site: str | None = Field(default=None, alias="X-Site")
    x_no_cache: bool | None = Field(default=None, alias="X-No-Cache")


class GroundingRequest(BaseModel):
    """Request model for Jina's Grounding API."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    statement: str = Field(
        ...,
        min_length=1,
        description="The statement to verify for factual accuracy"
    )
    headers: GroundingHeaders | None = None

    @field_validator("statement")
    @classmethod
    def statement_not_empty(cls, v: str) -> str:
        """Validate that the statement is not empty."""
        if not v.strip():
            raise ValueError("Statement cannot be empty")
        return v


@tool
async def grounding(
    statement: str,
    site: str | None = None,
    no_cache: bool | None = None,
    state: Annotated[JinaToolState, InjectedState] = None,
    store: Annotated[BaseStore, InjectedStore] = None,
    config: Annotated[RunnableConfig, InjectedToolArg] = None
) -> Dict[str, Any]:
    """Verify the factual accuracy of a statement using Jina AI.
    
    Args:
        statement: The statement to verify
        site: Limit search to a specific site (e.g. "example.com")
        no_cache: Bypass API caching
        state: Injected graph state with API key or retry config
        
    Returns:
        Dictionary containing verification results with evidence
    """
    # Create headers if needed
    headers_dict = {}
    if any(v is not None for v in [site, no_cache]):
        headers_dict = {
            "x_site": site,
            "x_no_cache": no_cache
        }
        # Remove None values
        headers_dict = {k: v for k, v in headers_dict.items() if v is not None}

    headers_model = GroundingHeaders(**headers_dict) if headers_dict else None

    # Create request object
    try:
        request = GroundingRequest(
            statement=statement,
            headers=headers_model
        )
    except Exception as e:
        error_msg = f"Invalid grounding request: {str(e)}"
        error_highlight(error_msg, "JinaTool")
        raise ToolException(error_msg) from None  # Disable exception chaining

    # Create cache key for store - don't cache if no_cache is True
    should_use_cache = not no_cache and state and state.get("cache_results", True)
    cache_key = f"jina_grounding_{statement}_{site}"

    # Check cache if store is available and caching is enabled
    if store and should_use_cache:
        try:
            cached_data = store.get(["jina", cache_key])
            if cached_data and cached_data.value:
                info_highlight("Using cached grounding result for statement", "JinaTool")
                return cached_data.value
        except Exception as e:
            warning_highlight(f"Error accessing cache: {str(e)}", "JinaTool")

    # Get API key and retry configuration
    api_key = _get_jina_api_key(state, config)
    retry_config = state.get("retry_config") if state else None

    # Set up headers
    request_headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    # Add custom headers if provided
    if request.headers:
        for field_name, field_value in request.headers.model_dump(exclude_none=True, by_alias=True).items():
            if field_value is not None:
                # Convert bool to string
                if isinstance(field_value, bool):
                    field_value = "true" if field_value else "false"
                request_headers[field_name] = str(field_value)

    # Make the API request
    response = await _make_request_with_retry(
        method="POST",
        url="https://g.jina.ai/",
        headers=request_headers,
        json_data={"statement": request.statement},
        retry_config=retry_config,
    )

    # Check for errors
    if "error" in response:
        error_highlight(f"Grounding API error: {response['error']}", "JinaTool")
        raise ToolException(f"Jina Grounding API error: {response['error'].get('message', 'Unknown error')}")

    # Cache result if store is available and caching is enabled (and no_cache wasn't set)
    if store and should_use_cache:
        try:
            store.put(["jina", cache_key], response)
            info_highlight("Cached grounding result for statement", "JinaTool")
        except Exception as e:
            warning_highlight(f"Error caching grounding result: {str(e)}", "JinaTool")

    return response


# -------------------------------------------------------------------------
# 6. Segmenter API
# -------------------------------------------------------------------------

class SegmenterRequest(BaseModel):
    """Request model for Jina's Segmenter API."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    content: str = Field(
        ...,
        min_length=1,
        description="The text content to segment"
    )
    tokenizer: Literal["cl100k_base", "p50k_base", "r50k_base"] = Field(
        default="cl100k_base",
        description="Tokenizer to use"
    )
    return_tokens: bool = Field(
        default=False,
        description="Include tokens and IDs in response"
    )
    return_chunks: bool = Field(
        default=False,
        description="Segment text into semantic chunks"
    )
    max_chunk_length: PositiveInt = Field(
        default=1000,
        description="Maximum characters per chunk if return_chunks=True"
    )
    head: PositiveInt | None = Field(
        default=None,
        description="Return first N tokens (exclusive with tail)"
    )
    tail: PositiveInt | None = Field(
        default=None,
        description="Return last N tokens (exclusive with head)"
    )

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Validate that the content is not empty."""
        if not v.strip():
            raise ValueError("Content cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_head_tail(self) -> "SegmenterRequest":
        """Ensure head and tail are not both specified."""
        if self.head is not None and self.tail is not None:
            raise ValueError("Only one of head or tail can be specified")
        return self


@tool
async def segmenter(
    content: str,
    tokenizer: str = "cl100k_base",
    return_tokens: bool = False,
    return_chunks: bool = False,
    max_chunk_length: int = 1000,
    head: int | None = None,
    tail: int | None = None,
    state: Annotated[JinaToolState, InjectedState] = None,
    store: Annotated[BaseStore, InjectedStore] = None,
    config: Annotated[RunnableConfig, InjectedToolArg] = None
) -> Dict[str, Any]:
    """Segment and tokenize text using Jina AI.
    
    Args:
        content: The text content to segment
        tokenizer: Tokenizer to use ("cl100k_base", "p50k_base", "r50k_base")
        return_tokens: Include tokens and IDs in the response
        return_chunks: Segment the text into semantic chunks
        max_chunk_length: Maximum characters per chunk if return_chunks=True
        head: Return first N tokens (exclusive with tail)
        tail: Return last N tokens (exclusive with head)
        state: Injected graph state with API key or retry config
        
    Returns:
        Dictionary containing tokenization results
    """
    # Create request object
    try:
        request = SegmenterRequest(
            content=content,
            tokenizer=tokenizer,
            return_tokens=return_tokens,
            return_chunks=return_chunks,
            max_chunk_length=max_chunk_length,
            head=head,
            tail=tail
        )
    except Exception as e:
        raise ToolException(f"Invalid segmenter request: {str(e)}") from e

    # Create cache key for store
    content_hash = str(hash(content))
    cache_key = f"jina_segmenter_{tokenizer}_{content_hash}_{return_tokens}_{return_chunks}_{max_chunk_length}_{head}_{tail}"

    # Check cache if store is available and caching is enabled
    if store and state and state.get("cache_results", True):
        try:
            cached_data = store.get(["jina", cache_key])
            if cached_data and cached_data.value:
                info_highlight("Using cached segmenter result", "JinaTool")
                return cached_data.value
        except Exception as e:
            warning_highlight(f"Error accessing cache: {str(e)}", "JinaTool")

    # Get API key and retry configuration
    api_key = _get_jina_api_key(state, config)
    retry_config = state.get("retry_config") if state else None

    # Set up headers
    request_headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    # Make the API request
    response = await _make_request_with_retry(
        method="POST",
        url="https://segment.jina.ai/",
        headers=request_headers,
        json_data=request.model_dump(exclude_none=True),
        retry_config=retry_config,
    )

    # Check for errors
    if "error" in response:
        error_highlight(f"Segmenter API error: {response['error']}", "JinaTool")
        raise ToolException(f"Jina Segmenter API error: {response['error'].get('message', 'Unknown error')}")

    # Cache result if store is available and caching is enabled
    if store and state and state.get("cache_results", True):
        try:
            store.put(["jina", cache_key], response)
            info_highlight("Cached segmenter result", "JinaTool")
        except Exception as e:
            warning_highlight(f"Error caching segmenter result: {str(e)}", "JinaTool")

    return response


# -------------------------------------------------------------------------
# 7. Classifier API
# -------------------------------------------------------------------------

class ClassifierRequest(BaseModel):
    """Request model for Jina's Classifier API."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str | None = Field(
        default=None,
        description="Model identifier, e.g. 'jina-embeddings-v3' for text"
    )
    classifier_id: str | None = Field(
        default=None,
        description="Identifier of an existing classifier"
    )
    input: List[Union[str, Dict[str, str]]] = Field(
        ...,
        min_length=1,
        description="Inputs for classification"
    )
    labels: List[str] = Field(
        ...,
        min_length=1,
        description="Classification labels"
    )

    @field_validator("input")
    @classmethod
    def validate_input_not_empty(cls, v: List[Union[str, Dict[str, str]]]) -> List[Union[str, Dict[str, str]]]:
        """Validate that the input list is not empty."""
        if not v:
            raise ValueError("Classifier input cannot be empty")
        return v

    @field_validator("labels")
    @classmethod
    def validate_labels_not_empty(cls, v: List[str]) -> List[str]:
        """Validate that the labels list has no empty strings."""
        if not v:
            raise ValueError("Classifier labels cannot be empty")
        for label in v:
            if not label.strip():
                raise ValueError("Classifier labels cannot contain empty strings")
        return v


@tool
async def classifier(
    inputs: List[Union[str, Dict[str, str]]],
    labels: List[str],
    model: str | None = None,
    classifier_id: str | None = None,
    state: Annotated[JinaToolState, InjectedState] = None,
    store: Annotated[BaseStore, InjectedStore] = None,
    config: Annotated[RunnableConfig, InjectedToolArg] = None
) -> Dict[str, Any]:
    """Perform zero-shot classification with Jina AI.
    
    Args:
        inputs: Inputs for classification (text strings or image dicts)
        labels: List of classification labels
        model: Model identifier (optional)
        classifier_id: Existing classifier ID (optional)
        state: Injected graph state with API key or retry config
        
    Returns:
        Dictionary containing classification results
    """
    # Create request object
    try:
        request = ClassifierRequest(
            input=inputs,
            labels=labels,
            model=model,
            classifier_id=classifier_id
        )
    except Exception as e:
        raise ToolException(f"Invalid classifier request: {str(e)}") from e

    # Create cache key for store
    inputs_hash = str(hash(str(inputs)))
    labels_hash = str(hash(str(labels)))
    cache_key = f"jina_classifier_{model}_{classifier_id}_{inputs_hash}_{labels_hash}"

    # Check cache if store is available and caching is enabled
    if store and state and state.get("cache_results", True):
        try:
            cached_data = store.get(["jina", cache_key])
            if cached_data and cached_data.value:
                info_highlight("Using cached classifier result", "JinaTool")
                return cached_data.value
        except Exception as e:
            warning_highlight(f"Error accessing cache: {str(e)}", "JinaTool")

    # Get API key and retry configuration
    api_key = _get_jina_api_key(state, config)
    retry_config = state.get("retry_config") if state else None

    # Set up headers
    request_headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    # Make the API request
    response = await _make_request_with_retry(
        method="POST",
        url="https://api.jina.ai/v1/classify",
        headers=request_headers,
        json_data=request.model_dump(exclude_none=True),
        retry_config=retry_config,
    )

    # Check for errors
    if "error" in response:
        error_highlight(f"Classifier API error: {response['error']}", "JinaTool")
        raise ToolException(f"Jina Classifier API error: {response['error'].get('message', 'Unknown error')}")

    # Cache result if store is available and caching is enabled
    if store and state and state.get("cache_results", True):
        try:
            store.put(["jina", cache_key], response)
            info_highlight("Cached classifier result", "JinaTool")
        except Exception as e:
            warning_highlight(f"Error caching classifier result: {str(e)}", "JinaTool")

    return response


# -------------------------------------------------------------------------
# LangGraph Integration
# -------------------------------------------------------------------------

def create_jina_toolnode(
    include_tools: List[str] | None = None,
    retry_config: RetryConfig | None = None,
    cache_results: bool = True
) -> ToolNode:
    """Create a LangGraph ToolNode with Jina AI tools.
    
    This function creates a ToolNode containing the specified Jina AI tools
    for use in a LangGraph workflow.
    
    Args:
        include_tools: Optional list of tool names to include. If None, all tools are included.
                      Available tools: ["embeddings", "rerank", "reader", 
                                       "search", "grounding", "segmenter", 
                                       "classifier"]
        retry_config: Optional retry configuration for HTTP requests
        cache_results: Whether to cache tool results in the store (default: True)
    
    Returns:
        A configured ToolNode instance with the specified Jina AI tools
        
    Example:
        ```python
        from langgraph.graph import StateGraph
        from react_agent.tools.jina import create_jina_toolnode
        
        # Create a graph with all Jina tools
        workflow = StateGraph(MessagesState)
        workflow.add_node("jina_tools", create_jina_toolnode())
        
        # Or with specific tools
        search_tools = create_jina_toolnode(include_tools=["search", "grounding"])
        workflow.add_node("search_tools", search_tools)
        ```
    """
    all_tools = [
        embeddings,
        rerank,
        reader,
        search,
        grounding,
        segmenter,
        classifier
    ]
    
    if include_tools:
        # Map of tool names to tool functions
        tool_map = {
            "embeddings": embeddings,
            "rerank": rerank,
            "reader": reader,
            "search": search,
            "grounding": grounding,
            "segmenter": segmenter,
            "classifier": classifier
        }
        
        # Get only the requested tools
        selected_tools = []
        for tool_name in include_tools:
            if tool_name in tool_map:
                selected_tools.append(tool_map[tool_name])
            else:
                warning_highlight(f"Tool {tool_name} not found in available Jina tools", "JinaTool")
        
        if not selected_tools:
            warning_highlight("No valid tools specified, including all tools", "JinaTool")
            selected_tools = all_tools
            
        tools = selected_tools
    else:
        tools = all_tools
    
    # Create and return the ToolNode
    return ToolNode(tools)