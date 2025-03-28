"""Jina Tools - Production-Ready Implementation.

Get your Jina AI API key for free: https://jina.ai/?sui=apikey

This module provides production-ready tool classes for interacting with
Jina AI Search Foundation APIs. Each tool is strictly typed, uses aiohttp
for HTTP requests, implements basic retries, and leverages utility modules
from src/react_agent/utils for logging, validation, and error handling.

Usage:
    1. Set the JINA_API_KEY environment variable with your API key.
    2. Instantiate any of the tool classes below and call their `arun(...)`
       method (async) with the appropriate input model.

Environment Variable:
    JINA_API_KEY - Your Bearer token for authorization
                   (obtain from https://jina.ai/?sui=apikey)

APIs Supported:
    - Embeddings (POST https://api.jina.ai/v1/embeddings)
    - Re-ranker (POST https://api.jina.ai/v1/rerank)
    - Reader (POST https://r.jina.ai/)
    - Search (POST https://s.jina.ai/)
    - Grounding (POST https://g.jina.ai/)
    - Segmenter (POST https://segment.jina.ai/)
    - Classifier (POST https://api.jina.ai/v1/classify)

Simplicity is prioritized. Each tool focuses on a single API endpoint.
No complex chaining is implemented. Retries and error handling are
provided for production robustness. Type safety is ensured via Pydantic v2.
"""

import asyncio
import json
import os
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict, Union

import aiohttp
from aiohttp.client_exceptions import ClientError
from langchain.tools import BaseTool

# Pydantic v2 imports
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    PositiveFloat,
    PositiveInt,
    StringConstraints,
    field_validator,
    model_validator,
)

# Utilities from src/react_agent/utils
from react_agent.utils.logging import (
    error_highlight,
    get_logger,
    info_highlight,
    warning_highlight,
)
from react_agent.utils.validations import is_valid_url

logger = get_logger(__name__)


# -------------------------------------------------------------------------
# Type definitions for API responses
# -------------------------------------------------------------------------
class ErrorResponse(TypedDict):
    """Type definition for error responses from Jina APIs."""
    error: Dict[str, Any]


class RawResponse(TypedDict):
    """Type definition for raw responses from Jina APIs."""
    raw_response: str
    status: int


ApiResponse = Dict[str, Any]


# -------------------------------------------------------------------------
# Global Retry Configuration
# -------------------------------------------------------------------------
class RetryConfig(BaseModel):
    """Basic configuration for retrying failed network requests."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_retries: PositiveInt = Field(
        default=3,
        description="Maximum number of retries."
    )
    base_delay: PositiveFloat = Field(
        default=1.0,
        description="Base delay in seconds."
    )
    max_delay: PositiveFloat = Field(
        default=5.0,
        description="Maximum delay in seconds."
    )

    @model_validator(mode="after")
    def validate_delays(self) -> "RetryConfig":
        """Ensure max_delay is greater than or equal to base_delay."""
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be greater than or equal to base_delay")
        return self


# -------------------------------------------------------------------------
# Shared Helper Function: _make_request_with_retry
# -------------------------------------------------------------------------
async def _make_request_with_retry(
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"],
    url: str,
    headers: Dict[str, str],
    json_data: Dict[str, Any] | None = None,
    retry_config: RetryConfig | None = None,
) -> ApiResponse:
    """Make an HTTP request with basic exponential backoff retries.

    Args:
        method: HTTP method (GET, POST, etc.)
        url: Target URL for the request
        headers: HTTP headers to include
        json_data: Optional JSON payload
        retry_config: Retry configuration settings

    Returns:
        Parsed JSON response or error information
    """
    if retry_config is None:
        retry_config = RetryConfig()

    attempt = 0
    last_exception: Optional[Exception] = None

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
                ) as resp:
                    status = resp.status
                    text = await resp.text()

                    # Handle special 422 (e.g. invalid input)
                    if status == 422:
                        warning_highlight(f"422 Unprocessable Entity - detail: {text}", "JinaTool")
                        # Return the error as part of the JSON for clarity
                        return {"error": {"status": status, "message": text}}

                    # For success codes 200-299
                    if 200 <= status < 300:
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            # If the response isn't valid JSON, fallback
                            return {"raw_response": text, "status": status}

                    # Otherwise, treat as error
                    error_msg = f"HTTP {status} - {text}"
                    raise ClientError(error_msg)

        except ClientError as ce:
            last_exception = ce
            warning_highlight(
                f"ClientError on attempt {attempt}/{retry_config.max_retries}: {str(ce)}",
                "JinaTool",
            )
        except Exception as e:
            last_exception = e
            warning_highlight(
                f"Exception on attempt {attempt}/{retry_config.max_retries}: {str(e)}",
                "JinaTool",
            )

        await asyncio.sleep(min(retry_config.max_delay, retry_config.base_delay * attempt))

    # If we get here, all attempts failed
    error_highlight(f"All retry attempts failed: {last_exception}", "JinaTool")
    return {"error": {"message": str(last_exception), "status": -1}}


# -------------------------------------------------------------------------
# Common: Acquire JINA_API_KEY
# -------------------------------------------------------------------------
def _get_jina_api_key() -> str:
    """Fetch the JINA_API_KEY from environment variable.

    Returns:
        str: The API key

    Raises:
        KeyError: If JINA_API_KEY environment variable is not set
    """
    key = os.environ.get("JINA_API_KEY", None)
    if not key:
        raise KeyError("JINA_API_KEY environment variable is not set.")
    return key


# -------------------------------------------------------------------------
# 1) Embeddings API
#    Endpoint: https://api.jina.ai/v1/embeddings
# -------------------------------------------------------------------------
class EmbeddingsRequest(BaseModel):
    """Request model for Jina's Embeddings API."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(
        ...,
        description="Identifier of the model to use. E.g. 'jina-embeddings-v3' or 'jina-clip-v2'."
    )
    input: List[str] = Field(
        ...,
        min_length=1,
        description="Array of input strings (text) or base64-encoded images to be embedded."
    )
    embedding_type: Optional[Union[Literal["float", "base64"], List[Literal["float", "base64"]]]] = Field(
        default="float",
        description="Format of returned embeddings. Could be 'float', 'base64', etc."
    )
    task: Optional[str] = Field(
        default=None,
        description="Intended downstream application, e.g. 'retrieval.query'."
    )
    dimensions: Optional[PositiveInt] = Field(
        default=None,
        description="Truncates output embeddings to the specified size if set."
    )
    normalized: bool = Field(
        default=False,
        description="Normalize embeddings to unit L2 norm if True."
    )
    late_chunking: bool = Field(
        default=False,
        description="Concatenate all sentences and treat as a single input for late chunking if True."
    )


# Response type definitions for embeddings API
class EmbeddingData(TypedDict):
    embedding: List[float]
    index: int
    object: str
"""Embedding data."""
 
class EmbeddingsResponse(TypedDict):
    data: List[EmbeddingData]
    model: str
    object: str
    usage: Dict[str, int]
"""Embeddings response."""


class EmbeddingsTool(BaseTool):
    """Tool for calling the Jina Embeddings API to generate embeddings for text/images.
    """
    name: str = "EmbeddingsTool"
    description: str = "Use this tool to convert text or images into vector embeddings via Jina's Embeddings API. Requires the JINA_API_KEY environment variable."

    retry_config: RetryConfig = Field(default_factory=RetryConfig)

    async def _arun(self, request: EmbeddingsRequest, **kwargs: Any) -> Union[EmbeddingsResponse, ErrorResponse, RawResponse]:
        """Asynchronously call the Jina Embeddings API.

        Args:
            request: The embeddings request parameters

        Returns:
            The API response containing embeddings or error information
        """
        url = "https://api.jina.ai/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {_get_jina_api_key()}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = request.model_dump(exclude_none=True)

        resp = await _make_request_with_retry(
            method="POST",
            url=url,
            headers=headers,
            json_data=payload,
            retry_config=self.retry_config,
        )
        return resp

    def run(self, request: EmbeddingsRequest, **kwargs: Any) -> Union[EmbeddingsResponse, ErrorResponse, RawResponse]:
        """Synchronous wrapper for calling the Jina Embeddings API.

        Args:
            request: The embeddings request parameters

        Returns:
            The API response containing embeddings or error information
        """
        return asyncio.run(self._arun(request, **kwargs))


# -------------------------------------------------------------------------
# 2) Re-Ranker API
#    Endpoint: https://api.jina.ai/v1/rerank
# -------------------------------------------------------------------------
class RerankerRequest(BaseModel):
    """Request model for Jina's Re-Ranker API."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(
        ...,
        description="Identifier of the model to use. E.g. 'jina-reranker-v2-base-multilingual'."
    )
    query: Annotated[str, StringConstraints(min_length=1)] = Field(
        ...,
        description="The search query for re-ranking."
    )
    documents: List[Union[str, Dict[str, Any]]] = Field(
        ...,
        min_length=1,
        description="List of documents or strings to re-rank."
    )
    top_n: Optional[PositiveInt] = Field(
        default=None,
        description="Number of top documents to return."
    )
    return_documents: bool = Field(
        default=True,
        description="Return documents in the response if True."
    )


# Response type definitions for re-ranker API
class RerankerScoreData(TypedDict):
    document: Union[str, Dict[str, Any]]
    index: int
    relevance_score: float
"""Re-ranker score data."""
 
class RerankerResponse(TypedDict):
    model: str
    results: List[RerankerScoreData]
"""Re-ranker response."""


class RerankerTool(BaseTool):
    """Tool for calling the Jina Re-Ranker API to refine search results.
    """
    name: str = "RerankerTool"
    description: str = "Use this tool to re-rank search results using Jina's Re-Ranker API. Requires the JINA_API_KEY environment variable."

    retry_config: RetryConfig = Field(default_factory=RetryConfig)

    async def _arun(self, request: RerankerRequest, **kwargs: Any) -> Union[RerankerResponse, ErrorResponse, RawResponse]:
        """Asynchronously call the Jina Re-Ranker API.

        Args:
            request: The re-ranker request parameters

        Returns:
            The API response containing re-ranked documents or error information
        """
        url = "https://api.jina.ai/v1/rerank"
        headers = {
            "Authorization": f"Bearer {_get_jina_api_key()}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = request.model_dump(exclude_none=True)

        resp = await _make_request_with_retry(
            method="POST",
            url=url,
            headers=headers,
            json_data=payload,
            retry_config=self.retry_config,
        )
        return resp

    def run(self, request: RerankerRequest, **kwargs: Any) -> Union[RerankerResponse, ErrorResponse, RawResponse]:
        """Synchronous wrapper for calling the Jina Re-Ranker API.

        Args:
            request: The re-ranker request parameters

        Returns:
            The API response containing re-ranked documents or error information
        """
        return asyncio.run(self._arun(request, **kwargs))


# -------------------------------------------------------------------------
# 3) Reader API
#    Endpoint: https://r.jina.ai/
# -------------------------------------------------------------------------
class ReaderHeaders(BaseModel):
    """Headers for the Reader API."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    x_engine: Optional[str] = Field(default=None, alias="X-Engine")
    x_timeout: Optional[PositiveInt] = Field(default=None, alias="X-Timeout")
    x_target_selector: Optional[str] = Field(default=None, alias="X-Target-Selector")
    x_wait_for_selector: Optional[str] = Field(default=None, alias="X-Wait-For-Selector")
    x_remove_selector: Optional[str] = Field(default=None, alias="X-Remove-Selector")
    x_with_links_summary: Optional[bool] = Field(default=None, alias="X-With-Links-Summary")
    x_with_images_summary: Optional[bool] = Field(default=None, alias="X-With-Images-Summary")
    x_with_generated_alt: Optional[bool] = Field(default=None, alias="X-With-Generated-Alt")
    x_no_cache: Optional[bool] = Field(default=None, alias="X-No-Cache")
    x_with_iframe: Optional[bool] = Field(default=None, alias="X-With-Iframe")
    x_return_format: Optional[Literal["markdown", "html", "text"]] = Field(default=None, alias="X-Return-Format")
    x_token_budget: Optional[PositiveInt] = Field(default=None, alias="X-Token-Budget")
    x_retain_images: Optional[Literal["none", "all"]] = Field(default=None, alias="X-Retain-Images")


class ReaderRequest(BaseModel):
    """Request model for Jina's Reader API."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    url: HttpUrl = Field(
        ...,
        description="The website URL to read/parse."
    )
    options: Optional[Literal["Default", "Markdown", "HTML", "Text", "Screenshot", "Pageshot"]] = Field(
        default=None,
        description="Extra options; often one of 'Default', 'Markdown', 'HTML', 'Text', etc."
    )
    headers: Optional[ReaderHeaders] = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validates that the URL is valid and not a placeholder."""
        if not is_valid_url(v):
            raise ValueError(f"Invalid or placeholder URL: {v}")
        return v


# Response type definitions for reader API
class ReaderResponse(TypedDict):
    content: str
    links: Optional[List[Dict[str, str]]]
    images: Optional[List[Dict[str, str]]]
    title: Optional[str]
    url: str
"""Reader response."""


class ReaderTool(BaseTool):
    """Tool for calling the Jina Reader API to retrieve/parse web content.
    """
    name: str = "ReaderTool"
    description: str = "Use this tool to scrape and parse a single webpage using Jina's Reader API. Requires the JINA_API_KEY environment variable."

    retry_config: RetryConfig = Field(default_factory=RetryConfig)

    async def _arun(self, request: ReaderRequest, **kwargs: Any) -> Union[ReaderResponse, ErrorResponse, RawResponse]:
        """Asynchronously call the Jina Reader API.

        Args:
            request: The reader request parameters

        Returns:
            The API response containing webpage content or error information
        """
        url = "https://r.jina.ai/"
        headers = {
            "Authorization": f"Bearer {_get_jina_api_key()}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Append custom headers if provided
        if request.headers:
            for field_name, field_value in request.headers.model_dump(exclude_none=True, by_alias=True).items():
                if field_value is not None:
                    # If bool, convert to "true"/"false"
                    if isinstance(field_value, bool):
                        field_value = "true" if field_value else "false"
                    headers[field_name] = str(field_value)

        payload = {
            "url": str(request.url),
            "options": request.options or "Default",
        }

        resp = await _make_request_with_retry(
            method="POST",
            url=url,
            headers=headers,
            json_data=payload,
            retry_config=self.retry_config,
        )
        return resp

    def run(self, request: ReaderRequest, **kwargs: Any) -> Union[ReaderResponse, ErrorResponse, RawResponse]:
        """Synchronous wrapper for calling the Jina Reader API.

        Args:
            request: The reader request parameters

        Returns:
            The API response containing webpage content or error information
        """
        return asyncio.run(self._arun(request, **kwargs))


# -------------------------------------------------------------------------
# 4) Search API
#    Endpoint: https://s.jina.ai/
# -------------------------------------------------------------------------
class SearchHeaders(BaseModel):
    """Headers for the Search API requests."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    x_site: Optional[str] = Field(default=None, alias="X-Site")
    x_no_cache: Optional[bool] = Field(default=None, alias="X-No-Cache")
    x_with_links_summary: Optional[bool] = Field(default=None, alias="X-With-Links-Summary")
    x_with_images_summary: Optional[bool] = Field(default=None, alias="X-With-Images-Summary")


class SearchRequest(BaseModel):
    """Request model for Jina's Search API."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    q: Annotated[str, StringConstraints(min_length=1)] = Field(
        ...,
        description="Search query text."
    )
    options: Optional[Literal["Default", "Markdown", "HTML", "Text"]] = Field(
        default=None,
        description="Extra options; often 'Default', 'Markdown', 'HTML', 'Text', etc."
    )
    headers: Optional[SearchHeaders] = None

    @field_validator("q")
    @classmethod
    def no_placeholder_query(cls, v: str) -> str:
        """Validates that the search query is not empty."""
        if not v.strip():
            raise ValueError("Search query cannot be empty.")
        return v


# Response type definitions for search API
class SearchResultItem(TypedDict):
    title: str
    url: str
    snippet: str


class SearchResponse(TypedDict):
    results: List[SearchResultItem]
    query: str


class SearchTool(BaseTool):
    """Tool for calling the Jina Search API to retrieve web results.
    """
    name: str = "SearchTool"
    description: str = (
        "Use this tool to perform web search queries using Jina's Search API. "
        "Requires the JINA_API_KEY environment variable."
    )

    retry_config: RetryConfig = Field(default_factory=RetryConfig)

    async def _arun(self, request: SearchRequest, **kwargs: Any) -> Union[SearchResponse, ErrorResponse, RawResponse]:
        """Asynchronously call the Jina Search API.

        Args:
            request: The search request parameters

        Returns:
            The API response containing search results or error information
        """
        url = "https://s.jina.ai/"
        headers = {
            "Authorization": f"Bearer {_get_jina_api_key()}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Append custom headers if provided
        if request.headers:
            for field_name, field_value in request.headers.model_dump(exclude_none=True, by_alias=True).items():
                if field_value is not None:
                    if isinstance(field_value, bool):
                        field_value = "true" if field_value else "false"
                    headers[field_name] = str(field_value)

        payload = {
            "q": request.q,
            "options": request.options or "Default",
        }

        resp = await _make_request_with_retry(
            method="POST",
            url=url,
            headers=headers,
            json_data=payload,
            retry_config=self.retry_config,
        )
        return resp

    def run(self, request: SearchRequest, **kwargs: Any) -> Union[SearchResponse, ErrorResponse, RawResponse]:
        """Synchronous wrapper for calling the Jina Search API.

        Args:
            request: The search request parameters

        Returns:
            The API response containing search results or error information
        """
        return asyncio.run(self._arun(request, **kwargs))


# -------------------------------------------------------------------------
# 5) Grounding API
#    Endpoint: https://g.jina.ai/
# -------------------------------------------------------------------------
class GroundingHeaders(BaseModel):
    """Headers for the Grounding API requests."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    x_site: Optional[str] = Field(default=None, alias="X-Site")
    x_no_cache: Optional[bool] = Field(default=None, alias="X-No-Cache")


class GroundingRequest(BaseModel):
    """Request model for Jina's Grounding API."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    statement: Annotated[str, StringConstraints(min_length=1)] = Field(
        ...,
        description="The statement to verify for factual accuracy."
    )
    headers: Optional[GroundingHeaders] = None

    @field_validator("statement")
    @classmethod
    def statement_not_empty(cls, v: str) -> str:
        """Validates that the statement is not empty."""
        if not v.strip():
            raise ValueError("Statement cannot be empty.")
        return v


# Response type definitions for grounding API
class GroundingEvidence(TypedDict):
    text: str
    url: str
    score: float


class GroundingResult(TypedDict):
    grounding: Literal["supported", "contradicted", "ambiguous", "opinion", "off-topic"]
    score: float
    statement: str
    evidence: List[GroundingEvidence]


class GroundingTool(BaseTool):
    """Tool for calling the Jina Grounding API to verify statements.
    """
    name: str = "GroundingTool"
    description: str = (
        "Use this tool to verify the factual accuracy of a statement using Jina's Grounding API. "
        "Requires the JINA_API_KEY environment variable."
    )

    retry_config: RetryConfig = Field(default_factory=RetryConfig)

    async def _arun(self, request: GroundingRequest, **kwargs: Any) -> Union[GroundingResult, ErrorResponse, RawResponse]:
        """Asynchronously call the Jina Grounding API.

        Args:
            request: The grounding request parameters

        Returns:
            The API response containing factual verification or error information
        """
        url = "https://g.jina.ai/"
        headers = {
            "Authorization": f"Bearer {_get_jina_api_key()}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Append custom headers if provided
        if request.headers:
            for field_name, field_value in request.headers.model_dump(exclude_none=True, by_alias=True).items():
                if field_value is not None:
                    if isinstance(field_value, bool):
                        field_value = "true" if field_value else "false"
                    headers[field_name] = str(field_value)

        payload = {
            "statement": request.statement,
        }

        resp = await _make_request_with_retry(
            method="POST",
            url=url,
            headers=headers,
            json_data=payload,
            retry_config=self.retry_config,
        )
        return resp

    def run(self, request: GroundingRequest, **kwargs: Any) -> Union[GroundingResult, ErrorResponse, RawResponse]:
        """Synchronous wrapper for calling the Jina Grounding API.

        Args:
            request: The grounding request parameters

        Returns:
            The API response containing factual verification or error information
        """
        return asyncio.run(self._arun(request, **kwargs))


# -------------------------------------------------------------------------
# 6) Segmenter API
#    Endpoint: https://segment.jina.ai/
# -------------------------------------------------------------------------
class SegmenterRequest(BaseModel):
    """Request model for Jina's Segmenter API."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    content: Annotated[str, StringConstraints(min_length=1)] = Field(
        ...,
        description="The text content to segment."
    )
    tokenizer: Literal["cl100k_base", "p50k_base", "r50k_base"] = Field(
        default="cl100k_base",
        description="Specifies the tokenizer to use (cl100k_base, p50k_base, etc.)."
    )
    return_tokens: bool = Field(
        default=False,
        description="If True, includes tokens and their IDs in the response."
    )
    return_chunks: bool = Field(
        default=False,
        description="If True, segments the text into semantic chunks."
    )
    max_chunk_length: PositiveInt = Field(
        default=1000,
        description="Maximum characters per chunk if return_chunks=True."
    )
    head: Optional[PositiveInt] = Field(
        default=None,
        description="Returns the first N tokens (exclusive with tail)."
    )
    tail: Optional[PositiveInt] = Field(
        default=None,
        description="Returns the last N tokens (exclusive with head)."
    )

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Validates that the content is not empty."""
        if not v.strip():
            raise ValueError("Segmenter content cannot be empty.")
        return v

    @model_validator(mode="after")
    def validate_head_tail(self) -> "SegmenterRequest":
        """Ensures head and tail are not both specified."""
        if self.head is not None and self.tail is not None:
            raise ValueError("Only one of head or tail can be specified, not both.")
        return self


# Response type definitions for segmenter API
class TokenInfo(TypedDict):
    id: int
    text: str
    start: int
    end: int


class ChunkInfo(TypedDict):
    text: str
    start: int
    end: int


class SegmenterResponse(TypedDict):
    tokens: Optional[List[TokenInfo]]
    chunks: Optional[List[ChunkInfo]]
    token_count: int
    character_count: int


class SegmenterTool(BaseTool):
    """Tool for calling the Jina Segmenter API to tokenize or chunk text.
    """
    name: str = "SegmenterTool"
    description: str = (
        "Use this tool to segment and tokenize text using Jina's Segmenter API. "
        "Requires the JINA_API_KEY environment variable."
    )

    retry_config: RetryConfig = Field(default_factory=RetryConfig)

    async def _arun(self, request: SegmenterRequest, **kwargs: Any) -> Union[SegmenterResponse, ErrorResponse, RawResponse]:
        """Asynchronously call the Jina Segmenter API.

        Args:
            request: The segmenter request parameters

        Returns:
            The API response containing segmented text or error information
        """
        url = "https://segment.jina.ai/"
        headers = {
            "Authorization": f"Bearer {_get_jina_api_key()}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        payload = request.model_dump(exclude_none=True)
        resp = await _make_request_with_retry(
            method="POST",
            url=url,
            headers=headers,
            json_data=payload,
            retry_config=self.retry_config,
        )
        return resp

    def run(self, request: SegmenterRequest, **kwargs: Any) -> Union[SegmenterResponse, ErrorResponse, RawResponse]:
        """Synchronous wrapper for calling the Jina Segmenter API.

        Args:
            request: The segmenter request parameters

        Returns:
            The API response containing segmented text or error information
        """
        return asyncio.run(self._arun(request, **kwargs))


# -------------------------------------------------------------------------
# 7) Classifier API
#    Endpoint: https://api.jina.ai/v1/classify
# -------------------------------------------------------------------------
class ClassifierRequest(BaseModel):
    """Request model for Jina's Classifier API."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: Optional[str] = Field(
        default=None,
        description="Identifier of the model, e.g. 'jina-embeddings-v3' for text or 'jina-clip-v2' for images."
    )
    classifier_id: Optional[str] = Field(
        default=None,
        description="Identifier of an existing classifier. If not provided, a new one is created."
    )
    input: List[Union[str, Dict[str, str]]] = Field(
        ...,
        min_length=1,
        description=(
            "Array of inputs for classification. For text classification, pass strings. "
            "For image classification, pass dicts like {'image': 'base64_string'}."
        )
    )
    labels: List[str] = Field(
        ...,
        min_length=1,
        description="List of labels used for classification."
    )

    @field_validator("input")
    @classmethod
    def validate_input_not_empty(cls, v: List[Union[str, Dict[str, str]]]) -> List[Union[str, Dict[str, str]]]:
        """Validates that the input list is not empty."""
        if not v:
            raise ValueError("Classifier input cannot be an empty list.")
        return v

    @field_validator("labels")
    @classmethod
    def validate_labels_not_empty(cls, v: List[str]) -> List[str]:
        """Validates that the labels list is not empty and contains no empty strings."""
        if not v:
            raise ValueError("Classifier labels cannot be empty.")
        # Also validate individual labels are not empty
        for label in v:
            if not label.strip():
                raise ValueError("Classifier labels cannot contain empty strings.")
        return v


# Response type definitions for classifier API
class ClassificationResult(TypedDict):
    input: Union[str, Dict[str, str]]
    labels: Dict[str, float]  # Label name -> confidence score


class ClassifierResponse(TypedDict):
    model: str
    classifier_id: str
    results: List[ClassificationResult]


class ClassifierTool(BaseTool):
    """Tool for calling the Jina Classifier API (zero-shot classification for text or images)."""
    name: str = "ClassifierTool"
    description: str = (
        "Use this tool to perform zero-shot classification with Jina's Classifier API. "
        "Requires the JINA_API_KEY environment variable."
    )

    retry_config: RetryConfig = Field(default_factory=RetryConfig)

    async def _arun(self, request: ClassifierRequest, **kwargs: Any) -> Union[ClassifierResponse, ErrorResponse, RawResponse]:
        """Asynchronously call the Jina Classifier API.

        Args:
            request: The classifier request parameters

        Returns:
            The API response containing classification results or error information
        """
        url = "https://api.jina.ai/v1/classify"
        headers = {
            "Authorization": f"Bearer {_get_jina_api_key()}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        payload = request.model_dump(exclude_none=True)
        resp = await _make_request_with_retry(
            method="POST",
            url=url,
            headers=headers,
            json_data=payload,
            retry_config=self.retry_config,
        )
        return resp

    def run(self, request: ClassifierRequest, **kwargs: Any) -> Union[ClassifierResponse, ErrorResponse, RawResponse]:
        """Synchronous wrapper for calling the Jina Classifier API.

        Args:
            request: The classifier request parameters

        Returns:
            The API response containing classification results or error information
        """
        return asyncio.run(self._arun(request, **kwargs))

