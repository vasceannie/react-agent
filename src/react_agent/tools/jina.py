"""Enhanced Jina AI Search Integration.

This module provides a more robust integration with Jina AI's search API
with improved error handling, result validation, and search strategies.
"""


from typing import Dict, List, Optional, Any, Union, cast, Tuple, Literal
import json
import time
import aiohttp
import asyncio
import contextlib
from urllib.parse import urljoin, quote, urlparse
import random
from datetime import datetime, timezone
import re
import os

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_core.tools import ToolException
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

from react_agent.configuration import Configuration
from react_agent.utils.logging import get_logger, log_dict, info_highlight, warning_highlight, error_highlight
from react_agent.utils.validations import is_valid_url
from react_agent.prompts.query import optimize_query, detect_vertical, expand_acronyms
from react_agent.utils.extraction import safe_json_parse
from react_agent.utils.cache import create_checkpoint, load_checkpoint, cache_result, ProcessorCache

# Initialize logger
logger = get_logger(__name__)

# Define search types for specialized search strategies
SearchType = Literal["general", "authoritative", "recent", "comprehensive", "technical"]

# Initialize memory saver for caching
processor_cache = ProcessorCache(thread_id="jina-search")

# Add at module level after imports
_query_cache: Dict[str, Tuple[List[Document], datetime]] = {}

class RetryConfig(BaseModel):
    """Configuration for retry behavior."""
    max_retries: int = Field(default=3, description="Maximum number of retries")
    base_delay: float = Field(default=1.0, description="Base delay between retries in seconds")
    max_delay: float = Field(default=10.0, description="Maximum delay between retries in seconds")
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff."""
        return min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))

class SearchParams(BaseModel):
    """Parameters for search operations."""
    query: str = Field(..., description="Search query")
    search_type: SearchType = Field(default="general", description="Type of search to perform")
    max_results: Optional[int] = Field(default=None, description="Maximum number of results to return")
    min_quality_score: Optional[float] = Field(default=0.5, description="Minimum quality score for results")
    recency_days: Optional[int] = Field(default=None, description="Maximum age of results in days")
    domains: Optional[List[str]] = Field(default=None, description="List of domains to search")
    category: Optional[str] = Field(default=None, description="Category to search in")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        result = {
            "q": self.query,
            "limit": self.max_results or 10,
            "min_score": self.min_quality_score or 0.5
        }
        
        if self.recency_days:
            result["recency_days"] = self.recency_days
            
        if self.domains:
            result["domains"] = ",".join(self.domains)
            
        if self.category:
            result["category"] = self.category
            
        return result

class JinaSearchClient:
    """Enhanced client for Jina AI search with retry and validation."""
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """Initialize Jina search client.
        
        Args:
            api_key: Jina AI API key
            base_url: Optional base URL for self-hosted instances
            retry_config: Configuration for retry behavior
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/') if base_url else "https://s.jina.ai"
        self.retry_config = retry_config or RetryConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Create aiohttp session."""
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

    def _get_endpoint(self, endpoint: str) -> str:
        """Get endpoint URL."""
        base = self.base_url.rstrip('/')
        if not endpoint.startswith('/'):
            endpoint = f'/{endpoint}'
        return f"{base}{endpoint}"

    async def _make_request_with_retry(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to Jina API with retry logic and better error handling."""
        if not self.session:
            raise ValueError("Session not initialized")

        last_exception = None
        for attempt in range(1, self.retry_config.max_retries + 1):
            try:
                url = self._get_endpoint(endpoint)
                info_highlight(f"Making request to: {url} with params: {params}")
                
                async with self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data
                ) as response:
                    # Check for no results error (422)
                    if response.status == 422:
                        error_text = await response.text()
                        if "No search results available" in error_text:
                            warning_highlight("No search results available for this query")
                            return {"results": []}  # Return empty results rather than raising an error
                    
                    # Handle other errors
                    if response.status != 200:
                        error_text = await response.text()
                        error_highlight(f"Request failed with status {response.status}: {error_text}")
                        raise aiohttp.ClientError(f"Request failed with status {response.status}: {error_text}")
                    
                    return await response.json()
            except Exception as e:
                last_exception = e
                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.get_delay(attempt)
                    warning_highlight(
                        f"Request failed (attempt {attempt}/{self.retry_config.max_retries}): {str(last_exception)}. Retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    error_highlight(
                        f"Request failed after {self.retry_config.max_retries} attempts: {str(last_exception)}"
                    )
                    raise

        # This should never be reached as the last failure should raise
        assert last_exception is not None
        raise last_exception

    async def search(
        self,
        params: SearchParams
    ) -> List[Document]:
        """Search with improved parameters and result validation.
        
        Args:
            params: Search parameters
            
        Returns:
            List of documents from search results
        """
        request_params = params.to_dict()
        info_highlight(f"Executing {params.search_type} search with query: {params.query}")
        
        try:
            # Use the search endpoint
            data = await self._make_request_with_retry(
                method="GET",
                endpoint="/search",  # Use the search endpoint
                params=request_params
            )
            
            results = self._parse_search_results(data)
            info_highlight(f"Retrieved {len(results)} search results")
            
            # If no results, try simplified query
            if not results:
                # Extract main keywords (first 3 words or up to 30 chars)
                simplified_query = " ".join(params.query.split()[:3])
                if len(simplified_query) > 30:
                    simplified_query = simplified_query[:30]
                
                info_highlight(f"No results found, trying simplified query: {simplified_query}")
                request_params["query"] = simplified_query
                
                data = await self._make_request_with_retry(
                    method="GET",
                    endpoint="/search",
                    params=request_params
                )
                
                results = self._parse_search_results(data)
                info_highlight(f"Retrieved {len(results)} results with simplified query")
            
            # Convert results to Documents
            documents = self._convert_to_documents(results)
            
            # Apply quality filtering
            min_score: float = float(params.min_quality_score or 0.5)  # Explicit type conversion
            filtered_docs = self._filter_documents(documents, min_score)
            info_highlight(f"Filtered to {len(filtered_docs)} high-quality results")
            
            return filtered_docs
        except Exception as e:
            error_highlight(f"Search failed: {str(e)}")
            return []

    def _parse_search_results(self, raw_results: Union[str, List[Dict], Dict]) -> List[Dict]:
        """Parse raw results with improved error handling."""
        try:
            # Convert raw_results to string if it's not already
            if isinstance(raw_results, (list, dict)):
                raw_results = json.dumps(raw_results)
            parsed = safe_json_parse(raw_results, "search_results")
            return self._extract_results_list(parsed)
        except Exception as e:
            error_highlight(f"Error parsing search results: {str(e)}")
            return []

    def _extract_results_list(self, data: Union[List[Dict], Dict]) -> List[Dict]:
        """Extract results list from various response formats."""
        try:
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Try common response formats
                for key in ['results', 'data', 'items', 'hits', 'matches', 'documents', 'response']:
                    if key in data:
                        if isinstance(data[key], list):
                            return data[key]
                        elif isinstance(data[key], dict):
                            # Try to extract from nested structure
                            for nested_key in ['results', 'data', 'items', 'hits', 'matches']:
                                if nested_key in data[key] and isinstance(data[key][nested_key], list):
                                    return data[key][nested_key]

                # If no list found, try to extract single result
                if 'result' in data:
                    return [data['result']]

                # If still no results, try to extract from nested structure
                for value in data.values():
                    if isinstance(value, list):
                        return value
                    elif isinstance(value, dict):
                        if nested_results := self._extract_results_list(value):
                            return nested_results

                # If still no results, return empty list
                return []
            return []
        except Exception as e:
            error_highlight(f"Error extracting results list: {str(e)}")
            return []

    def _extract_content(self, result: Dict) -> Optional[str]:
        """Extract content from result with fallbacks."""
        content_fields = ['snippet', 'content', 'text', 'description', 'summary', 'body', 'raw']
        for field in content_fields:
            if content := result.get(field):
                return content.strip().strip('```json').strip('```')
        return None

    def _build_metadata(self, result: Dict, content: str) -> Dict:
        """Build metadata dictionary from result."""
        field_mapping = {
            'url': ['url', 'link', 'href', 'source_url', 'web_url'],
            'title': ['title', 'name', 'heading', 'subject', 'headline'],
            'source': ['source', 'domain', 'site', 'provider', 'publisher'],
            'published_date': ['published_date', 'date', 'timestamp', 'published', 'created_at', 'publication_date']
        }
        
        metadata = {
            'quality_score': self._calculate_quality_score(result, content),
            'extraction_status': 'success',
            'extraction_timestamp': datetime.now().isoformat(),
            'original_result': result
        }
        
        for field_type, fields in field_mapping.items():
            for field in fields:
                if value := result.get(field):
                    metadata[field_type] = value
                    break
        
        if url := metadata.get('url'):
            try:
                metadata['domain'] = urlparse(url).netloc
            except Exception:
                metadata['domain'] = ""
                
        return metadata

    def _convert_to_documents(self, results: List[Dict]) -> List[Document]:
        """Convert search results to Document objects with improved metadata."""
        documents = []
        for idx, result in enumerate(results, 1):
            if not isinstance(result, dict):
                continue
                
            if not (content := self._extract_content(result)):
                warning_highlight(f"No content found for result {idx}")
                continue
                
            try:
                metadata = self._build_metadata(result, content)
                documents.append(Document(page_content=content, metadata=metadata))
                info_highlight(f"Successfully converted result {idx} to Document")
            except Exception as e:
                warning_highlight(f"Error converting result {idx} to Document: {str(e)}")
                continue
                
        return documents

    @cache_result(ttl=3600)
    def _calculate_quality_score(self, result: Dict[str, Any], content: str) -> float:
        """Calculate quality score for a search result."""
        score = 0.5  # Base score

        # Add points for authoritative domains
        authoritative_domains = [
            '.gov', '.edu', '.org', 'wikipedia.org', 
            'research', 'journal', 'university', 'association'
        ]
        url = result.get('url', '')
        if any(domain in url.lower() for domain in authoritative_domains):
            score += 0.2

        # Add points for content length (substantive content)
        if len(content) > 500:
            score += 0.1

        # Add points for having title/publication date
        if result.get('title'):
            score += 0.05
        if result.get('published_date') or result.get('date'):
            score += 0.05

        if date_field := result.get('published_date', result.get('date', '')):
            with contextlib.suppress(Exception):
                # Try to parse date
                from dateutil import parser
                published_date = parser.parse(date_field)
                current_date = datetime.now(timezone.utc)
                days_old = (current_date - published_date).days

                # Fresher content gets higher score
                if days_old < 30:  # Last month
                    score += 0.1
                elif days_old < 180:  # Last 6 months
                    score += 0.05
        return min(1.0, score)  # Cap at 1.0

    @cache_result(ttl=3600)
    def _filter_documents(self, documents: List[Document], min_score: float) -> List[Document]:
        """Filter documents based on quality score."""
        return [
            doc for doc in documents 
            if doc.metadata.get('quality_score', 0) >= min_score
        ]

class JinaSearchTool(BaseTool):
    """Enhanced Jina AI search integration with caching and parallel processing."""
    
    name: str = "jina_search"
    description: str = "Search for information using Jina AI's search engine with enhanced caching and parallel processing"
    
    config: Configuration = Field(default_factory=Configuration)
    
    def __init__(self, config: Optional[Configuration] = None):
        """Initialize the Jina search tool.
        
        Args:
            config: Optional configuration for the search tool
        """
        super().__init__()
        if config:
            self.config = config
        if not self.config.jina_api_key:
            raise ValueError("Jina API key is required")
            
    async def _arun(
        self,
        query: str,
        search_type: Optional[SearchType] = None,
        max_results: Optional[int] = None,
        min_quality_score: Optional[float] = None,
        recency_days: Optional[int] = None,
        domains: Optional[List[str]] = None,
        category: Optional[str] = None
    ) -> List[Document]:
        """Execute search with caching and parallel processing."""
        config = RunnableConfig(configurable={"jina_api_key": self.config.jina_api_key})
        return await search(
            query=query,
            search_type=search_type,
            max_results=max_results,
            min_quality=min_quality_score,
            recency_days=recency_days,
            domains=domains,
            category=category,
            config=config
        )

async def _execute_search_strategy(
    client: JinaSearchClient,
    params: SearchParams,
    category: Optional[str] = None
) -> List[List[Document]]:
    """Execute search with optional category-specific search."""
    search_tasks = [client.search(params)]

    if not category:
        if vertical := detect_vertical(params.query):
            category_params = SearchParams(**params.model_dump())
            category_params.category = vertical
            search_tasks.append(client.search(category_params))

    return await asyncio.gather(*search_tasks)

@cache_result(ttl=3600)
def _merge_and_filter_results(
    results_list: List[List[Document]],
    min_quality: float
) -> List[Document]:
    """Merge, deduplicate and filter search results."""
    seen_urls = set()
    all_results = []
    
    for results in results_list:
        for doc in results:
            url = doc.metadata.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                all_results.append(doc)
    
    all_results.sort(key=lambda x: x.metadata.get("quality_score", 0), reverse=True)
    return [doc for doc in all_results if doc.metadata.get("quality_score", 0) >= min_quality]

async def _load_cached_results(cache_key: str) -> Optional[List[Document]]:
    """Load and validate cached search results."""
    try:
        if cached := load_checkpoint(cache_key):
            if not isinstance(cached, dict) or "results" not in cached or "timestamp" not in cached:
                return None
                
            if (datetime.now() - datetime.fromisoformat(cached["timestamp"])).total_seconds() >= 86400:
                return None
                
            results = cached["results"]
            if not isinstance(results, list) or not all(isinstance(doc, Document) for doc in results):
                return None
                
            info_highlight(f"Retrieved {len(results)} results from cache")
            return results
    except Exception as e:
        warning_highlight(f"Error loading from cache: {str(e)}")
    return None

async def _perform_search(
    configuration: Configuration,
    params: SearchParams,
    cache_key: str
) -> List[Document]:
    """Execute search and cache results."""
    async with JinaSearchClient(
        api_key=configuration.jina_api_key or "",
        base_url=configuration.jina_url,
        retry_config=RetryConfig()
    ) as client:
        results_list = await _execute_search_strategy(client, params, params.category)
        all_results = _merge_and_filter_results(results_list, params.min_quality_score or 0.5)
        
        try:
            create_checkpoint(
                cache_key,
                {
                    "results": all_results,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "params": params.to_dict()
                },
                ttl=86400
            )
            info_highlight(f"Cached {len(all_results)} results")
        except Exception as e:
            warning_highlight(f"Error caching results: {str(e)}")
        
        return all_results

async def search(
    query: str,
    search_type: Optional[SearchType] = None,
    max_results: Optional[int] = None,
    min_quality: Optional[float] = None,
    recency_days: Optional[int] = None,
    domains: Optional[List[str]] = None,
    category: Optional[str] = None,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> List[Document]:
    """Enhanced search with multiple strategies, quality filters, and improved caching."""
    configuration = Configuration.from_runnable_config(config)
    if not configuration.jina_api_key:
        error_highlight("Jina API key is required")
        return []

    os.environ["JINA_API_KEY"] = configuration.jina_api_key
    if configuration.jina_url:
        os.environ["JINA_URL"] = configuration.jina_url

    params = SearchParams(
        query=query,
        search_type=search_type or "general",
        max_results=max_results or configuration.max_search_results,
        min_quality_score=min_quality or 0.5,
        recency_days=recency_days,
        domains=domains or (['.edu', '.gov', '.org'] if search_type in ["authoritative", None] else None),
        category=category
    )

    cache_key = f"jina_search_{params.query}_{params.search_type}_{params.max_results}"
    
    if cached_results := await _load_cached_results(cache_key):
        return cached_results

    try:
        return await _perform_search(configuration, params, cache_key)
    except Exception as e:
        error_highlight(f"Error in Jina search: {str(e)}")
        return []

# Export available tools
TOOLS = [search]