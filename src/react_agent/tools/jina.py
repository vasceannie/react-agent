"""Enhanced Jina AI Search Integration.

This module provides a more robust integration with Jina AI's search API
with improved error handling, result validation, and search strategies.
"""


from typing import Dict, List, Optional, Any, Union, cast
import json
import time
import aiohttp
import asyncio
import contextlib
from urllib.parse import urljoin, quote
import random
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated, Literal

from react_agent.configuration import Configuration
from react_agent.utils.logging import get_logger, log_dict, info_highlight, warning_highlight, error_highlight
from react_agent.utils.validations import is_valid_url

# Initialize logger
logger = get_logger(__name__)

# Define search types for specialized search strategies
SearchType = Literal["general", "authoritative", "recent", "comprehensive", "technical"]

class RetryConfig:
    """Configuration for retry behavior."""
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and optional jitter."""
        delay = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        if self.jitter:
            delay = delay * (0.5 + random.random())
        return delay

class SearchParams:
    """Parameters for search operations."""
    def __init__(
        self,
        query: str,
        search_type: SearchType = "general",
        max_results: int = 10,
        min_quality_score: float = 0.5,
        recency_days: Optional[int] = None,
        domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ):
        self.query = query
        self.search_type = search_type
        self.max_results = max_results
        self.min_quality_score = min_quality_score
        self.recency_days = recency_days
        self.domains = domains or []
        self.exclude_domains = exclude_domains or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        # Clean and encode the query
        cleaned_query = self.query.replace('\n', ' ').replace('\r', ' ').strip()
        # Remove any "Additional context" or similar prefixes
        cleaned_query = cleaned_query.split("Additional context:")[0].strip()
        # Limit query length to avoid issues
        cleaned_query = cleaned_query[:200] if len(cleaned_query) > 200 else cleaned_query
        
        result = {
            "q": quote(cleaned_query),
            "limit": self.max_results
        }

        # Add search type specific parameters
        if self.search_type == "authoritative":
            # Increase authority weight
            result["authority_boost"] = 2.0
            if self.domains:
                result["domains"] = ",".join(self.domains)

        elif self.search_type == "comprehensive":
            # Increase diversity
            result["diversity"] = 2.0

        elif self.search_type == "recent":
            # Add recency filter
            result["recency_days"] = self.recency_days or 30
        elif self.search_type == "technical":
            # Focus on technical content
            result["content_type"] = "technical"

        # Add domain filters if not already added
        if self.domains and "domains" not in result:
            result["domains"] = ",".join(self.domains)

        if self.exclude_domains:
            result["exclude_domains"] = ",".join(self.exclude_domains)

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
        """Make HTTP request to Jina API with retry logic."""
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
            
            # Convert results to Documents
            documents = self._convert_to_documents(results)
            
            # Apply quality filtering
            filtered_docs = self._filter_documents(documents, params.min_quality_score)
            info_highlight(f"Filtered to {len(filtered_docs)} high-quality results")
            
            return filtered_docs
        except Exception as e:
            error_highlight(f"Search failed: {str(e)}")
            return []

    def _parse_search_results(self, raw_results: Union[str, List[Dict], Dict]) -> List[Dict]:
        """Parse raw results with improved error handling."""
        try:
            if isinstance(raw_results, str):
                try:
                    # Try to parse as JSON first
                    parsed = json.loads(raw_results)
                except json.JSONDecodeError:
                    # If not valid JSON, try to clean and parse again
                    cleaned = raw_results.strip()
                    if cleaned.startswith('```json'):
                        cleaned = cleaned[7:]
                    if cleaned.endswith('```'):
                        cleaned = cleaned[:-3]
                    try:
                        parsed = json.loads(cleaned)
                    except json.JSONDecodeError as e:
                        error_highlight(f"Failed to parse JSON results: {str(e)}")
                        return []
                return self._extract_results_list(parsed)
            elif isinstance(raw_results, (list, dict)):
                return self._extract_results_list(raw_results)
            else:
                warning_highlight(f"Unexpected result type: {type(raw_results)}")
                return []
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
                        # Recursively try to extract from nested dictionaries
                        nested_results = self._extract_results_list(value)
                        if nested_results:
                            return nested_results
                
                # If still no results, return empty list
                return []
            return []
        except Exception as e:
            error_highlight(f"Error extracting results list: {str(e)}")
            return []

    def _convert_to_documents(self, results: List[Dict]) -> List[Document]:
        """Convert search results to Document objects with improved metadata."""
        documents = []
        
        for idx, result in enumerate(results):
            try:
                if not isinstance(result, dict):
                    continue
                    
                # Extract content with fallbacks
                content = None
                for field in ['snippet', 'content', 'text', 'description', 'summary', 'body', 'raw']:
                    if field in result and result[field]:
                        content = result[field]
                        if content:
                            info_highlight(f"Found content in field '{field}' for result {idx + 1}")
                            break
                
                if not content:
                    warning_highlight(f"No content found for result {idx + 1}")
                    continue
                    
                # Build metadata with comprehensive information
                metadata = {}
                
                # Process field types with logging
                field_mapping = {
                    'url': ['url', 'link', 'href', 'source_url', 'web_url'],
                    'title': ['title', 'name', 'heading', 'subject', 'headline'],
                    'source': ['source', 'domain', 'site', 'provider', 'publisher'],
                    'published_date': ['published_date', 'date', 'timestamp', 'published', 'created_at', 'publication_date']
                }
                
                for field_type, fields in field_mapping.items():
                    for field in fields:
                        if field in result and result[field]:
                            metadata[field_type] = result[field]
                            info_highlight(f"Found {field_type} in field '{field}' for result {idx + 1}")
                            break
                
                # Add content quality estimate
                metadata['quality_score'] = self._calculate_quality_score(result, content)
                
                # Extract domain from URL for filtering
                if 'url' in metadata and metadata['url']:
                    from urllib.parse import urlparse
                    try:
                        parsed_url = urlparse(metadata['url'])
                        metadata['domain'] = parsed_url.netloc
                    except Exception:
                        metadata['domain'] = ""
                    
                # Add original result for debugging
                metadata['original_result'] = result
                
                # Add extraction metadata
                metadata['extraction_status'] = 'success'
                metadata['extraction_timestamp'] = datetime.now().isoformat()
                
                # Clean content if needed
                if isinstance(content, str):
                    content = content.strip()
                    if content.startswith('```json'):
                        content = content[7:]
                    if content.endswith('```'):
                        content = content[:-3]
                
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)
                info_highlight(f"Successfully converted result {idx + 1} to Document")
            except Exception as e:
                warning_highlight(f"Error converting result {idx + 1} to Document: {str(e)}")
                continue

        return documents

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
                from datetime import datetime, timezone
                published_date = parser.parse(date_field)
                current_date = datetime.now(timezone.utc)
                days_old = (current_date - published_date).days

                # Fresher content gets higher score
                if days_old < 30:  # Last month
                    score += 0.1
                elif days_old < 180:  # Last 6 months
                    score += 0.05
        return min(1.0, score)  # Cap at 1.0

    def _filter_documents(self, documents: List[Document], min_score: float) -> List[Document]:
        """Filter documents based on quality score."""
        return [
            doc for doc in documents 
            if doc.metadata.get('quality_score', 0) >= min_score
        ]

async def search(
    query: str,
    search_type: Optional[SearchType] = None,
    domains: Optional[List[str]] = None,
    recency_days: Optional[int] = None,
    min_quality: Optional[float] = None,
    max_results: Optional[int] = None,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[List[Document]]:
    """Enhanced search with multiple strategies and quality filters.
    
    Args:
        query: The search query string
        search_type: Type of search to perform (general, authoritative, recent, etc.)
        domains: List of domains to include in search
        recency_days: Only include results from the last N days
        min_quality: Minimum quality score for results (0.0-1.0)
        max_results: Maximum number of results to return
        config: Configuration containing API key and settings
        
    Returns:
        Optional[List[Document]]: List of search results as Documents, or None if search fails
    """
    configuration = Configuration.from_runnable_config(config)
    if not configuration.jina_api_key:
        error_highlight("Jina API key is required")
        return None

    # Set environment variables for Jina (used by some libraries)
    import os
    os.environ["JINA_API_KEY"] = configuration.jina_api_key
    if configuration.jina_url:
        os.environ["JINA_URL"] = configuration.jina_url

    # Use provided params or defaults
    params = SearchParams(
        query=query,
        search_type=search_type or "general",
        max_results=max_results or configuration.max_search_results,
        min_quality_score=min_quality or 0.5,
        recency_days=recency_days,
        domains=domains
    )

    # Add default domains for educational/authoritative content if not provided
    if not domains and search_type in ["authoritative", None]:
        params.domains = ['.edu', '.gov', '.org']

    try:
        retry_config = RetryConfig(max_retries=3, base_delay=1.0, max_delay=10.0)
        async with JinaSearchClient(
                    api_key=configuration.jina_api_key,
                    base_url=configuration.jina_url,
                    retry_config=retry_config
                ) as client:
            return await client.search(params)
    except Exception as e:
        error_highlight(f"Search failed: {str(e)}")
        return None

# Helper function to create optimized queries
def optimize_query(original_query: str, category: str, is_higher_ed: bool = True) -> str:
    """Create optimized queries for specific research categories."""
    # Clean the original query
    original_query = original_query.split("Additional context:")[0].strip()
    
    # Define shorter, more focused query templates
    query_templates = {
        "market_dynamics": "{query} market trends analysis",
        "provider_landscape": "{query} vendors suppliers",
        "technical_requirements": "{query} technical specifications",
        "cost_considerations": "{query} pricing cost budget",
        "best_practices": "{query} best practices case studies",
        "regulatory_landscape": "{query} regulations compliance",
        "implementation_factors": "{query} implementation factors"
    }
    
    if category in query_templates:
        template = query_templates[category]
        return template.format(query=original_query)
    else:
        return original_query

# Export available tools
TOOLS = [search]