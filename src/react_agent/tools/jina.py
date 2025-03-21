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
from urllib.parse import urljoin, quote, urlparse
import random
from datetime import datetime
import re

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated, Literal

from react_agent.configuration import Configuration
from react_agent.utils.logging import get_logger, log_dict, info_highlight, warning_highlight, error_highlight
from react_agent.utils.validations import is_valid_url
from react_agent.prompts.query import optimize_query, detect_vertical, expand_acronyms

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
        category: Optional[str] = None,
    ):
        self.query = query
        self.search_type = search_type
        self.max_results = max_results
        self.min_quality_score = min_quality_score
        self.recency_days = recency_days
        self.domains = domains or []
        self.exclude_domains = exclude_domains or []
        self.category = category

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        # Clean and encode the query
        cleaned_query = self.query.replace('\n', ' ').replace('\r', ' ').strip()
        
        # Remove any "Additional context" or similar prefixes
        cleaned_query = cleaned_query.split("Additional context:")[0].strip()
        
        # Break down long queries into keyword-focused search
        if len(cleaned_query.split()) > 10:
            # Extract key terms using basic keyword extraction
            # We're looking for nouns and specific terms, avoiding basic verbs and help phrases
            skip_words = ["help", "me", "research", "find", "information", "about", "on", "for", 
                         "the", "and", "or", "in", "to", "with", "by", "is", "are", "was", "were"]
            
            # Extract meaningful terms for search
            terms = [word for word in cleaned_query.split() 
                    if word.lower() not in skip_words and len(word) > 3]
            
            # If we found meaningful terms, use those; otherwise fall back to truncated original
            if len(terms) >= 3:
                cleaned_query = " ".join(terms[:7])  # Use up to 7 meaningful keywords
            else:
                # Just take the first few meaningful words if we couldn't extract good keywords
                words = cleaned_query.split()
                cleaned_query = " ".join(words[:5])
        
        # Fix common typos
        cleaned_query = cleaned_query.replace("resaerch", "research")
        
        # Remove duplicate terms and phrases
        # Remove duplicate phrases (e.g., "maintenance repair operations" repeated)
        cleaned_query = re.sub(r'(\b\w+\s+\w+\s+\w+\b)(?:\s+\1)+', r'\1', cleaned_query)
        
        # Remove duplicate single words
        words = cleaned_query.split()
        cleaned_query = ' '.join(dict.fromkeys(words))
        
        # Remove redundant parentheses and their contents
        cleaned_query = re.sub(r'\([^)]*\)', '', cleaned_query)
        
        # Optimize the query using query.py functions
        if self.category:
            # Detect vertical from the query
            vertical = detect_vertical(cleaned_query)
            # Expand acronyms
            expanded_query = expand_acronyms(cleaned_query)
            # Optimize query for the category
            optimized_query = optimize_query(
                original_query=expanded_query,
                category=self.category,
                vertical=vertical,
                include_all_keywords=self.search_type == "comprehensive"
            )
        else:
            # If no category, just expand acronyms
            optimized_query = expand_acronyms(cleaned_query)
        
        # Limit query length to avoid issues
        optimized_query = optimized_query[:100] if len(optimized_query) > 100 else optimized_query
        
        # Clean up any remaining special characters and extra spaces
        optimized_query = re.sub(r'[^\w\s-]', ' ', optimized_query)
        optimized_query = re.sub(r'\s+', ' ', optimized_query).strip()
        
        # Remove any remaining duplicate words
        words = optimized_query.split()
        optimized_query = ' '.join(dict.fromkeys(words))
        
        # Further simplify if the query is still too complex
        if len(optimized_query.split()) > 7:
            optimized_query = ' '.join(optimized_query.split()[:7])
        
        info_highlight(f"Original query: {self.query}")
        info_highlight(f"Optimized query for search: {optimized_query}")
        
        result = {
            "q": quote(optimized_query),
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
    category: Optional[str] = None,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[List[Document]]:
    """Enhanced search with multiple strategies and quality filters."""
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
        domains=domains,
        category=category
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
            
            # First attempt with normal query
            results = await client.search(params)

            # If no results, try with simplified query
            if not results:
                info_highlight("No results with initial query, trying with simplified query")

                # Create simplified query by keeping only key terms
                words = query.split()
                if simplified_query := " ".join(
                    [
                        w
                        for w in words
                        if len(w) > 3
                        and w.lower()
                        not in [
                            "help",
                            "find",
                            "about",
                            "information",
                            "research",
                            "please",
                        ]
                    ][:5]
                ):
                    params.query = simplified_query
                    results = await client.search(params)

            # If still no results, try with just the most important keyword
            if not results and len(words) > 1:
                info_highlight("Still no results, trying with most important keyword")
                # Find longest word as it's likely most important
                important_term = max(words, key=len)
                if len(important_term) > 3:
                    params.query = important_term
                    results = await client.search(params)

            return results

    except Exception as e:
        error_highlight(f"Search failed: {str(e)}")
        return None

# Export available tools
TOOLS = [search]