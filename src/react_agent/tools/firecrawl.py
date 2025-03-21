from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    NoReturn,
    Optional,
    TypeVar,
    Union,
    cast,
)
import time
import aiohttp
from urllib.parse import urljoin

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated, Literal, ParamSpec, assert_never

from react_agent.configuration import Configuration

P = ParamSpec("P")
R = TypeVar("R")

# All available modes for FireCrawl operations
FirecrawlMode = Literal["scrape", "crawl", "map", "search", "extract"]

class FireCrawlClient:
    """Client for interacting with FireCrawl API."""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """Initialize FireCrawl client.
        
        Args:
            api_key: FireCrawl API key
            base_url: Optional base URL for self-hosted instances
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/') if base_url else "https://api.firecrawl.com"
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Create aiohttp session."""
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

    def _get_endpoint(self, mode: FirecrawlMode) -> str:
        """Get endpoint URL for given mode."""
        endpoints = {
            "search": "/search",
            "scrape": "/scrape",
            "crawl": "/crawl",
            "map": "/map",
            "extract": "/extract"
        }
        return urljoin(self.base_url, endpoints[mode])

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to FireCrawl API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            json: JSON body
            
        Returns:
            API response as dictionary
            
        Raises:
            ValueError: If session is not initialized
            aiohttp.ClientError: If request fails
        """
        if not self.session:
            raise ValueError("Session not initialized")
            
        async with self.session.request(
            method=method,
            url=endpoint,
            params=params,
            json=json
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def search(
        self,
        query: str,
        limit: Optional[int] = None
    ) -> List[Document]:
        """Search for documents matching query."""
        params = {"q": query}
        if limit:
            params["limit"] = str(limit)
            
        data = await self._make_request(
            method="GET",
            endpoint=self._get_endpoint("search"),
            params=params
        )
        
        return [
            Document(
                page_content=result["content"],
                metadata=result.get("metadata", {})
            )
            for result in data.get("results", [])
        ]

    async def scrape(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Scrape content from URL."""
        data = await self._make_request(
            method="POST",
            endpoint=self._get_endpoint("scrape"),
            json={"url": url, **(params or {})}
        )
        
        return [
            Document(
                page_content=result["content"],
                metadata=result.get("metadata", {})
            )
            for result in data.get("results", [])
        ]

    async def crawl(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start crawl job and return job ID."""
        data = await self._make_request(
            method="POST",
            endpoint=self._get_endpoint("crawl"),
            json={
                "url": url,
                "onlyMainContent": True,
                **(params or {})
            }
        )
        return data["id"]

    async def check_job(self, job_id: str) -> Dict[str, Any]:
        """Check status of crawl job."""
        return await self._make_request(
            method="GET",
            endpoint=urljoin(self.base_url, f"/crawl/{job_id}")
        )

    async def map_url(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Map URL to find related pages."""
        data = await self._make_request(
            method="POST",
            endpoint=self._get_endpoint("map"),
            json={"url": url, **(params or {})}
        )
        
        return [
            Document(
                page_content=result["content"],
                metadata=result.get("metadata", {})
            )
            for result in data.get("results", [])
        ]

    async def extract(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Extract structured data from URL."""
        data = await self._make_request(
            method="POST",
            endpoint=self._get_endpoint("extract"),
            json={"url": url, **(params or {})}
        )
        
        return [
            Document(
                page_content=result["content"],
                metadata=result.get("metadata", {})
            )
            for result in data.get("results", [])
        ]

def Web(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """Decorate web-related functions."""
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            config = cast(Optional[RunnableConfig], kwargs.get('config'))
            if not config:
                raise ValueError("Configuration is required")
                
            configuration = Configuration.from_runnable_config(config)
            if not configuration.firecrawl_api_key:
                raise ValueError("FireCrawl API key not found in configuration")
                
            return await func(*args, **kwargs)
            
        except Exception as e:
            print(f"Error in web operation: {str(e)}")
            return cast(R, None)
            
    return wrapper

@Web
async def search(
    query: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[List[Document]]:
    """Search the web for information about a topic."""
    configuration = Configuration.from_runnable_config(config)
    print(f"Searching with query: {query}")
    
    try:
        async with FireCrawlClient(
            api_key=cast(str, configuration.firecrawl_api_key),
            base_url=configuration.firecrawl_url
        ) as client:
            documents = await client.search(
                query=query,
                limit=configuration.max_search_results
            )
            print(f"Found {len(documents)} results")
            return documents
    except Exception as e:
        print(f"Search failed: {str(e)}")
        return None

@Web
async def scrape(
    url: str,
    mode: Literal["scrape", "crawl"] = "scrape",
    params: Optional[Dict[str, Any]] = None,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[List[Document]]:
    """Scrape content from a URL using FireCrawl."""
    configuration = Configuration.from_runnable_config(config)
    
    try:
        async with FireCrawlClient(
            api_key=cast(str, configuration.firecrawl_api_key),
            base_url=configuration.firecrawl_url
        ) as client:
            if mode == "crawl":
                # Start crawl job
                job_id = await client.crawl(url=url, params=params)
                
                # Poll for completion
                for _ in range(10):  # Max 10 retries
                    status = await client.check_job(job_id)
                    if status.get("status") == "completed":
                        return [
                            Document(
                                page_content=doc["content"],
                                metadata=doc.get("metadata", {})
                            )
                            for doc in status.get("results", [])
                        ]
                    time.sleep(30)  # Wait 30 seconds between checks
                return None
            else:
                return await client.scrape(url=url, params=params)
    except Exception as e:
        print(f"Operation failed: {str(e)}")
        return None

@Web
async def map_url(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[List[Document]]:
    """Map a URL to find related pages."""
    configuration = Configuration.from_runnable_config(config)
    
    try:
        async with FireCrawlClient(
            api_key=cast(str, configuration.firecrawl_api_key),
            base_url=configuration.firecrawl_url
        ) as client:
            return await client.map_url(url=url, params=params)
    except Exception as e:
        print(f"Map operation failed: {str(e)}")
        return None

@Web
async def extract(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[List[Document]]:
    """Extract structured data from a URL using AI."""
    configuration = Configuration.from_runnable_config(config)
    
    try:
        async with FireCrawlClient(
            api_key=cast(str, configuration.firecrawl_api_key),
            base_url=configuration.firecrawl_url
        ) as client:
            return await client.extract(url=url, params=params)
    except Exception as e:
        print(f"Extract operation failed: {str(e)}")
        return None

# Export all available tools
TOOLS: List[Callable[..., Any]] = [search, scrape, map_url, extract]