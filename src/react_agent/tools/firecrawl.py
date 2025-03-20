from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast, Awaitable, NoReturn
from typing_extensions import Annotated, Literal, ParamSpec, assert_never

from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg

from react_agent.configuration import Configuration

P = ParamSpec("P")
R = TypeVar("R")

def Web(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """Decorator for web-related functions.
    
    This decorator handles common web operation patterns like:
    - Configuration management
    - Error handling
    - Rate limiting
    - Caching (if implemented)
    
    Args:
        func: The async function to decorate
        
    Returns:
        The decorated async function
    """
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            # Extract config from kwargs
            config = cast(Optional[RunnableConfig], kwargs.get('config'))
            if not config:
                raise ValueError("Configuration is required")
                
            # Get FireCrawl configuration
            configuration = Configuration.from_runnable_config(config)
            if not configuration.firecrawl_api_key:
                raise ValueError("FireCrawl API key not found in configuration")
                
            # Call the original function
            return await func(*args, **kwargs)
            
        except Exception as e:
            print(f"Error in web operation: {str(e)}")
            return cast(R, None)
            
    return wrapper

def create_loader(
    url: str,
    mode: Literal["scrape", "crawl", "map"],
    api_key: Optional[str],
    base_url: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None
) -> FireCrawlLoader:
    """Create a FireCrawl loader with the given configuration.
    
    Args:
        url: The URL to process
        mode: The operation mode
        api_key: FireCrawl API key
        base_url: Optional base URL for self-hosted instances
        params: Optional parameters for the FireCrawl API
        
    Returns:
        Configured FireCrawlLoader instance
        
    Raises:
        ValueError: If api_key is None
    """
    if not api_key:
        raise ValueError("FireCrawl API key is required")
        
    # At this point, we know api_key is not None
    api_key_str: str = api_key
        
    loader_kwargs = {
        "url": url,
        "mode": mode,
        "api_key": api_key_str,
        "params": params or {}
    }
    
    if base_url:
        loader_kwargs["base_url"] = base_url
        
    return FireCrawlLoader(**loader_kwargs)

@Web
async def search(
    query: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[List[Document]]:
    """Search the web for information about a topic.
    
    This function uses FireCrawl's map mode to find semantically related pages
    about the given query.
    
    Args:
        query: The search query
        config: Configuration containing API key and other settings
        
    Returns:
        A list of Document objects containing search results,
        or None if the search failed
    """
    configuration = Configuration.from_runnable_config(config)
    
    # Initialize the loader in map mode
    loader = create_loader(
        url=query,  # Using query as the starting point
        mode="map",
        api_key=configuration.firecrawl_api_key,
        base_url=configuration.firecrawl_url,
        params={
            "max_results": configuration.max_search_results
        }
    )
    
    # Load and return the documents
    documents = loader.load()
    return cast(List[Document], documents)

@Web
async def scrape(
    url: str,
    mode: Literal["scrape", "crawl"] = "scrape",
    params: Optional[Dict[str, Any]] = None,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[List[Document]]:
    """Scrape content from a URL using FireCrawl.
    
    This function supports both single-page scraping and recursive crawling.
    
    Args:
        url: The URL to scrape/crawl
        mode: Whether to scrape a single page or crawl recursively
        params: Optional parameters to pass to FireCrawl API
        config: Configuration containing API key and other settings
        
    Returns:
        A list of Document objects containing the scraped content,
        or None if scraping failed
    """
    configuration = Configuration.from_runnable_config(config)
    
    # Initialize the loader
    loader = create_loader(
        url=url,
        mode=mode,
        api_key=configuration.firecrawl_api_key,
        base_url=configuration.firecrawl_url,
        params=params or {}
    )
    
    # Load and return the documents
    documents = loader.load()
    return cast(List[Document], documents)

@Web
async def map_url(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[List[Document]]:
    """Map a URL to find semantically related pages.
    
    This function uses FireCrawl's map mode to discover pages related to the given URL.
    
    Args:
        url: The URL to map
        params: Optional parameters to pass to FireCrawl API
        config: Configuration containing API key and other settings
        
    Returns:
        A list of Document objects containing related pages,
        or None if mapping failed
    """
    configuration = Configuration.from_runnable_config(config)
    
    # Initialize the loader in map mode
    loader = create_loader(
        url=url,
        mode="map",
        api_key=configuration.firecrawl_api_key,
        base_url=configuration.firecrawl_url,
        params=params or {}
    )
    
    # Load and return the documents
    documents = loader.load()
    return cast(List[Document], documents)

# Export the tools
TOOLS: List[Callable[..., Any]] = [search, scrape, map_url]