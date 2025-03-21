"""Jina AI Search Provider Implementation using LangChain toolkit.

This module implements a search tool that interfaces with Jina AI's search API
using the LangChain integration.
"""

from typing import Dict, List, Optional

from langchain_community.tools import JinaSearch
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated

from react_agent.configuration import Configuration

class JinaSearchRunnable(Runnable):
    """A runnable that performs Jina AI search."""
    
    def __init__(self) -> None:
        """Initialize the Jina search runnable."""
        super().__init__()
        self._tool: Optional[JinaSearch] = None
    
    def _get_tool(self, config: RunnableConfig) -> JinaSearch:
        """Get or create the Jina search tool with configuration."""
        if not self._tool:
            configuration = Configuration.from_runnable_config(config)
            if not configuration.jina_api_key:
                raise ValueError("Jina API key is required")
                
            # Set up environment for Jina
            import os
            os.environ["JINA_API_KEY"] = configuration.jina_api_key
            if configuration.jina_url:
                os.environ["JINA_URL"] = configuration.jina_url
                
            self._tool = JinaSearch()
        return self._tool

    def invoke(
        self,
        input: Dict[str, str],
        config: Optional[RunnableConfig] = None,
    ) -> List[Document]:
        """Synchronously invoke the Jina search.
        
        Args:
            input: Dictionary containing the query
            config: Configuration for the search
            
        Returns:
            List of Document objects containing search results
        """
        query = input.get("query", "")
        if not query:
            return []

        print(f"Searching with query: {query}")
        tool = self._get_tool(config or {})

        try:
            results = tool.invoke({"query": query})
            print(f"Found {len(results)} results")

            # Convert results to Documents
            documents = []
            documents.extend(
                Document(
                    page_content=result.get("snippet", ""),
                    metadata={
                        "title": result.get("title", ""),
                        "url": result.get("link", ""),  # Using link as per docs
                        "source": result.get("source", ""),
                        "published_date": result.get("published_date"),
                    },
                )
                for result in results
                if isinstance(result, dict)
            )
            return documents

        except Exception as e:
            print(f"Search failed: {str(e)}")
            return []

    async def ainvoke(
        self,
        input: Dict[str, str],
        config: Optional[RunnableConfig] = None,
    ) -> List[Document]:
        """Asynchronously invoke the Jina search.
        
        Args:
            input: Dictionary containing the query
            config: Configuration for the search
            
        Returns:
            List of Document objects containing search results
        """
        return self.invoke(input, config)

async def search(
    query: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[List[Document]]:
    """Search the web using Jina AI's search API via LangChain.
    
    Args:
        query: The search query string
        config: Configuration containing API key and settings
        
    Returns:
        Optional[List[Document]]: List of search results as Documents, or None if search fails
    """
    runnable = JinaSearchRunnable()
    return await runnable.ainvoke({"query": query}, config)

# Export available tools
TOOLS = [search]
