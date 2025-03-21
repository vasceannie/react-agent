"""Jina AI Search Provider Implementation using LangChain toolkit.

This module implements a search tool that interfaces with Jina AI's search API
using the LangChain integration.
"""

from typing import Dict, List, Optional
import json

from langchain_community.tools import JinaSearch
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated

from react_agent.configuration import Configuration
from react_agent.utils.logging import get_logger, log_dict, info_highlight, warning_highlight, error_highlight

# Initialize logger
logger = get_logger(__name__)

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
                error_highlight("Jina API key is required")
                raise ValueError("Jina API key is required")
                
            # Set up environment for Jina
            import os
            os.environ["JINA_API_KEY"] = configuration.jina_api_key
            if configuration.jina_url:
                os.environ["JINA_URL"] = configuration.jina_url
                
            self._tool = JinaSearch()
            info_highlight("Initialized Jina search tool")
        return self._tool

    # Modification for src/react_agent/tools/jina.py - improve URL validation and error handling

    def _parse_results(self, raw_results: str | List[Dict] | Dict) -> List[Dict]:
        """Parse raw results into a list of dictionaries with improved error handling."""
        try:
            if isinstance(raw_results, str):
                try:
                    parsed = json.loads(raw_results)
                    if isinstance(parsed, list):
                        return parsed
                    elif isinstance(parsed, dict) and "results" in parsed:
                        return parsed["results"]
                    else:
                        warning_highlight("Unexpected JSON structure in results")
                        return []
                except json.JSONDecodeError as e:
                    error_highlight(f"Failed to parse JSON results: {str(e)}")
                    return []
            elif isinstance(raw_results, list):
                return raw_results
            elif isinstance(raw_results, dict) and "results" in raw_results:
                return raw_results["results"]
            else:
                warning_highlight(f"Unexpected result type: {type(raw_results)}")
                return []
        except Exception as e:
            error_highlight(f"Error parsing search results: {str(e)}")
            return []

    def invoke(
        self,
        input: Dict[str, str],
        config: Optional[RunnableConfig] = None,
    ) -> List[Document]:
        """Synchronously invoke the Jina search."""
        query = input.get("query", "")
        if not query:
            warning_highlight("Empty query provided")
            return []

        info_highlight(f"Executing search with query: {query}")
        tool = self._get_tool(config or {})

        try:
            raw_results = tool.invoke({"query": query})
            info_highlight(f"Retrieved {len(raw_results) if raw_results else 0} raw results")
            
            # Log raw results format
            if raw_results:
                log_dict(
                    {
                        "results_type": str(type(raw_results)),
                        "first_result_sample": (
                            raw_results[:500] + "..."
                            if isinstance(raw_results, str)
                            else json.dumps(
                                raw_results[0] if isinstance(raw_results, list) else raw_results,
                                indent=2,
                                ensure_ascii=False
                            )[:500] + "..."
                        )
                    },
                    title="Search Results Format"
                )

            # Parse results into proper format
            results = self._parse_results(raw_results)
            info_highlight(f"Parsed {len(results)} results")

            # Convert results to Documents
            documents = []
            for idx, result in enumerate(results):
                if len(documents) >= 10:
                    info_highlight("Reached maximum of 10 documents")
                    break
                
                if isinstance(result, dict):
                    # Extract content with logging
                    content = None
                    for field in ['snippet', 'content', 'text', 'description']:
                        if field in result:
                            content = result.get(field)
                            if content:
                                info_highlight(f"Found content in field '{field}' for result {idx + 1}")
                                break
                    
                    if not content and 'raw' in result:
                        content = str(result['raw'])
                        info_highlight(f"Using raw content for result {idx + 1}")
                    
                    if content:
                        # Build metadata
                        metadata = {}
                        
                        # Process each field type with logging
                        field_types = {
                            'url': ['url', 'link', 'href'],
                            'title': ['title', 'name', 'heading'],
                            'source': ['source', 'domain', 'site'],
                            'published_date': ['published_date', 'date', 'timestamp']
                        }
                        
                        for field_type, fields in field_types.items():
                            for field in fields:
                                if field in result:
                                    metadata[field_type] = result[field]
                                    info_highlight(f"Found {field_type} in field '{field}' for result {idx + 1}")
                                    break
                        
                        # Store original result for debugging
                        metadata['original_result'] = result
                        
                        doc = Document(
                            page_content=content,
                            metadata=metadata
                        )
                        documents.append(doc)
                        info_highlight(f"Successfully converted result {idx + 1} to Document")

            info_highlight(f"Successfully converted {len(documents)} results to Documents")
            return documents

        except Exception as e:
            error_highlight(f"Search failed: {str(e)}")
            return []

    async def ainvoke(
        self,
        input: Dict[str, str],
        config: Optional[RunnableConfig] = None,
    ) -> List[Document]:
        """Asynchronously invoke the Jina search."""
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
