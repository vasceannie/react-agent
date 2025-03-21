"""Define research nodes for the agent graph.

This module provides functionality for conducting structured research
using a series of nodes in a LangGraph workflow.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import (
    Any,
    Awaitable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.chat import ChatMessage, ChatMessageChunk
from langchain_core.messages.function import FunctionMessage, FunctionMessageChunk
from langchain_core.messages.human import HumanMessage, HumanMessageChunk
from langchain_core.messages.system import SystemMessage, SystemMessageChunk
from langchain_core.messages.tool import ToolMessage, ToolMessageChunk
from langchain_core.runnables import RunnableConfig, ensure_config
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.prompts import (
    CLARIFICATION_PROMPT,
    QUERY_ANALYSIS_PROMPT,
    RESEARCH_AGENT_PROMPT,
    RESEARCH_BASE_PROMPT,
)
from react_agent.state import State
from react_agent.tools.jina import search as jina_search
from react_agent.utils.llm import call_model, call_model_json

T = TypeVar('T')

def ensure_state(state: Any) -> State:
    """Ensure the input is a valid State object.
    
    Args:
        state: Input state to validate
        
    Returns:
        State: Validated state object
        
    Raises:
        TypeError: If state is not a valid State object
    """
    if not isinstance(state, State):
        raise TypeError(f"Expected State object, got {type(state)}")
    return state

async def create_search_query(state: State, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Create a search query and perform the search using Jina.

    Args:
        state (State): The current state of the conversation.
        config (Optional[RunnableConfig]): Configuration for the model run.

    Returns:
        Dict[str, Any]: Updated state with search results.
    """
    # Get the last user query
    query = ""
    for msg in reversed(state.messages):
        if msg.type == "human":
            content = msg.content
            query = " ".join([item for item in content if isinstance(item, str)]) if isinstance(content, list) else str(content)
            break

    if not query:
        return {
            "messages": [ToolMessage(
                content="No query found to search with.",
                tool_call_id="search_empty",
                name="jina_search"
            )]
        }

    # Enrich the query using the LLM
    enrichment_prompt = f"""Analyze the following search query and enhance it for better search results.
    Original query: "{query}"

    1. Identify key concepts and topics
    2. Add relevant synonyms or alternative phrasings
    3. Include any missing context that would improve results
    4. Format as a clear, focused search query
    5. Ensure query captures the original intent

    Return a JSON object with:
    {{
        "enhanced_query": "the improved search query",
        "key_concepts": ["list of main concepts"],
        "search_strategy": "brief explanation of search approach"
    }}
    """

    # Ensure we have a valid config
    validated_config = ensure_config(config or {})
    
    try:
        enrichment_result = await call_model_json(
            messages=[{"role": "user", "content": enrichment_prompt}],
            config=validated_config
        )
        
        enhanced_query = enrichment_result.get("enhanced_query", query)
        key_concepts = enrichment_result.get("key_concepts", [])
        search_strategy = enrichment_result.get("search_strategy", "")
        
        # Log the enrichment process
        print(f"Original query: {query}")
        print(f"Enhanced query: {enhanced_query}")
        print(f"Key concepts: {', '.join(key_concepts)}")
        print(f"Search strategy: {search_strategy}")
        
        # Perform the search with enhanced query
        search_results = await jina_search(enhanced_query, config=validated_config)
    except Exception as e:
        print(f"Query enrichment failed: {str(e)}, falling back to original query")
        search_results = await jina_search(query, config=validated_config)

    if not search_results:
        return {
            "messages": [ToolMessage(
                content="No search results found.",
                tool_call_id="search_empty",
                name="jina_search"
            )]
        }

    return {
        "messages": [ToolMessage(
            content=f"Found {len(search_results)} relevant results.",
            tool_call_id="search",
            name="jina_search"
        )],
        "search_results": search_results,
        "original_query": query,
        "enhanced_query": enhanced_query if "enhanced_query" in locals() else query,
        "key_concepts": key_concepts if "key_concepts" in locals() else [],
        "search_strategy": search_strategy if "search_strategy" in locals() else ""
    }

async def process_search_results(state: State, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Process and prioritize search results.

    Args:
        state (State): The current state of the conversation.
        config (Optional[RunnableConfig]): Configuration for the model run.

    Returns:
        Dict[str, Any]: Updated state with processed search results.
    """
    # Extract search results from state
    search_results: List[Dict[str, Any]] = getattr(state, "search_results", [])
    visited_urls: List[str] = getattr(state, "visited_urls", [])

    if not search_results:
        message: ToolMessage = ToolMessage(
            content="No search results found. Please try a different query.",
            tool_call_id="process_empty",
            name="process_search_results"
        )
        return {"messages": [message]}

    # Extract keywords from the last query
    query = ""
    for msg in reversed(state.messages):
        if msg.type == "human":
            content = msg.content
            query = " ".join([item for item in content if isinstance(item, str)]) if isinstance(content, list) else str(content)
            break

    keywords = query.lower().split()

    # Score and sort results
    scored_results = []
    for result in search_results:
        if not isinstance(result, dict):
            continue

        title = result.get("title", "")
        snippet = result.get("snippet", "")
        text = f"{title} {snippet}".lower()

        # Calculate relevance score
        score = sum(keyword.lower() in text for keyword in keywords)

        # Boost credible sources
        url = result.get("url", "")
        if isinstance(url, str) and any(
            domain in url for domain in ["wikipedia.org", "github.com", ".gov", ".edu"]
        ):
            score += 2

        # Penalize already visited URLs
        if url in visited_urls:
            score -= 10

        scored_results.append((score, result))

    # Sort by score and take top results
    sorted_results = sorted(scored_results, key=lambda x: x[0], reverse=True)
    top_results = []
    new_visited_urls = []

    for _, result in sorted_results[:3]:
        url = result.get("url")
        if isinstance(url, str) and url and url not in visited_urls:
            top_results.append(result)
            new_visited_urls.append(url)

    # Update visited URLs
    all_visited_urls = visited_urls + new_visited_urls

    # Create a tool message to record this step
    message = ToolMessage(
        content=f"Selected {len(top_results)} most relevant results for processing.",
        tool_call_id="process",
        name="process_search_results"
    )

    return {
        "messages": [message],
        "search_results": top_results,
        "visited_urls": all_visited_urls
    }

async def _extract_content_from_url(url: str, config: RunnableConfig) -> Optional[Dict[str, Any]]:
    """Extract content from a URL using Jina."""
    # Use the URL as a search query to find content about that specific URL
    scraped_docs = await jina_search(f"site:{url}", config=config)
    if not scraped_docs:
        return None
        
    content = "\n\n".join(doc.page_content for doc in scraped_docs)
    return {
        "content": content,
        "metadata": {
            doc.metadata.get("url", ""): doc.metadata 
            for doc in scraped_docs
        }
    }

async def extract_key_information(state: State, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Extract key information from search results using Jina."""
    search_results = getattr(state, "search_results", [])
    visited_urls = getattr(state, "visited_urls", [])

    if not search_results:
        return {
            "messages": [ToolMessage(
                content="No search results to extract information from.",
                tool_call_id="extract_empty",
                name="jina_extract"
            )]
        }

    query = ""
    for msg in reversed(state.messages):
        if msg.type == "human":
            content = msg.content
            if isinstance(content, list):
                query = " ".join([item for item in content if isinstance(item, str)])
            else:
                query = str(content)
            break
    
    validated_config = ensure_config(config)
    extracted_info: Dict[str, List[Any]] = {}
    sources: List[Dict[str, Any]] = []
    
    for result in search_results:
        url = result.get("url", "")
        if not url or url in visited_urls:
            continue

        try:
            extracted = await _extract_content_from_url(url, validated_config)
            if not extracted:
                continue

            extraction_result = await call_model_json(
                messages=[{
                    "role": "user", 
                    "content": f'Extract key information from the following content relevant to: "{query}"\n\nContent:\n{extracted["content"]}\n\nReturn a JSON object with: {{"key_points": [], "entities": [], "quotes": []}}'
                }],
                config=validated_config
            )

            for key, value in extraction_result.items():
                if key not in extracted_info:
                    extracted_info[key] = []
                if isinstance(value, list):
                    extracted_info[key].extend(value)

            sources.append({
                "url": url,
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
                "content": extracted["content"],
                "metadata": extracted["metadata"]
            })
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")

    return {
        "messages": [ToolMessage(
            content=f"Extracted information from {len(sources)} sources using Jina.",
            tool_call_id="extract",
            name="jina_extract"
        )],
        "extracted_info": extracted_info,
        "sources": sources
    }

async def synthesize_research(state: State, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Synthesize research from search results.

    Args:
        state (State): The current state of the conversation.
        config (Optional[RunnableConfig]): Configuration for the model run.

    Returns:
        Dict[str, Any]: Updated state with research synthesis.
    """
    search_results = getattr(state, "search_results", [])
    
    if not search_results:
        return {
            "messages": [ToolMessage(
                content="No search results to synthesize.",
                tool_call_id="synthesize_empty",
                name="synthesize_research"
            )]
        }

    # Get the original query
    query = ""
    for msg in reversed(state.messages):
        if msg.type == "human":
            content = msg.content
            query = " ".join([item for item in content if isinstance(item, str)]) if isinstance(content, list) else str(content)
            break

    # Prepare content for synthesis
    content_for_synthesis = []
    for doc in search_results:
        content_for_synthesis.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })

    # Create synthesis prompt
    prompt = f"""
    Based on the search results, provide a comprehensive answer to: "{query}"

    Search Results:
    {json.dumps(content_for_synthesis, indent=2)}

    Synthesize the information into a clear, well-structured response.
    Include relevant citations to sources where appropriate.
    """

    synthesis_response = await call_model_json(
        messages=[{"role": "user", "content": prompt}],
        config=config
    )

    synthesis = synthesis_response.get("synthesis", "")
    if not synthesis and isinstance(synthesis_response, dict):
        synthesis = synthesis_response.get("content", "")

    return {
        "messages": [ToolMessage(
            content=synthesis,
            tool_call_id="synthesize",
            name="synthesize_research"
        )],
        "synthesis": synthesis,
        "synthesis_complete": True
    }

async def validate_research_output(state: State, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Validate the research output for quality and accuracy.

    Args:
        state (State): The current state of the conversation.
        config (Optional[RunnableConfig]): Configuration for the model run.

    Returns:
        Dict[str, Any]: Updated state with validation results.
    """
    # Extract synthesis from state
    synthesis = getattr(state, "synthesis", "")

    if not synthesis:
        message = ToolMessage(
            content="No synthesis to validate.",
            tool_call_id="validate_empty",
            name="validate_research_output"
        )
        return {"messages": [message]}

    # Prepare a prompt for validation
    prompt = f"""
    Validate the following research synthesis:

    {synthesis}

    Evaluate based on:
    1. Factual accuracy
    2. Comprehensiveness
    3. Clarity
    4. Citation of sources

    Provide a validation score (0-100) and list any issues found.
    """

    validation_result = await call_model_json(
        messages=[{"role": "user", "content": prompt}],
        config=config
    )

    score = validation_result.get("score", 0)
    validation_passed = isinstance(score, (int, float)) and score >= 70
    # Create a tool message with validation results
    message = ToolMessage(
        content=f"Validation {'passed' if validation_passed else 'failed'} with score {score}/100.",
        tool_call_id="validate",
        name="validate_research_output"
    )

    return {
        "messages": [message],
        "validation": validation_result,
        "validation_passed": validation_passed
    }

def route_research_flow(state: State) -> Literal["create_search_query", "process_search_results", "extract_key_information", "synthesize_research", "validate_research_output", "__end__"]:
    """Determine the next step in the research flow.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call.
    """
    if getattr(state, "validation_passed", False):
        return "__end__"

    if getattr(state, "synthesis", None):
        return "validate_research_output"

    if getattr(state, "extracted_info", None):
        return "synthesize_research"

    if getattr(state, "search_results", None) and not getattr(state, "processed_results", None):
        return "process_search_results"

    if getattr(state, "processed_results", None):
        return "extract_key_information"

    return "create_search_query"

def create_research_graph(config: Optional[RunnableConfig] = None) -> CompiledStateGraph:
    """Create the research workflow graph.

    Args:
        config (Optional[RunnableConfig]): Configuration for the graph.

    Returns:
        CompiledStateGraph: The compiled research graph.
    """
    # Create the graph with the state
    builder = StateGraph(State)

    # Add nodes
    builder.add_node("create_search_query", 
                    create_search_query)
    builder.add_node("process_search_results", 
                    process_search_results)
    builder.add_node("extract_key_information", 
                    extract_key_information)
    builder.add_node("synthesize_research", 
                    synthesize_research)
    builder.add_node("validate_research_output", 
                    validate_research_output)

    # Add conditional edges
    builder.add_conditional_edges(
        "__start__",
        route_research_flow,
        {
            "create_search_query": "create_search_query",
            "process_search_results": "process_search_results",
            "extract_key_information": "extract_key_information",
            "synthesize_research": "synthesize_research",
            "validate_research_output": "validate_research_output",
            "__end__": "__end__"
        }
    )

    # Add edges between nodes
    builder.add_edge("create_search_query", "process_search_results")
    builder.add_edge("process_search_results", "extract_key_information")
    builder.add_edge("extract_key_information", "synthesize_research")
    builder.add_edge("synthesize_research", "validate_research_output")
    builder.add_edge("validate_research_output", "__end__")

    return builder.compile()

# Create default graph instance
research_graph = create_research_graph()