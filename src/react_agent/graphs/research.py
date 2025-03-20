"""Define research nodes for the agent graph.

This module provides functionality for conducting structured research
using a series of nodes in a LangGraph workflow.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Sequence, Union, cast, Awaitable, TypeVar

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.chat import ChatMessage, ChatMessageChunk
from langchain_core.messages.function import FunctionMessage, FunctionMessageChunk
from langchain_core.messages.human import HumanMessage, HumanMessageChunk
from langchain_core.messages.system import SystemMessage, SystemMessageChunk
from langchain_core.messages.tool import ToolMessage, ToolMessageChunk
from langchain_core.runnables import RunnableConfig, ensure_config
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import State
from react_agent.utils.llm import call_model, call_model_json
from react_agent.prompts import (
    RESEARCH_BASE_PROMPT,
    RESEARCH_AGENT_PROMPT,
    QUERY_ANALYSIS_PROMPT,
    CLARIFICATION_PROMPT
)
from react_agent.tools.firecrawl import search as firecrawl_search, scrape as firecrawl_scrape

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
    """Create a search query and perform the search using FireCrawl.

    Args:
        state (State): The current state of the conversation.
        config (Optional[RunnableConfig]): Configuration for the model run.

    Returns:
        Dict[str, Any]: Updated state with search query and results.
    """
    def get_user_query(messages: Sequence[BaseMessage]) -> str:
        """Extract the last user query from messages."""
        for msg in reversed(messages):
            if msg.type == "human":
                content = msg.content
                return " ".join(item for item in content if isinstance(item, str)) if isinstance(content, list) else content
        return ""

    def create_default_analysis() -> Dict[str, Any]:
        """Create default analysis structure."""
        return {
            "search_terms": {section: [] for section in [
                "porters_5_forces", "swot", "pestel", "market_trends",
                "vendor_analysis", "compliance", "technical"
            ]},
            "primary_keywords": [],
            "industry_context": "",
            "search_priority": ["market_trends", "technical", "compliance"],
            "missing_context": ["Analysis failed, needs clarification"]
        }

    async def perform_search(term: str, keywords: List[str], context: str, section: str,
                           visited_urls: set, config: RunnableConfig) -> List[Dict[str, Any]]:
        """Perform search for a single term and return results."""
        results = []
        search_query = f"{term} {' '.join(keywords)} {context}".strip()
        try:
            search_results = await firecrawl_search(search_query, config=config)
            for doc in search_results or []:
                url = doc.metadata.get("source", "")
                if url and url not in visited_urls:
                    visited_urls.add(url)
                    results.append({
                        "url": url,
                        "title": doc.metadata.get("title", ""),
                        "snippet": doc.page_content,
                        "section": section,
                        "search_term": term
                    })
        except Exception as e:
            print(f"Search failed for term '{term}': {str(e)}")
        return results

    # Get user query
    user_query = get_user_query(state.messages)

    # Get search terms analysis
    try:
        analysis_response = await call_model_json(
            messages=[{"role": "user", "content": QUERY_ANALYSIS_PROMPT.format(query=user_query)}],
            config=config
        )
        if not isinstance(analysis_response, dict):
            analysis_response = create_default_analysis()
    except Exception:
        analysis_response = create_default_analysis()

    # Perform searches
    visited_urls = set()
    all_search_results = []
    validated_config = ensure_config(config)

    for section in analysis_response.get("search_priority", []):
        search_terms = analysis_response.get("search_terms", {}).get(section, [])
        if not search_terms:
            continue

        primary_keywords = analysis_response.get("primary_keywords", [])
        industry_context = analysis_response.get("industry_context", "")
        
        for term in search_terms:
            results = await perform_search(
                term, primary_keywords, industry_context, section,
                visited_urls, validated_config
            )
            all_search_results.extend(results)

    return {
        "messages": [ToolMessage(
            content=f"Generated {len(all_search_results)} search results across {len(analysis_response.get('search_priority', []))} analysis sections",
            tool_call_id="search",
            name="firecrawl_search"
        )],
        "search_query": user_query,
        "search_results": all_search_results,
        "search_analysis": analysis_response,
        "visited_urls": list(visited_urls)
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
            query = msg.content
            if isinstance(query, list):
                query = " ".join([item for item in query if isinstance(item, str)])
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
    """Helper function to extract content from a URL using FireCrawl."""
    scraped_docs = await firecrawl_scrape(url, mode="scrape", config=config)
    if not scraped_docs:
        return None
        
    content = "\n\n".join(doc.page_content for doc in scraped_docs)
    return {
        "content": content,
        "metadata": {
            doc.metadata.get("source", ""): doc.metadata 
            for doc in scraped_docs
        }
    }

async def extract_key_information(state: State, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Extract key information from search results using FireCrawl."""
    search_results = getattr(state, "search_results", [])
    visited_urls = getattr(state, "visited_urls", [])

    if not search_results:
        return {
            "messages": [ToolMessage(
                content="No search results to extract information from.",
                tool_call_id="extract_empty",
                name="firecrawl_extract"
            )]
        }

    query = next((msg.content for msg in reversed(state.messages) 
                  if msg.type == "human" and isinstance(msg.content, str)), "")
    
    validated_config = ensure_config(config)
    extracted_info = {}
    sources = []
    
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
            content=f"Extracted information from {len(sources)} sources using FireCrawl.",
            tool_call_id="extract",
            name="firecrawl_extract"
        )],
        "extracted_info": extracted_info,
        "sources": sources
    }


async def synthesize_research(state: State, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Synthesize research from extracted information.

    Args:
        state (State): The current state of the conversation.
        config (Optional[RunnableConfig]): Configuration for the model run.

    Returns:
        Dict[str, Any]: Updated state with research synthesis.
    """
    # Extract relevant information from state
    extracted_info = getattr(state, "extracted_info", {})
    sources = getattr(state, "sources", [])

    if not extracted_info:
        message = ToolMessage(
            content="No information to synthesize. Please try a different query.",
            tool_call_id="synthesize_empty",
            name="synthesize_research"
        )
        return {"messages": [message]}

    # Get the query from the last user message
    query = ""
    for msg in reversed(state.messages):
        if msg.type == "human":
            query = msg.content
            if isinstance(query, list):
                query = " ".join([item for item in query if isinstance(item, str)])
            break

    # Prepare a prompt for synthesis
    prompt = f"""
    Synthesize the following information to answer the query: "{query}"

    Extracted Information:
    {json.dumps(extracted_info, indent=2)}

    Sources:
    {json.dumps([{"url": s.get("url"), "title": s.get("title")} for s in sources], indent=2)}

    Provide a comprehensive synthesis that addresses the query directly.
    Include citations to sources where appropriate.
    """

    synthesis_response = await call_model_json(
        messages=[{"role": "user", "content": prompt}],
        config=config
    )

    synthesis = synthesis_response.get("synthesis", "")
    if not synthesis and "content" in synthesis_response:
        synthesis = synthesis_response["content"]

    # Create a tool message with the synthesis
    message = ToolMessage(
        content=synthesis,
        tool_call_id="synthesize",
        name="synthesize_research"
    )

    return {
        "messages": [message],
        "synthesis": synthesis
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


async def validate_search_terms(state: State, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Validate if the search terms are sufficient for meaningful research.
    
    Args:
        state (State): Current state containing search analysis
        config (Optional[RunnableConfig]): Configuration for the model run
        
    Returns:
        Dict[str, Any]: Updated state with validation results
    """
    def get_safe_value(data: Dict[str, Any], key: str, default: Any, expected_type: type) -> Any:
        """Safely get a value from a dictionary with type checking."""
        value = data.get(key, default)
        return value if isinstance(value, expected_type) else default

    analysis = getattr(state, "search_analysis", {})
    if not analysis:
        return {
            "messages": [AIMessage(content="No search analysis found. Please try your query again.")],
            "needs_clarification": True,
            "clarification_count": getattr(state, "clarification_count", 0) + 1
        }

    # Extract and validate components
    search_terms = get_safe_value(analysis, "search_terms", {}, dict)
    primary_keywords = get_safe_value(analysis, "primary_keywords", [], list)
    industry_context = get_safe_value(analysis, "industry_context", "", str)

    # Define expected sections
    sections = ["porters_5_forces", "swot", "pestel", "market_trends", 
                "vendor_analysis", "compliance", "technical"]

    # Validation criteria
    sections_covered = sum(
        len(get_safe_value(search_terms, section, [], list)) >= 2 
        for section in sections
    )
    
    validation_criteria = {
        "sections_coverage": (sections_covered, 4),
        "keywords": (len(primary_keywords), 2),
        "context": (bool(industry_context.strip()), True)
    }

    # Check validation criteria
    failed_criteria = {
        key: actual 
        for key, (actual, required) in validation_criteria.items() 
        if actual < required
    }

    if not failed_criteria:
        return {
            "messages": [ToolMessage(
                content="Search terms validation passed",
                tool_call_id="validate_terms",
                name="validate_search_terms"
            )],
            "needs_clarification": False,
            "search_terms_validated": True
        }

    # Generate clarification message
    query = next(
        (msg.content for msg in reversed(state.messages) 
         if msg.type == "human" and isinstance(msg.content, str)),
        ""
    )
    
    missing_info = []
    if "sections_coverage" in failed_criteria:
        missing_info.extend(
            f"Need more details for: {section}"
            for section in sections
            if len(get_safe_value(search_terms, section, [], list)) < 2
        )
    if "keywords" in failed_criteria:
        missing_info.append("Need more industry-specific keywords")
    if "context" in failed_criteria:
        missing_info.append("Need more context about the industry/domain")

    return {
        "messages": [AIMessage(content=CLARIFICATION_PROMPT.format(
            query=query,
            industry_context=industry_context,
            primary_keywords=", ".join(primary_keywords),
            missing_sections="\n".join(f"- {item}" for item in missing_info)
        ))],
        "needs_clarification": True,
        "clarification_count": getattr(state, "clarification_count", 0) + 1,
        "search_analysis": {
            "search_terms": search_terms,
            "primary_keywords": primary_keywords,
            "industry_context": industry_context
        }
    }

def route_research_flow(state: State) -> Literal["create_search_query", "validate_search_terms", "process_search_results", "extract_key_information", "synthesize_research", "validate_research_output", "__end__"]:
    """Determine the next step in the research flow.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call.
    """
    # Check if we've exceeded maximum clarification attempts
    if getattr(state, "clarification_count", 0) >= 3:
        return "__end__"
        
    if validation_passed := getattr(state, "validation_passed", False):
        return "__end__"

    if synthesis := getattr(state, "synthesis", ""):
        return "validate_research_output"

    if extracted_info := getattr(state, "extracted_info", {}):
        return "synthesize_research"

    if search_results := getattr(state, "search_results", []):
        return "extract_key_information"

    if search_query := getattr(state, "search_query", ""):
        if not getattr(state, "search_terms_validated", False):
            return "validate_search_terms"
        return "process_search_results"

    # Default to creating a search query
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

    # Create async handlers for each node
    async def handle_create_search_query(state: Any) -> Dict[str, Any]:
        validated_state = ensure_state(state)
        return await create_search_query(validated_state, config)
        
    async def handle_validate_terms(state: Any) -> Dict[str, Any]:
        validated_state = ensure_state(state)
        return await validate_search_terms(validated_state, config)
        
    async def handle_process_results(state: Any) -> Dict[str, Any]:
        validated_state = ensure_state(state)
        return await process_search_results(validated_state, config)
        
    async def handle_extract_info(state: Any) -> Dict[str, Any]:
        validated_state = ensure_state(state)
        return await extract_key_information(validated_state, config)
        
    async def handle_synthesize(state: Any) -> Dict[str, Any]:
        validated_state = ensure_state(state)
        return await synthesize_research(validated_state, config)
        
    async def handle_validate(state: Any) -> Dict[str, Any]:
        validated_state = ensure_state(state)
        return await validate_research_output(validated_state, config)

    # Add nodes with async handlers
    builder.add_node("create_search_query", handle_create_search_query)
    builder.add_node("validate_search_terms", handle_validate_terms)
    builder.add_node("process_search_results", handle_process_results)
    builder.add_node("extract_key_information", handle_extract_info)
    builder.add_node("synthesize_research", handle_synthesize)
    builder.add_node("validate_research_output", handle_validate)

    # Add conditional edges
    builder.add_conditional_edges(
        "__start__",
        route_research_flow,
        {
            "create_search_query": "create_search_query",
            "validate_search_terms": "validate_search_terms",
            "process_search_results": "process_search_results",
            "extract_key_information": "extract_key_information",
            "synthesize_research": "synthesize_research",
            "validate_research_output": "validate_research_output",
            "__end__": "__end__"
        }
    )

    # Add edges between nodes
    builder.add_edge("create_search_query", "validate_search_terms")
    builder.add_edge("validate_search_terms", "process_search_results")
    builder.add_edge("process_search_results", "extract_key_information")
    builder.add_edge("extract_key_information", "synthesize_research")
    builder.add_edge("synthesize_research", "validate_research_output")

    # Add conditional edge from validation
    builder.add_conditional_edges(
        "validate_research_output",
        lambda state: "__end__" if getattr(state, "validation_passed", False) else "create_search_query",
        {
            "__end__": "__end__",
            "create_search_query": "create_search_query"
        }
    )

    # Compile with interrupt points
    return builder.compile(
        interrupt_after=["validate_search_terms"]  # Allow user input after validation
    )

# Create default graph instance with default configuration
research_graph = create_research_graph()