"""Modular research framework using LangGraph.

This module implements a modular approach to the research process,
with specialized components for different research categories and
improved error handling and validation.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from datetime import UTC, datetime, timezone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig, ensure_config
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# Define a no-op replacement for step_caching
def step_caching(*args, **kwargs):
    """Replacement for langgraph's step_caching decorator.
    This is a no-op version that simply returns the function unchanged.
    """
    def decorator(func): return func
    return decorator

from langgraph.graph.state import CompiledStateGraph
from langgraph.managed import IsLastStep
from langgraph.prebuilt import InjectedState, InjectedStore, ToolNode
from langgraph.store.base import BaseStore
import operator
from pydantic import BaseModel, Field
from typing_extensions import (
    Annotated,
    NotRequired,
    Required,
    TypedDict,
    Unpack,
    get_type_hints,
)

from react_agent.prompts.query import detect_vertical, expand_acronyms, optimize_query

# Import prompt templates
from react_agent.prompts.research import (
    CLARIFICATION_PROMPT,
    QUERY_ANALYSIS_PROMPT,
    SEARCH_QUALITY_THRESHOLDS,
    SYNTHESIS_PROMPT,
    VALIDATION_PROMPT,
    get_default_extraction_result,
    get_extraction_prompt,
)
from react_agent.prompts.enhanced_templates import (
    ENHANCED_CATEGORY_TEMPLATES,
)
from react_agent.tools.derivation import (
    CategoryExtractionTool,
    ExtractionTool,
    ResearchValidationTool,
    StatisticsAnalysisTool,
    create_derivation_toolnode,
)

# Import tools and tool-related utilities
from react_agent.tools.jina import search
from react_agent.types import (
    ContentChunk,
    DocumentContent,
    DocumentPage,
    ExtractedFact,
    ProcessedDocument,
    ResearchSettings,
    SearchResult,
    Statistics,
    TextContent,
    WebContent,
)
from react_agent.utils.cache import create_checkpoint, load_checkpoint
from react_agent.utils.content import (
    DOCLING_AVAILABLE,
    detect_content_type,
    process_document_with_docling,
    should_skip_content,
    validate_content,
)
from react_agent.utils.search import (
    CategorySuccessTracker,
    standardize_search_result,
    enhance_search_results,
    get_optimized_query,
    log_search_event,
)
from react_agent.utils.enhanced_search import (
    execute_progressive_search,
)
from react_agent.utils.extraction import enrich_extracted_fact
from react_agent.utils.extraction import (
    extract_category_information as original_extract_category_info,
)
from react_agent.utils.llm import call_model_json, get_extraction_model, log_dict
from react_agent.utils.logging import (
    error_highlight,
    get_logger,
    info_highlight,
    log_step,
    warning_highlight,
)
from react_agent.utils.statistics import (
    assess_synthesis_quality,
    calculate_overall_confidence,
)

# Initialize logger
logger = get_logger("research_graph")

# Define SearchType as a Literal type
SearchType = Literal['general', 'authoritative', 'recent', 'comprehensive', 'technical']

# Enhanced state types with proper annotations
class ResearchCategoryState(TypedDict):
    """State for a specific research category with improved type safety."""
    category: str
    query: str
    search_results: List[Dict[str, Any]]
    extracted_facts: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    complete: bool
    quality_score: float
    retry_count: int
    last_search_query: Optional[str]
    status: Literal["pending", "in_progress", "searching", "searched", "search_failed", 
                    "facts_extracted", "complete", "failed"]
    statistics: List[Dict[str, Any]]
    confidence_score: float
    cross_validation_score: float
    source_quality_score: float
    recency_score: float
    statistical_content_score: float


class EnhancedResearchState(TypedDict):
    """Main research state with improved type safety and annotations."""
    # Basic conversation data
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Original query and analysis
    original_query: str
    query_analysis: Optional[Dict[str, Any]]
    
    # Clarity and context
    missing_context: List[str]
    needs_clarification: bool
    clarification_request: Optional[str]
    human_feedback: Optional[str]
    
    # Category-specific research
    categories: Annotated[Dict[str, ResearchCategoryState], operator.or_]
    
    # Synthesis and validation
    synthesis: Optional[Dict[str, Any]]
    validation_result: Optional[Dict[str, Any]]
    consolidated_report: Optional[Dict[str, Any]]
    
    # Overall status
    status: Annotated[Literal["initialized", "analyzed", "clarified", "researched", "synthesized", "validated", "report_generated", "complete", "error"], operator.add]
    error: Optional[Dict[str, Any]]
    complete: bool
    is_last_step: NotRequired[IsLastStep]


class ResearchState(TypedDict):
    """Main research state with modular components."""
    # Basic conversation data
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Original query and analysis
    original_query: str
    query_analysis: Optional[Dict[str, Any]]

    # Clarity and context
    missing_context: List[str]
    needs_clarification: bool
    clarification_request: Optional[str]
    human_feedback: Optional[str]

    # Category-specific research
    categories: Annotated[Dict[str, ResearchCategoryState], operator.or_]

    # Synthesis and validation
    synthesis: Optional[Dict[str, Any]]
    validation_result: Optional[Dict[str, Any]]

    # Overall status
    status: Annotated[Literal["initialized", "analyzed", "clarified", "researched", "synthesized", "validated", "report_generated", "complete", "error"], operator.add]
    error: Optional[Dict[str, Any]]
    complete: bool


# Initialize tools
statistics_tool = StatisticsAnalysisTool()
extraction_tool = ExtractionTool()
validation_tool = ResearchValidationTool()
category_tool = None  # Will be initialized with extraction_model when needed

# --------------------------------------------------------------------
# 1. Core control flow nodes
# --------------------------------------------------------------------

async def initialize_research(state: ResearchState) -> Dict[str, Any]:
    """Initialize the research process with the user's query."""
    log_step("Initializing research process", 1, 10)

    if not state["messages"]:
        warning_highlight("No messages found in state")
        return {"error": {"message": "No messages in state", "phase": "initialization"}}

    last_message = state["messages"][-1]
    query = last_message.content if isinstance(last_message, BaseMessage) else ""

    if not query:
        warning_highlight("Empty query")
        return {"error": {"message": "Empty query", "phase": "initialization"}}

    info_highlight(f"Initializing research for query: {query}")

    # Initialize categories with default values for each research category
    categories = {
        category: {
            "category": category,
            "query": "",  # Will be filled by query analysis
            "search_results": [],
            "extracted_facts": [],
            "sources": [],
            "complete": False,
            "quality_score": 0.0,
            "retry_count": 0,
            "last_search_query": None,
            "status": "pending",
            "statistics": [],
            "confidence_score": 0.0,
            "cross_validation_score": 0.0,
            "source_quality_score": 0.0,
            "recency_score": 0.0,
            "statistical_content_score": 0.0
        }
        for category in SEARCH_QUALITY_THRESHOLDS.keys()
    }
    
    return {
        "original_query": query,
        "status": "initialized",
        "categories": categories,
        "missing_context": [],
        "needs_clarification": False,
        "complete": False
    }


async def analyze_query(state: ResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Analyze the query to determine research categories and search terms."""
    log_step("Analyzing research query", 2, 10)

    query = state["original_query"].strip()
    if not query:
        warning_highlight("No query to analyze")
        return {"error": {"message": "No query to analyze", "phase": "query_analysis"}}

    if human_feedback := state.get("human_feedback", ""):
        info_highlight(f"Including user feedback in analysis: {human_feedback}")
        query = f"{query}\n\nAdditional context: {human_feedback}"

    # Prepare the analysis prompt
    analysis_prompt = QUERY_ANALYSIS_PROMPT.format(query=query)

    try:
        analysis_result = await call_model_json(
            messages=[{"role": "human", "content": analysis_prompt}],
            config=ensure_config(config)
        )

        # Ensure the response has the required structure
        if not isinstance(analysis_result, dict):
            error_highlight("Invalid response format from query analysis")
            return {"error": {"message": "Invalid response format", "phase": "query_analysis"}}

        # Initialize default values
        analysis_result = {
            "unspsc_categories": analysis_result.get("unspsc_categories", []),
            "search_components": analysis_result.get("search_components", {
                "primary_topic": "",
                "industry": "",
                "product_type": "",
                "geographical_focus": ""
            }),
            "search_terms": analysis_result.get("search_terms", {
                "market_dynamics": [],
                "provider_landscape": [],
                "technical_requirements": [],
                "regulatory_landscape": [],
                "cost_considerations": [],
                "best_practices": [],
                "implementation_factors": []
            }),
            "boolean_query": analysis_result.get("boolean_query", ""),
            "missing_context": analysis_result.get("missing_context", [])
        }

        # Check for missing context
        missing_context = analysis_result.get("missing_context", [])
        needs_clarification = len(missing_context) >= 2  # Only request clarification for multiple missing elements

        # Update category queries based on analysis
        categories = state["categories"]
        for category, category_state in categories.items():
            if search_terms := analysis_result.get("search_terms", {}).get(category, []):
                # Detect vertical from the query
                vertical = detect_vertical(query)

                # Create optimized query for this category
                optimized_query = optimize_query(
                    original_query=query,
                    category=category,
                    vertical=vertical,
                    include_all_keywords=len(search_terms) > 5  # Include all keywords for complex queries
                )
                category_state["query"] = optimized_query
                info_highlight(f"Set optimized query for {category}: {optimized_query}")
            else:
                # Use default query if no specific terms
                category_state["query"] = query

        return {
            "query_analysis": analysis_result,
            "categories": categories,
            "missing_context": missing_context,
            "needs_clarification": needs_clarification,
            "status": "analyzed"
        }
    except Exception as e:
        error_highlight(f"Error in query analysis: {str(e)}")
        return {"error": {"message": f"Error in query analysis: {str(e)}", "phase": "query_analysis"}}


async def request_clarification(state: ResearchState) -> Dict[str, Any]:
    """Request clarification from the user for missing context."""
    log_step("Requesting clarification", 3, 10)

    missing_context = state["missing_context"]
    if not missing_context:
        return {"needs_clarification": False}

    # Format the missing context items
    missing_sections = "\n".join([f"- {item}" for item in missing_context])

    # Get analysis data
    analysis = state.get("query_analysis", {})
    search_components = analysis.get("search_components", {}) if analysis else {}
    
    # Prepare the clarification request using the prompt template
    try:
        clarification_message = CLARIFICATION_PROMPT.format(
            query=state["original_query"],
            product_vs_service=search_components.get("product_type", "Unknown"),
            industry_context=search_components.get("industry", "Unknown"),
            geographical_focus=search_components.get("geographical_focus", "Unknown"),
            missing_sections=missing_sections
        )

        info_highlight("Generated clarification request")

        # Set up interrupt for user input
        return {
            "clarification_request": clarification_message,
            "__interrupt__": {
                "value": {
                    "question": clarification_message,
                    "missing_context": missing_context
                },
                "resumable": True,
                "ns": ["request_clarification"],
                "when": "during"
            }
        }
    except Exception as e:
        error_highlight(f"Error creating clarification request: {str(e)}")
        # Proceed without clarification in case of error
        return {"needs_clarification": False}


async def process_clarification(state: ResearchState) -> Dict[str, Any]:
    """Process user clarification and update the research context."""
    log_step("Processing clarification", 4, 10)

    if not state["messages"]:
        warning_highlight("No messages found for clarification")
        return {}

    # Get the latest message which should contain the user's clarification
    last_message = state["messages"][-1]
    clarification_content = str(last_message.content).strip()

    info_highlight(f"Processing user clarification: {clarification_content}")

    # Update the original query with the clarification
    updated_query = f"{state['original_query']}\n\nAdditional context: {clarification_content}"

    return {
        "human_feedback": clarification_content,
        "original_query": updated_query,  # Update the original query with clarification
        "needs_clarification": False,
        "missing_context": [],
        "status": "clarified"
    }


# --------------------------------------------------------------------
# 2. Category-specific research nodes
# --------------------------------------------------------------------

async def execute_category_search(
        state: ResearchState,
        category: str,
        fallback_level: int = 0,
        config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Execute search for a specific research category."""
    log_step(f"Executing search for category: {category}", 5, 10)

    categories = state["categories"]
    if category not in categories:
        warning_highlight(f"Unknown category: {category}")
        return {}
    
    category_state = categories[category]
    query = category_state["query"]

    if not query:
        warning_highlight(f"No query available for category: {category}")
        return {}

    # Update status
    category_state["status"] = "searching"
    category_state["retry_count"] += 1
    
    # Initialize category success tracker if not already present in state
    success_tracker = getattr(state, "category_success_tracker", CategorySuccessTracker())

    # Get category-specific search parameters
    thresholds = SEARCH_QUALITY_THRESHOLDS.get(category, {})
    recency_days = int(str(thresholds.get("recency_threshold_days", 365)))

    # Configure search params based on category
    search_type_mapping: Dict[str, SearchType] = {
        "market_dynamics": "recent",  # Need fresh market data
        "provider_landscape": "comprehensive",  # Need diverse providers
        "technical_requirements": "technical",  # Technical content
        "regulatory_landscape": "authoritative",  # Authoritative sources
        "cost_considerations": "recent",  # Fresh pricing data
        "best_practices": "comprehensive",  # Diverse practices
        "implementation_factors": "comprehensive"  # Diverse approaches
    }

    search_type = search_type_mapping.get(category, "general")
    
    # Extract primary terms from query
    primary_terms = " ".join(query.split()[:5])  # Use first 5 terms as primary
    
    # Check if we should use optimized query
    if fallback_level > 0 or category in ["best_practices", "regulatory_landscape", "implementation_factors"]:
        # Use our enhanced query optimization
        optimized_query = get_optimized_query(category, primary_terms, fallback_level)
        info_highlight(f"Using optimized query for {category} (fallback level {fallback_level}): {optimized_query}")
        query = optimized_query
    
    # Record start time for performance tracking
    start_time = time.time()

    info_highlight(f"Executing {search_type} search for {category} with query: {query} (attempt {category_state['retry_count']})")

    try:
        # Keep track of the query for retry tracking
        category_state["last_search_query"] = query

        # Execute the search with proper parameters using ainvoke
        from langchain.tools import StructuredTool
        
        # Convert the imported search function to a StructuredTool if it's not already
        if not isinstance(search, StructuredTool):
            from langchain.tools import Tool
            search_tool = Tool.from_function(
                func=search,
                name="search",
                description="Search the web for information",
                coroutine=search
            )
        else:
            search_tool = search
            
        # Use ainvoke to call the tool asynchronously
        try:
            # Call the search tool with just the query parameter
            search_results = await search_tool.ainvoke(query)
            info_highlight(f"Search completed for {category} with {len(search_results) if isinstance(search_results, list) else 'non-list'} results")
        except Exception as e:
            error_highlight(f"Search failed for {category}: {str(e)}")
            category_state["status"] = "search_failed"
            return {"categories": categories}
        
        if not search_results:
            warning_highlight(f"No search results for {category}")
            
            # Try fallback if we haven't reached max retries
            if fallback_level < 2 and category_state["retry_count"] < 3:
                info_highlight(f"Attempting fallback query for {category} (level {fallback_level + 1})")
                # Track the failed attempt
                success_tracker.track_attempt(category, False)
                # Try again with fallback
                return await execute_category_search(state, category, fallback_level + 1, config)
            else:
                category_state["status"] = "search_failed"
                # Track the failed attempt
                success_tracker.track_attempt(category, False)
                return {"categories": categories}

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Use our standardization utility to handle different result formats
        formatted_results = standardize_search_result(
            search_results, 
            category=category,
            query=query,
            start_time=start_time
        )
        
        # Enhance results with additional metadata
        enhanced_results = enhance_search_results(
            formatted_results,
            category=category,
            query=query,
            processing_time_ms=processing_time_ms
        )
        
        # Update the category state
        category_state["search_results"] = enhanced_results
        category_state["status"] = "searched"

        # Log search event
        log_search_event(
            category=category,
            query=query,
            result_count=len(enhanced_results),
            duration_ms=processing_time_ms,
            success=True
        )
        
        # Track successful attempt
        success_tracker.track_attempt(category, True)
        
        # Store success tracker in state for future use
        state["category_success_tracker"] = success_tracker

        info_highlight(f"Found {len(enhanced_results)} results for {category}")
        return {"categories": categories}

    except Exception as e:
        error_highlight(f"Error in search for {category}: {str(e)}")
        category_state["status"] = "search_failed"
        
        # Track failed attempt
        success_tracker.track_attempt(category, False)
        
        # Store success tracker in state for future use
        state["category_success_tracker"] = success_tracker
        
        return {"categories": categories}


async def _process_search_result(
        result: Dict[str, Any],
        category: str,
        original_query: str,
        config: Optional[RunnableConfig] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process a single search result and extract information."""
    url = result.get("url", "")
    if not url or should_skip_content(url):
        return [], [], []

    # Check checkpoint first
    cache_key = f"search_result_{url}_{category}"
    if cached_state := load_checkpoint(cache_key):
        info_highlight(f"Using cached result for {url} in {category}")
        return cached_state.get("facts", []), cached_state.get("sources", []), cached_state.get("statistics", [])

    content = result.get("snippet", "")
    if not validate_content(content):
        return [], [], []

    content_type = detect_content_type(url, content)
    
    # Handle document file types with Docling if available
    if DOCLING_AVAILABLE and content_type in ('pdf', 'doc', 'excel', 'presentation', 'document'):
        try:
            info_highlight(f"Processing document with Docling: {url}", category="extraction")
            extracted_text, detected_type = process_document_with_docling(url)
            if validate_content(extracted_text):
                content = extracted_text
                content_type = detected_type
                info_highlight(f"Successfully extracted text from document: {url}", category="extraction")
            else:
                warning_highlight(f"Extracted text from document didn't meet validation criteria: {url}", category="extraction")
        except Exception as e:
            warning_highlight(f"Error processing document with Docling: {str(e)}", category="extraction")
            # Continue with whatever content we have
    
    prompt_template = get_extraction_prompt(
        category=category,
        query=original_query,
        url=url,
        content=content
    )

    try:
        # Extract category information
        # Try using the new extract_with_tool function first
        try:
            extraction_result = await extract_with_tool(
                content=content,
                url=url,
                title=result.get("title", ""),
                category=category,
                original_query=original_query
            )
            facts = extraction_result.get("facts", [])
            relevance_score = extraction_result.get("relevance", 0.5)
        except Exception as e:
            # Fall back to the original method if the new one fails
            facts, relevance_score = await original_extract_category_info(
                content=content,
                url=url,
                title=result.get("title", ""),
                category=category,
                original_query=original_query,
                prompt_template=prompt_template,
                extraction_model=call_model_json,
                config=ensure_config(config)
            )

        # Extract statistics using the statistics tool
        statistics = []
        try:
            # Try to use the statistics analysis tool first
            try:
                info_highlight(f"Using statistics analysis tool for {url}")
                
                # Use the statistics analysis tool
                statistics = await statistics_tool.ainvoke({
                    "operation": "synthesis",  
                    "text": content,
                    "url": url,
                    "source_title": result.get("title", ""),
                    "synthesis": {  
                        "content": content,
                        "url": url,
                        "title": result.get("title", "")
                    }
                }, config)
                
                if statistics:
                    info_highlight(f"Successfully extracted {len(statistics)} statistics using analysis tool")
            except Exception as stats_error:
                warning_highlight(f"Error using statistics analysis tool: {str(stats_error)}")
                # Fall back to the original statistics tool
                info_highlight(f"Falling back to original statistics tool for {url}")
                statistics = statistics_tool.run(tool_input={
                    "operation": "statistics",  # Add required operation parameter
                    "text": content,
                    "url": url,
                    "source_title": result.get("title", "")
                })
        except Exception as e:
            logger.error(f"Error extracting statistics: {str(e)}")
            logger.exception(e)

        # Add source information to facts
        for fact in facts:
            fact["source_url"] = url
            fact["source_title"] = result.get("title", "")

        source = {
            "url": url,
            "title": result.get("title", ""),
            "published_date": result.get("published_date"),
            "fact_count": len(facts),
            "relevance_score": relevance_score,
            "quality_score": result.get("quality_score", 0.5),
            "content_type": content_type
        }

        result_tuple = (facts, [source], statistics)

        # Save to checkpoint with TTL
        create_checkpoint(
            cache_key,
            {
                "facts": facts,
                "sources": [source],
                "statistics": statistics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            ttl=3600  # 1 hour TTL
        )

        return result_tuple

    except Exception as e:
        warning_highlight(f"Error extracting from {url}: {str(e)}")
        return [], [], []


async def extract_with_tool(
    content: str,
    url: str,
    title: str,
    category: str,
    original_query: str,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Extract category information using the extraction tool.
    
    This function uses the ExtractionTool from the tools package to extract
    information from content for a specific category.
    
    Args:
        content: The text content to extract from
        url: The source URL
        title: The source title
        category: The category to extract information for
        original_query: The original search query
        config: Optional runnable config
        
    Returns:
        Dictionary with extracted facts and relevance score
    """
    try:
        # Use the extraction tool
        extraction_result = await extraction_tool.ainvoke({
            "operation": "category",
            "text": content,
            "url": url,
            "source_title": title,
            "category": category,
            "original_query": original_query
        }, config)
        
        # Return the extraction result
        return {
            "facts": extraction_result.get("facts", []),
            "relevance": extraction_result.get("relevance", 0.5)
        }
    except Exception as e:
        warning_highlight(f"Error using extraction tool: {str(e)}")
        raise


async def extract_category_facts(
        state: ResearchState,
        category: str,
        config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Extract facts from search results for a specific category."""
    log_step(f"Extracting facts for category: {category}", 6, 10)
    
    categories = state["categories"]
    if category not in categories:
        warning_highlight(f"Unknown category: {category}")
        return {}
    
    category_state = categories[category]
    search_results = category_state["search_results"]
    
    if not search_results:
        warning_highlight(f"No search results for category: {category}")
        category_state["status"] = "extraction_failed"
        return {"categories": categories}
    
    try:
        # Try to use the tools node first for better integration with langgraph
        if "tools" in state:
            info_highlight(f"Using tools node for category extraction: {category}")
            
            # Create a list of extraction tasks using the tools node
            extraction_tasks = []
            for result in search_results:
                # Prepare the input for the extraction tool
                tool_input = {
                    "operation": "category",
                    "text": result.get("snippet", ""),
                    "url": result.get("url", ""),
                    "source_title": result.get("title", ""),
                    "category": category,
                    "original_query": state["original_query"],
                    "extraction_model": state.get("extraction_model")
                }
                
                # Create a task to invoke the extraction tool
                task = asyncio.create_task(extraction_tool.ainvoke(tool_input, config))
                extraction_tasks.append((task, result))
            
            # Wait for all extraction tasks to complete
            all_facts, all_sources, all_statistics = [], [], []
            for i, (task, result) in enumerate(extraction_tasks):
                try:
                    extraction_result = await task
                    facts = extraction_result.get("facts", [])
                    all_facts.extend(facts)
                    
                    # Add source information
                    source = {
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                        "published_date": result.get("published_date"),
                        "fact_count": len(facts)
                    }
                    all_sources.append(source)
                    
                    # Extract statistics using the statistics tool
                    try:
                        info_highlight(f"Extracting statistics for result {i+1}/{len(extraction_tasks)} in {category}")

                        # Use the statistics extraction tool with the correct operation
                        stats_input = {
                            "operation": "statistics",  # Add required operation parameter
                            "text": result.get("snippet", ""),
                            "url": result.get("url", ""),
                            "source_title": result.get("title", "")
                        }

                        # Create a task to invoke the statistics tool
                        stats_result = await statistics_tool.ainvoke(stats_input, config)

                        if stats_result:
                            info_highlight(f"Extracted {len(stats_result)} statistics from result {i+1}")
                            all_statistics.extend(stats_result)
                    except Exception as stats_error:
                        warning_highlight(f"Error extracting statistics: {str(stats_error)}")
                        # Continue with the next result
                except Exception as e:
                    error_highlight(f"Error in extraction task: {str(e)}")
                    # Continue with the next result
                    continue
        else:
            # Fall back to the original method
            info_highlight(f"Using original extraction method for category: {category}")
            extraction_tasks = [asyncio.create_task(_process_search_result(
                result=result, category=category, 
                original_query=state["original_query"], config=config
            )) for result in search_results]
            
            results = await asyncio.gather(*extraction_tasks)
            all_facts, all_sources, all_statistics = [], [], []
            for facts, sources, statistics in results:
                all_facts.extend(facts)
                all_sources.extend(sources)
                all_statistics.extend(statistics)
        
        # Update the category state
        category_state["extracted_facts"] = all_facts
        category_state["sources"] = all_sources
        category_state["statistics"] = all_statistics
        category_state["status"] = "extracted" if all_facts else "extraction_failed"
        category_state["complete"] = True
        
        # Save results to cache if category processing is complete
        if category_state.get("complete", False):
            try:
                cache_key = f"category_{state['original_query']}_{category}"
                cache_data = {
                    "search_results": category_state.get("search_results", []),
                    "extracted_facts": category_state.get("extracted_facts", []),
                    "sources": category_state.get("sources", [])
                }
                from react_agent.utils.cache import create_checkpoint
                create_checkpoint(cache_key, cache_data, ttl=86400)  # Cache for 24 hours
                info_highlight(f"Cached results for category: {category}")
            except Exception as e:
                error_highlight(f"Error caching results for {category}: {str(e)}")
        
        return {"categories": categories}

    except Exception as e:
        error_highlight(f"Error extracting facts for category {category}: {str(e)}")
        category_state["status"] = "extraction_failed"
        
    return {"categories": categories}


@step_caching()
async def extract_category_information(
    state: ResearchState, 
    content: TextContent | None = None, 
    category: str | None = None,
    extraction_model: str | None = None
) -> ResearchState:
    """Extract category information from content using the category tool."""
    global category_tool
    
    # Handle content-free operations
    if not content:
        return state
    
    # Initialize the category tool if needed
    if category_tool is None:
        model_name = extraction_model or get_extraction_model()
        category_tool = CategoryExtractionTool(extraction_model=model_name)
    
    # Find the relevant category in the state
    if not category:
        return state
    
    category_obj = next((cat for cat in state.categories if cat.category == category), None)
    if not category_obj:
        return state
    
    # Extract category information
    try:
        url = None
        if isinstance(content, WebContent):
            url = content.url
        
        # Try to use the extraction tool first
        try:
            # Check if we have access to the tools node
            if hasattr(state, "tools") and state.get("tools"):
                info_highlight(f"Using tools node for category extraction: {category}")
                
                # Use the extraction tool from the tools node
                extraction_result = await extraction_tool.ainvoke({
                    "operation": "category",
                    "text": content.content,
                    "url": url,
                    "source_title": getattr(content, "title", ""),
                    "category": category,
                    "original_query": state.get("original_query", ""),
                    "extraction_model": state.get("extraction_model")
                }, config=ensure_config(extraction_model))
                
                extracted_info = extraction_result.get("facts", [])
            else:
                # Fall back to the category tool
                info_highlight(f"Using category tool for extraction: {category}")
                extracted_info = category_tool.run(
                    text=content.content,
                    category=category,
                    url=url
                )
            
            log_dict(logger, f"Extracted category information for {category}", extracted_info)
            
            # Add extracted facts to the category
            if extracted_info:
                for fact in extracted_info:
                    # Ensure we don't add duplicates
                    if fact not in category_obj.extracted_facts:
                        category_obj.extracted_facts.append(fact)
            
            # Extract statistics using the statistics tool
            try:
                info_highlight(f"Extracting statistics for category: {category}")
                
                # Use the statistics tool
                stats_input = {
                    "operation": "synthesis",  
                    "text": content.content,
                    "url": url,
                    "source_title": getattr(content, "title", ""),
                    "synthesis": {  
                        "content": content.content,
                        "url": url,
                        "title": getattr(content, "title", "")
                    }
                }
                
                # Extract statistics
                stats_result = await statistics_tool.ainvoke(stats_input)
                if stats_result:
                    info_highlight(f"Extracted {len(stats_result)} statistics for {category}")
                    
                    # Add statistics to the category
                    for stat in stats_result:
                        if stat not in category_obj.statistics:
                            category_obj.statistics.append(stat)
            except Exception as stats_error:
                warning_highlight(f"Error extracting statistics: {str(stats_error)}")
        except Exception as tool_error:
            error_highlight(f"Error using extraction tool: {str(tool_error)}")
    
    except Exception as e:
        logger.error(f"Error extracting category information: {str(e)}")
        logger.exception(e)
    
    return state

async def execute_research_for_categories(
        state: ResearchState,
        config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Execute research for all categories in parallel with improved caching."""
    log_step("Executing research for all categories", 5, 10)

    categories = state["categories"]
    research_tasks = []
    
    # Get or initialize category success tracker
    success_tracker = getattr(state, "category_success_tracker", CategorySuccessTracker())

    # Track completed categories to avoid reprocessing
    completed_categories = set()

    for category, category_state in categories.items():
        if category_state["complete"]:
            info_highlight(f"Category {category} already complete, skipping")
            completed_categories.add(category)
            continue

        # Check checkpoint for this category
        cache_key = f"category_{state['original_query']}_{category}"
        if cached_state := load_checkpoint(cache_key):
            info_highlight(f"Using cached results for category: {category}")
            category_state["search_results"] = cached_state.get("search_results", [])
            category_state["extracted_facts"] = cached_state.get("extracted_facts", [])
            category_state["sources"] = cached_state.get("sources", [])
            category_state["complete"] = True
            completed_categories.add(category)
            continue
        
        # Check if we should skip this category based on success rate
        if success_tracker.should_skip_category(category, threshold=0.15):
            warning_highlight(f"Skipping category {category} due to low success rate: {success_tracker.success_rates.get(category, 0):.2f}")
            category_state["status"] = "skipped_low_success"
            category_state["complete"] = True
            completed_categories.add(category)
            continue
    
    # Prioritize categories based on success rates
    prioritized_categories = []
    for category, category_state in categories.items():
        if category not in completed_categories:
            priority = success_tracker.get_category_priority(category)
            prioritized_categories.append((category, priority))
    
    # Sort by priority (higher priority first)
    prioritized_categories.sort(key=lambda x: x[1], reverse=True)
    
    if prioritized_categories:
        priority_info = ", ".join([f"{cat}({pri})" for cat, pri in prioritized_categories])
        info_highlight(f"Prioritized categories: {priority_info}")
    
    # Process categories in priority order
    for category, priority in prioritized_categories:
        category_state = categories[category]
        
        # Create async task for this category
        async def process_category(cat: str) -> None:
            # Start with fallback level 0
            search_result = await execute_category_search(state, cat, fallback_level=0, config=config)
            
            # Only proceed if search was successful
            if cat not in search_result.get("categories", {}) or search_result["categories"][cat]["status"] != "searched":
                return
                
            try:
                # Try to use the tools node first for better integration with langgraph
                if "tools" in state:
                    info_highlight(f"Using tools node for category: {cat}")
                    await state["tools"].ainvoke({
                        "category": cat,
                        "state": state
                    }, config)
                else:
                    # Fall back to the original method
                    await extract_category_facts(state, cat, config)
            except Exception as e:
                error_highlight(f"Error processing category {cat}: {str(e)}")
                # Fall back to the original method
                await extract_category_facts(state, cat, config)
            
            # Save results to cache if category processing is complete
            if cat in state["categories"] and state["categories"][cat].get("complete", False):
                try:
                    cache_key = f"category_{state['original_query']}_{cat}"
                    cache_data = {
                        "search_results": state["categories"][cat].get("search_results", []),
                        "extracted_facts": state["categories"][cat].get("extracted_facts", []),
                        "sources": state["categories"][cat].get("sources", [])
                    }
                    from react_agent.utils.cache import create_checkpoint
                    create_checkpoint(cache_key, cache_data, ttl=86400)  # Cache for 24 hours
                    info_highlight(f"Cached results for category: {cat}")
                except Exception as e:
                    error_highlight(f"Error caching results for {cat}: {str(e)}")
        
        task = asyncio.create_task(process_category(category))
        research_tasks.append(task)


async def _search_and_process(
        state: ResearchState,
        category: str,
        results_queue: asyncio.Queue,
        config: Optional[RunnableConfig] = None
) -> None:
    """Search for a category and process results as they arrive.
    
    This helper function executes the search for a specific category and
    immediately processes the results, allowing for better parallelization
    and resource utilization.
    
    Args:
        state: The current research state
        category: The category to search for
        results_queue: Queue for processing results
        config: Optional runnable configuration
    """
    try:
        # Execute search with fallback support
        search_result = await execute_category_search(state, category, fallback_level=0, config=config)
        
        # Check if search was successful
        if (category not in search_result.get("categories", {}) or 
                search_result["categories"][category]["status"] != "searched"):
            warning_highlight(f"Search failed for category: {category}")
            return
            
        # Get the search results
        category_state = state["categories"][category]
        search_results = category_state.get("search_results", [])
        
        if not search_results:
            warning_highlight(f"No search results for category: {category}")
            return
            
        # Process each result immediately
        for result in search_results:
            # Add to processing queue
            await results_queue.put((category, result, state["original_query"]))
            
        # Process results and extract facts
        if "tools" in state:
            info_highlight(f"Using tools node for category: {category}")
            await state["tools"].ainvoke({
                "category": category,
                "state": state
            }, config)
        else:
            # Fall back to the original method
            await extract_category_facts(state, category, config)
            
        # Mark category as complete
        category_state["complete"] = True
        
        # Cache results
        try:
            cache_key = f"category_{state['original_query']}_{category}"
            cache_data = {
                "search_results": category_state.get("search_results", []),
                "extracted_facts": category_state.get("extracted_facts", []),
                "sources": category_state.get("sources", [])
            }
            create_checkpoint(cache_key, cache_data, ttl=86400)  # Cache for 24 hours
            info_highlight(f"Cached results for category: {category}")
        except Exception as e:
            error_highlight(f"Error caching results for {category}: {str(e)}")
            
    except Exception as e:
        error_highlight(f"Error in _search_and_process for {category}: {str(e)}")


async def process_results_queue(
        state: ResearchState,
        results_queue: asyncio.Queue,
        config: Optional[RunnableConfig] = None
) -> None:
    """Process search results from the queue as they arrive.
    
    Args:
        state: The current research state
        results_queue: Queue containing search results to process
        config: Optional runnable configuration
    """
    while True:
        try:
            # Get the next item from the queue
            item = await results_queue.get()
            
            if item is None:  # End marker
                break
                
            category, result, original_query = item
            # Process the result (implementation would go here)
            
        except Exception as e:
            error_highlight(f"Error processing result from queue: {str(e)}")


# --------------------------------------------------------------------
# 3. Synthesis and validation nodes
# --------------------------------------------------------------------
# Synthesis and validation nodes remain largely unchanged, but use the
# new utility functions where appropriate.

async def synthesize_research(
        state: ResearchState,
        config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Synthesize all research data into a comprehensive result."""
    # The implementation for this function is omitted as the core logic
    # related to synthesis remains largely unchanged. It continues to use
    # the existing utility functions and prompt templates.
    # Key parts are retained for clarity and context.
    log_step("Synthesizing research results", 7, 10)

    categories = state["categories"]
    original_query = state["original_query"]

    research_data = {
        category: {
            "facts": category_state["extracted_facts"],
            "sources": category_state["sources"],
            "quality_score": category_state["quality_score"],
            "statistics": category_state["statistics"],
            "confidence_score": category_state["confidence_score"],
            "cross_validation_score": category_state["cross_validation_score"],
            "source_quality_score": category_state["source_quality_score"],
            "recency_score": category_state["recency_score"],
            "statistical_content_score": category_state[
                "statistical_content_score"
            ],
        }
        for category, category_state in categories.items()
    }
    
    # Generate prompt using the template
    synthesis_prompt = SYNTHESIS_PROMPT.format(
        query=original_query,
        research_json=str(research_data)
    )

    try:
        synthesis_result = await call_model_json(
            messages=[{"role": "human", "content": synthesis_prompt}],
            config=ensure_config(config)
        )

        # Calculate overall confidence using the imported utility
        category_scores = {
            category: state["quality_score"]
            for category, state in categories.items()
        }

        synthesis_quality = assess_synthesis_quality(synthesis_result)
        validation_score = 0.8  # Default validation score, can be updated later

        overall_confidence = calculate_overall_confidence(
            category_scores=category_scores,
            synthesis_quality=synthesis_quality,
            validation_score=validation_score
        )

        # Add confidence assessment to synthesis result
        synthesis_result["confidence_assessment"] = {
            "overall_score": overall_confidence,
            "synthesis_quality": synthesis_quality,
            "validation_score": validation_score,
            "category_scores": category_scores
        }

        info_highlight(f"Research synthesis complete with confidence score: {overall_confidence:.2f}")

        return {
            "synthesis": synthesis_result,
            "status": "synthesized"
        }
    except Exception as e:
        error_highlight(f"Error in research synthesis: {str(e)}")
        return {"error": {"message": f"Error in research synthesis: {str(e)}", "phase": "synthesis"}}

async def validate_synthesis(
        state: ResearchState,
        config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Validate the synthesized research results."""
    log_step("Validating research synthesis", 8, 10)
    
    synthesis = state.get("synthesis")
    if not synthesis:
        warning_highlight("No synthesis to validate")
        return {"error": {"message": "No synthesis to validate", "phase": "validation"}}
    
    # Generate validation prompt
    import json
    validation_prompt = VALIDATION_PROMPT.format(
        synthesis_json=json.dumps(synthesis, indent=2)
    )
    
    try:
        validation_result = await call_model_json(
            messages=[{"role": "human", "content": validation_prompt}],
            config=ensure_config(config)
        )
        
        # Get validation status
        validation_results = validation_result.get("validation_results", {})
        is_valid = validation_results.get("is_valid", False)
        validation_score = validation_results.get("validation_score", 0.0)
        
        if is_valid:
            info_highlight(f"Validation passed with score: {validation_score:.2f}")
            return {
                "validation_result": validation_result,
                "status": "validated",
                "complete": True
            }
        else:
            warning_highlight(f"Validation failed with score: {validation_score:.2f}")
            return {
                "validation_result": validation_result,
                "status": "validation_failed"
            }
    except Exception as e:
        error_highlight(f"Error in synthesis validation: {str(e)}")
        return {"error": {"message": f"Error in synthesis validation: {str(e)}", "phase": "validation"}}

async def prepare_final_response(
        state: ResearchState,
        config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Prepare the final response with research results."""
    log_step("Preparing final response", 9, 10)
    
    synthesis = state.get("synthesis", {})
    validation = state.get("validation_result", {})
    
    # Import response formatting from prompts module
    from react_agent.prompts.synthesis import ENHANCED_REPORT_TEMPLATE
    
    try:
        # Using the defined functions in the existing codebase
        synthesis = state.get("synthesis", {})
        synthesis_content = synthesis.get("synthesis", {}) if synthesis else {}
        validation = state.get("validation_result", {})
        original_query = state["original_query"]
        
        # Format a basic response
        report_content = f"""
# Research Results: {original_query}

## Executive Summary
{synthesis_content.get("executive_summary", {}).get("content", "Research complete.")}

## Key Findings
The research found information across multiple categories including market dynamics, 
technical requirements, and implementation considerations.

## Confidence
Research confidence: {(synthesis or {}).get("confidence_assessment", {}).get("overall_score", 0.5):.2f}/1.0
Generated: {datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")}
"""
        
        return {
            "messages": [AIMessage(content=report_content)],
            "status": "complete",
            "complete": True
        }
    except Exception as e:
        error_highlight(f"Error preparing final response: {str(e)}")
        # Fallback to basic response
        return {
            "messages": [AIMessage(content=f"Research complete for query: {state['original_query']}")],
            "status": "complete",
            "complete": True
        }


# --------------------------------------------------------------------
# 4. Conditional routing functions
# --------------------------------------------------------------------

def should_request_clarification(state: ResearchState) -> Hashable:
    """Determine if clarification is needed."""
    needs_clarification = state.get("needs_clarification", False)
    has_feedback = bool(state.get("human_feedback"))

    if needs_clarification and not has_feedback:
        return "request_clarification"
    else:
        return "execute_research_for_categories"


def check_retry_or_continue(state: ResearchState) -> Hashable:
    """Determine if we should retry incomplete categories or continue."""
    categories = state["categories"]

    # Check if any categories need retry
    categories_to_retry: list[str] = []
    categories_to_retry.extend(
        category
        for category, category_state in categories.items()
        if not category_state["complete"] and category_state["retry_count"] < 3
    )
    if categories_to_retry:
        # Still have categories to retry
        info_highlight(f"Categories to retry: {categories_to_retry}")
        return "execute_research_for_categories"
    else:
        # All categories either complete or max retries reached
        info_highlight("Research complete or max retries reached for all categories")
        return "synthesize_research"


def choose_research_approach(state: ResearchState) -> Hashable:
    """Determine which research approach to use based on query complexity.
    
    Args:
        state: The current research state
        
    Returns:
        Next node to execute
    """
    # Check if we have a complex query with multiple categories
    categories = state["categories"]
    active_categories = sum(1 for cat_state in categories.values() if not cat_state["complete"])
    
    # Use progressive approach for complex queries with multiple categories
    if active_categories >= 3:
        return "execute_progressive_research"
    return "execute_research_for_categories"


def validate_or_complete(state: ResearchState) -> Hashable:
    """Determine if we should validate the synthesis or finish."""
    validation_result = state.get("validation_result", {})
    validation_results = validation_result.get("validation_results", {}) if validation_result else {}
    is_valid = validation_results.get("is_valid", False)
    
    if is_valid:
        info_highlight("Validation passed, preparing final response")
    else:
        # Not valid, but we'll still finish
        warning_highlight("Validation failed, but preparing final response anyway")
    
    return "prepare_final_response"


def handle_error_or_continue(state: ResearchState) -> Hashable:
    """Determine if we should handle an error or continue."""
    error = state.get("error")

    if error:
        error_highlight(f"Error detected: {error.get('message')} in phase {error.get('phase')}")
        return "handle_error"
    else:
        return "continue"


# --------------------------------------------------------------------
# 5. Error handling
# --------------------------------------------------------------------

async def handle_error(state: ResearchState) -> Dict[str, Any]:
    """Handle errors gracefully and return a helpful message."""
    error = state.get("error") or {}
    phase = error.get("phase", "unknown")
    message = error.get("message", "An unknown error occurred")

    error_highlight(f"Handling error in phase {phase}: {message}")
    
    # Use built-in error handling logic

    # Build error response with partial results
    error_response = [
        f"I encountered an issue while researching your query: {message}",
        "\nHere's what I was able to find before the error occurred:"
    ]

    # Add completed categories and their facts
    categories = state.get("categories", {})
    complete_categories = [
        cat
        for cat, state in categories.items()
        if state.get("complete", False)
    ]
    
    if complete_categories:
        error_response.append(f"\n\nI completed research on: {', '.join(complete_categories)}")

        for category in complete_categories:
            facts = categories[category].get("extracted_facts", [])
            if facts:
                error_response.append(f"\n\n## {category.replace('_', ' ').title()}")
                for fact in facts[:3]:  # Show top 3 facts
                    fact_text = _format_fact(fact)
                    if fact_text:
                        error_response.append(fact_text)
    else:
        error_response.append(
            "\n\nUnfortunately, I wasn't able to complete any research categories before the error occurred.")

    error_response.append("\n\nWould you like me to try again with a more specific query?")

    return {
        "messages": [AIMessage(content="".join(error_response))],
        "status": "error",
        "complete": True
    }

def _format_fact(fact: Dict[str, Any]) -> Optional[str]:
    """Format a single fact into a readable string."""
    if not isinstance(fact, dict):
        return None
        
    data = fact.get("data", {})
    fact_text = (
        data.get("fact") or
        data.get("requirement") or
        next((v for v in data.values() if isinstance(v, str) and v), None) or
        fact.get("fact")
    )
    return f"\n- {fact_text}" if fact_text else None

# --------------------------------------------------------------------
# 6. Build the graph
# --------------------------------------------------------------------

def create_research_graph() -> CompiledStateGraph:
    """Create the modular research graph."""
    graph = StateGraph(ResearchState)

    # Create a tool node for extraction and validation tools
    # Initialize tools node function
    async def initialize_tools(state: ResearchState) -> Dict[str, Any]:
        """Initialize the tools node with extraction tools."""
        log_step("Initializing tools node", 2, 10)
        
        # Initialize the extraction model
        extraction_model = get_extraction_model()
        
        # Initialize category tool if needed
        global category_tool
        if category_tool is None:
            model_name = extraction_model or get_extraction_model()
            category_tool = CategoryExtractionTool(extraction_model=model_name)
        
        # Initialize the extraction tool if needed
        global extraction_tool
        if extraction_tool is None:
            extraction_tool = ExtractionTool()
            info_highlight("Initialized extraction tool")
        
        # Initialize the statistics analysis tool if needed
        global statistics_tool
        if statistics_tool is None:
            statistics_tool = StatisticsAnalysisTool()
            info_highlight("Initialized statistics analysis tool")
        
        # Add the extraction model and tools to the state 
        # The tools node will use these for extraction operations
        return {
            "extraction_model": extraction_model,
            "tools": {
                "category": category_tool,
                "extraction": extraction_tool,
                "statistics": statistics_tool
            }
        }
    
    # Create the tools node
    tools_node = create_derivation_toolnode(
        include_tools=["extraction", "statistics", "validation"]
    )
    
    # Add the tools node to the graph
    # This allows for more efficient tool usage and better integration with langgraph
    graph.add_node(
        "tools", 
        tools_node, 
        metadata={"description": "Node for executing extraction and validation tools"}
    )

    # Add the main nodes
    graph.add_node("initialize", initialize_research)
    graph.add_node("initialize_tools", initialize_tools)
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("request_clarification", request_clarification)
    graph.add_node("process_clarification", process_clarification)
    graph.add_node("execute_research_for_categories", execute_research_for_categories)
    graph.add_node("synthesize_research", synthesize_research)
    graph.add_node("validate_synthesis", validate_synthesis)  # Keep validation for now
    graph.add_node("prepare_final_response", prepare_final_response)
    graph.add_node("handle_error", handle_error)
    graph.add_node("check_retry", lambda state: {"next_step": check_retry_or_continue(state)})
    graph.add_node("validate_or_complete", lambda state: {"next_step": validate_or_complete(state)})

    # Add error handling edges
    graph.add_conditional_edges(
        "initialize",
        handle_error_or_continue,
        {
            "handle_error": "handle_error",
            "continue": "initialize_tools"
        }
    )

    # Add the START node connection - This is the entry point of the graph
    graph.add_edge(START, "initialize")

    # Set start point
    graph.add_edge("initialize_tools", "analyze_query")
    graph.add_edge("initialize", "analyze_query")  # Changed: Directly to analyze_query
    
    # Connect initialize to tools node
    graph.add_edge("initialize", "tools")

    # Add conditional branch for clarification from analyze_query
    graph.add_conditional_edges(
        "analyze_query",
        lambda state: "request_clarification" if state.get("needs_clarification", False) and not state.get("human_feedback") else "choose_research_approach",
        {
            "request_clarification": "request_clarification",
            "choose_research_approach": "choose_research_approach"
            
        }
    )

    graph.add_edge("request_clarification", "process_clarification")
    graph.add_edge("process_clarification", "analyze_query")  # Loop back to analysis

    # Research flow
    graph.add_conditional_edges(
        "execute_research_for_categories",
        handle_error_or_continue,
        {
            "handle_error": "handle_error",
            "continue": "check_retry"
        }
    )

    # Add progressive research node
    graph.add_node(
        "execute_progressive_research",
        execute_progressive_search
    )

    # Add node to choose research approach
    graph.add_node(
        "choose_research_approach",
        lambda state: {"next_step": choose_research_approach(state)}
    )
    
    # Connect to appropriate research method
    graph.add_conditional_edges(
        "choose_research_approach",
        lambda state: state.get("next_step", "execute_research_for_categories"),
        {
            "execute_research_for_categories": "execute_research_for_categories",
            "execute_progressive_research": "execute_progressive_research"
        }
    )

    # Connect research flow to tools node
    graph.add_edge("execute_research_for_categories", "tools")
    
    # Connect tools node back to research flow
    graph.add_edge("tools", "execute_research_for_categories")
    graph.add_edge("tools", "synthesize_research")

    graph.add_conditional_edges(
        "check_retry",
        lambda state: state.get("next_step", "synthesize_research"),
        {
            "execute_research_for_categories": "execute_research_for_categories",
            "synthesize_research": "synthesize_research"
        }
    )

    # Connect progressive research to check_retry
    graph.add_edge(
        "execute_progressive_research",
        "check_retry"
    )

    # Synthesis and validation (keep validation for now)
    graph.add_conditional_edges(
        "synthesize_research",
        handle_error_or_continue,
        {
            "handle_error": "handle_error",
            "continue": "validate_synthesis" # Keep validation step
        }
    )
    
    graph.add_conditional_edges(
        "validate_synthesis",
        handle_error_or_continue,
        {
            "handle_error": "handle_error",
            "continue": "validate_or_complete"  # Keep validation
        }
    )
    
    # Connect synthesis to tools node for validation
    graph.add_edge("synthesize_research", "tools")
    
    graph.add_conditional_edges(
        "validate_or_complete",
        lambda state: state.get("next_step", "prepare_final_response"),
        {
            "prepare_final_response": "prepare_final_response"
        }
    )

    # Final steps
    graph.add_edge("prepare_final_response", END)
    graph.add_edge("handle_error", END)

    # Create a store for cross-thread data persistence
    # Removed store initialization as it's an abstract class

    # Compile the graph with checkpointer and store
    return graph.compile(
        checkpointer=MemorySaver()
    )

# Create the graph instance
research_graph = create_research_graph()

@step_caching()
async def add_statistics_to_research_state(
    state: ResearchState, content: TextContent | None = None, category_title: str | None = None
) -> ResearchState:
    """Add statistics from content to the research state."""
    source_title = ""

    # Handle content-free operations
    if not content:
        return state

    url = None
    if isinstance(content, WebContent):
        url = content.url
        # Update source_title if title exists
        if hasattr(content, "title") and content.title:
            source_title = content.title

    # Extract statistics from content using the statistics tool
    extracted_statistics = []
    try:
        # Try to use the statistics analysis tool first
        try:
            # Check if we have access to the tools node
            if hasattr(state, "tools") and state.tools:
                info_highlight(f"Using statistics analysis tool for extraction")
                
                # Use the statistics analysis tool with the correct operation
                analysis_result = await statistics_tool.ainvoke({
                    "operation": "statistics",  # Add required operation parameter
                    "text": content.content,
                    "url": url,
                    "source_title": source_title
                })
                
                extracted_statistics = analysis_result
            else:
                # Fall back to the original statistics tool
                info_highlight(f"Using original statistics tool for extraction")
                extracted_statistics = statistics_tool.run(tool_input={
                    "operation": "statistics",  # Add required operation parameter
                    "text": content.content, 
                    "url": url, 
                    "source_title": source_title
                })
        except Exception as tool_error:
            warning_highlight(f"Error using statistics analysis tool: {str(tool_error)}")
            # Fall back to the original statistics tool
            extracted_statistics = statistics_tool.run(tool_input={
                "operation": "statistics",  # Add required operation parameter
                "text": content.content,
                "url": url,
                "source_title": source_title
            })
        log_dict(logger, "Extracted statistics", extracted_statistics)
    except Exception as e:
        logger.error(f"Error extracting statistics: {str(e)}")
        logger.exception(e)

    # Check if we need to update specific category or all categories
    if category_title:
        categories_to_update = [
            cat for cat in state.categories if cat.category == category_title
        ]
        if not categories_to_update:
            return state
    else:
        categories_to_update = state.categories

    # Update statistics for each category
    for category in categories_to_update:
        for statistic in extracted_statistics:
            if statistic not in category.statistics:
                category.statistics.append(statistic)

    return state

# Add this at the top of the file where other async functions are defined
async def extract_category_info(
    content: str,
    url: str,
    title: str,
    category: str,
    original_query: str,
    prompt_template: str,
    extraction_model: Callable,
    config: Optional[RunnableConfig] = None
) -> Tuple[List[Dict[str, Any]], float]:
    """Shim function to use the deprecated extract_category_information function.
    
    Will be updated to use the new CategoryExtractionTool directly in future.
    This version now uses the ExtractionTool directly for better integration with langgraph."""
    global category_tool
    
    # Initialize the category tool if needed
    if category_tool is None:
        model_name = get_extraction_model()
        category_tool = CategoryExtractionTool(extraction_model=model_name)
    
    try:
        # Try to use the extraction tool directly first
        try:
            info_highlight(f"Using extraction tool for category: {category}")
            result = await extraction_tool.ainvoke({
                "operation": "category",
                "text": content,
                "url": url,
                "source_title": title,
                "category": category,
                "original_query": original_query,
                "prompt_template": prompt_template,
                "extraction_model": extraction_model
            }, config)
            
            return result.get("facts", []), result.get("relevance", 0.5)
        except Exception as tool_error:
            # Fall back to the original method if the new one fails
            warning_highlight(f"Extraction tool failed, falling back to original method: {str(tool_error)}")
            facts, relevance = await original_extract_category_info(
                content=content, url=url, title=title, 
                category=category, original_query=original_query,
                prompt_template=prompt_template, 
                extraction_model=extraction_model, config=config
            )
            return facts, relevance
    except Exception as e:
        logger.error(f"Error extracting category information: {str(e)}")
        logger.exception(e)
        return [], 0.0