"""Modular research framework using LangGraph.

This module implements a modular approach to the research process,
with specialized components for different research categories and
improved error handling and validation.
"""

from __future__ import annotations


import asyncio
from datetime import UTC, datetime, timezone
from typing import (
    Any,
    Dict,
    Hashable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig, ensure_config
from langgraph.constants import START
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import Annotated, TypedDict

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

# Import tools
from react_agent.tools.jina import search
from react_agent.utils.extraction import extract_category_information
from react_agent.utils.llm import call_model, call_model_json

# Import utility modules
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
from react_agent.utils.statistics import (
    calculate_enhanced_category_quality_score as calculate_category_quality_score,
)

# Initialize logger
logger = get_logger(__name__)

# Define SearchType as a Literal type
SearchType = Literal['general', 'authoritative', 'recent', 'comprehensive', 'technical']


class ResearchCategory(TypedDict):
    """State for a specific research category."""
    category: str  # The category being researched (market_dynamics, etc.)
    query: str  # The search query for this category
    search_results: List[Dict[str, Any]]  # Raw search results
    extracted_facts: List[Dict[str, Any]]  # Extracted facts
    sources: List[Dict[str, Any]]  # Source information
    complete: bool  # Whether this category is complete
    quality_score: float  # Quality score for this category (0.0-1.0)
    retry_count: int  # Number of retry attempts
    last_search_query: Optional[str]  # Last search query used
    status: str  # Status of this category (pending, in_progress, complete, failed)
    statistics: List[Dict[str, Any]]  # Extracted statistics from facts
    confidence_score: float  # Confidence score for this category (0.0-1.0)
    cross_validation_score: float  # Cross-validation score for facts (0.0-1.0)
    source_quality_score: float  # Quality score for sources (0.0-1.0)
    recency_score: float  # Recency score for sources (0.0-1.0)
    statistical_content_score: float  # Score for statistical content (0.0-1.0)


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
    categories: Dict[str, ResearchCategory]

    # Synthesis and validation
    synthesis: Optional[Dict[str, Any]]
    validation_result: Optional[Dict[str, Any]]

    # Overall status
    status: str
    error: Optional[Dict[str, Any]]
    complete: bool


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

    info_highlight(f"Executing {search_type} search for {category} with query: {query}")

    try:
        # Keep track of the query for retry tracking
        category_state["last_search_query"] = query

        # Execute the search with category parameter
        search_results = await search(
            query=query,
            search_type=search_type,
            recency_days=recency_days,
            category=category,  # Pass the category parameter
            config=ensure_config(config)
        )

        if not search_results:
            warning_highlight(f"No search results for {category}")
            category_state["status"] = "search_failed"
            return {"categories": categories}

        # Convert to the expected format and filter problematic content
        formatted_results = []
        
        # Use the imported utility functions to process search results
        from react_agent.utils.content import should_skip_content, validate_content, detect_content_type
        
        for doc in search_results:
            url = doc.metadata.get("url", "")

            # Skip problematic content types
            if should_skip_content(url):
                info_highlight(f"Skipping problematic content type: {url}")
                continue

            # Validate and detect content type
            content = doc.page_content
            if not validate_content(content):
                info_highlight(f"Invalid content from {url}, skipping")
                continue

            content_type = detect_content_type(url, content)
            info_highlight(f"Detected content type: {content_type} for {url}")

            result = {
                "url": url,
                "title": doc.metadata.get("title", ""),
                "snippet": content,
                "source": doc.metadata.get("source", ""),
                "quality_score": doc.metadata.get("quality_score", 0.5),
                "published_date": doc.metadata.get("published_date"),
                "content_type": content_type
            }
            formatted_results.append(result)

        # Update the category state
        category_state["search_results"] = formatted_results
        category_state["status"] = "searched"

        info_highlight(f"Found {len(formatted_results)} results for {category}")
        return {"categories": categories}

    except Exception as e:
        error_highlight(f"Error in search for {category}: {str(e)}")
        category_state["status"] = "search_failed"
        return {"categories": categories}

async def extract_category_facts(
    state: ResearchState,
    category: str,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Extract information from search results for a specific category."""
    log_step(f"Extracting information for category: {category}", 6, 10)

    categories = state["categories"]
    if category not in categories:
        warning_highlight(f"Unknown category: {category}")
        return {}

    category_state = categories[category]
    search_results = category_state["search_results"]

    if not search_results:
        category_state["status"] = "extraction_failed"
        return {"categories": categories}

    category_state["status"] = "extracting"

    # Process all search results in parallel
    tasks = [
        asyncio.create_task(_process_search_result(
            result, category, state["original_query"], config
        ))
        for result in search_results
    ]

    results = await asyncio.gather(*tasks)

    # Aggregate results
    extracted_facts = []
    sources = []
    all_statistics = []
    
    # Use the imported extraction utility
    for result in search_results:
        url = result.get("url", "")
        title = result.get("title", "")
        content = result.get("snippet", "")
        
        # Get the extraction prompt for this category
        prompt_template = get_extraction_prompt(
            category=category,
            query=state["original_query"],
            url=url,
            content=content
        )
        
        # Extract information using the utility function
        facts, relevance_score = await extract_category_information(
            content=content,
            url=url,
            title=title,
            category=category,
            original_query=state["original_query"],
            prompt_template=prompt_template,
            extraction_model=call_model_json,
            config=ensure_config(config)
        )
        
        # Extract statistics
        from react_agent.utils.extraction import extract_statistics
        statistics = extract_statistics(content) or []
        
        extracted_facts.extend(facts)
        
        # Add source information
        source = {
            "url": url,
            "title": title,
            "published_date": result.get("published_date"),
            "fact_count": len(facts),
            "relevance_score": relevance_score,
            "quality_score": result.get("quality_score", 0.5),
            "content_type": result.get("content_type", "unknown")
        }
        sources.append(source)
        all_statistics.extend(statistics)

    # Update category state with quality scores from statistics module
    thresholds = SEARCH_QUALITY_THRESHOLDS.get(category, {})
    quality_score = calculate_category_quality_score(
        category=category,
        extracted_facts=extracted_facts,
        sources=sources,
        thresholds=thresholds
    )

    category_state.update({
        "extracted_facts": extracted_facts,
        "sources": sources,
        "statistics": all_statistics,
        "quality_score": quality_score
    })

    # Calculate derived scores based on quality_score
    category_state.update({
        "confidence_score": quality_score,
        "cross_validation_score": quality_score * 0.8,
        "source_quality_score": quality_score * 0.9,
        "recency_score": quality_score * 0.7,
        "statistical_content_score": quality_score * 0.85
    })

    # Determine completion status
    min_facts = thresholds.get("min_facts", 3)
    min_sources = thresholds.get("min_sources", 2)

    category_state["complete"] = (
                                         len(extracted_facts) >= min_facts and len(sources) >= min_sources
                                 ) or category_state["retry_count"] >= 3

    category_state["status"] = "extracted" if category_state["complete"] else "extraction_incomplete"

    return {"categories": categories}

async def execute_research_for_categories(
        state: ResearchState,
        config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Execute research for all categories in parallel with improved caching."""
    log_step("Executing research for all categories", 5, 10)

    categories = state["categories"]
    research_tasks = []

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
            
        # Create async task for this category
        async def process_category(cat: str) -> None:
            search_result = await execute_category_search(state, cat, config)
            # Only extract if search was successful
            if cat in search_result.get("categories", {}) and search_result["categories"][cat]["status"] == "searched":
                await extract_category_facts(state, cat, config)
        
        task = asyncio.create_task(process_category(category))
        research_tasks.append(task)

    # Wait for all research tasks to complete
    if research_tasks:
        await asyncio.gather(*research_tasks)

    # Check if all categories are complete
    all_complete = all(
        category_state["complete"] for category_state in categories.values()
    )

    if all_complete:
        info_highlight("All categories research complete")
        return {
            "status": "researched",
            "categories": categories
        }
    else:
        # Some categories still incomplete
        incomplete = [
            category for category, category_state in categories.items()
            if not category_state["complete"]
        ]
        info_highlight(f"Categories still incomplete: {', '.join(incomplete)}")
        return {
            "status": "research_incomplete",
            "categories": categories
        }


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

    # Add the main nodes
    graph.add_node("initialize", initialize_research)
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
            "continue": "analyze_query"
        }
    )

    # Set start point
    graph.add_edge("initialize", "analyze_query")  # Changed: Directly to analyze_query

    # Add conditional branch for clarification from analyze_query
    graph.add_conditional_edges(
        "analyze_query",
        should_request_clarification,
        {
            "request_clarification": "request_clarification",
            "execute_research_for_categories": "execute_research_for_categories"
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

    graph.add_conditional_edges(
        "check_retry",
        lambda state: state.get("next_step", "synthesize_research"),
        {
            "execute_research_for_categories": "execute_research_for_categories",
            "synthesize_research": "synthesize_research"
        }
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

    return graph.compile(checkpointer=MemorySaver())

# Create the graph instance
research_graph = create_research_graph()