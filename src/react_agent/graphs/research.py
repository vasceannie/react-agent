"""Enhanced modular research framework using LangGraph.

This module implements a modular approach to the research process,
with specialized components for different research categories and
improved error handling and validation.
"""

from __future__ import annotations
import json
import asyncio
from typing import Any, Dict, List, Optional, Sequence, Union, cast, Tuple, Literal, Hashable
from datetime import datetime, timezone
from urllib.parse import urlparse

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.constants import START
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_core.documents import Document
from typing_extensions import Annotated, TypedDict

from react_agent.utils.validations import is_valid_url
from react_agent.utils.llm import call_model, call_model_json
from react_agent.tools.jina import search, optimize_query
from react_agent.prompts.research import (
    QUERY_ANALYSIS_PROMPT,
    CLARIFICATION_PROMPT,
    EXTRACTION_PROMPTS,
    SYNTHESIS_PROMPT,
    VALIDATION_PROMPT,
    SEARCH_QUALITY_THRESHOLDS,
    get_extraction_prompt
)
from react_agent.utils.logging import get_logger, log_dict, info_highlight, warning_highlight, error_highlight, log_step

# Initialize logger
logger = get_logger(__name__)

# Define SearchType as a Literal type
SearchType = Literal['general', 'authoritative', 'recent', 'comprehensive', 'technical']

# --------------------------------------------------------------------
# 1. Define the modular state classes
# --------------------------------------------------------------------

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
# 2. Core control flow nodes
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

        log_dict(analysis_result, title="Query Analysis Result")

        # Check for missing context
        missing_context = analysis_result.get("missing_context", [])
        needs_clarification = len(missing_context) >= 2  # Only request clarification for multiple missing elements

        # Update category queries based on analysis
        categories = state["categories"]
        for category, category_state in categories.items():
            search_terms = analysis_result.get("search_terms", {}).get(category, [])
            if search_terms:
                # Create optimized query for this category
                is_higher_ed = "higher ed" in query.lower() or "education" in query.lower()
                optimized_query = optimize_query(query, category, is_higher_ed)
                category_state["query"] = optimized_query
                info_highlight(f"Set query for {category}: {optimized_query}")
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
    
    # Prepare the clarification request
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
    
    return {
        "human_feedback": clarification_content,
        "needs_clarification": False,
        "missing_context": [],
        "status": "clarified"
    }

# --------------------------------------------------------------------
# 3. Module-specific research nodes
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
    recency_days = int(thresholds.get("recency_threshold_days", 365))
    
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
        
        # Execute the search
        search_results = await search(
            query=query,
            search_type=search_type,
            recency_days=recency_days,
            config=ensure_config(config)
        )
        
        if not search_results:
            warning_highlight(f"No search results for {category}")
            category_state["status"] = "search_failed"
            return {"categories": categories}
        
        # Convert to the expected format
        formatted_results = []
        for doc in search_results:
            result = {
                "url": doc.metadata.get("url", ""),
                "title": doc.metadata.get("title", ""),
                "snippet": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "quality_score": doc.metadata.get("quality_score", 0.5),
                "published_date": doc.metadata.get("published_date")
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

async def extract_category_information(
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
    original_query = state["original_query"]
    
    if not search_results:
        warning_highlight(f"No search results to extract for {category}")
        category_state["status"] = "extraction_failed"
        return {"categories": categories}
    
    # Update status
    category_state["status"] = "extracting"
    
    # Extract facts from each search result
    extracted_facts = []
    sources = []
    
    for idx, result in enumerate(search_results):
        url = result.get("url", "")
        content = result.get("snippet", "")
        
        if not url or not content or not is_valid_url(url):
            continue
        
        info_highlight(f"Extracting from {url} for {category}")
        
        try:
            # Get the appropriate extraction prompt for this category
            extraction_prompt = get_extraction_prompt(
                category=category,
                query=original_query,
                url=url,
                content=content
            )
            
            # Call the model for extraction
            extraction_result = await call_model_json(
                messages=[{"role": "human", "content": extraction_prompt}],
                config=ensure_config(config)
            )
            
            # Validate the extraction result minimally to ensure it has the expected structure
            if not extraction_result:
                warning_highlight(f"Empty extraction result for {url}")
                continue
                
            # Get relevance score
            relevance_score = extraction_result.get("relevance_score", 0.0)
            
            # Only include if relevant
            if relevance_score < 0.3:
                info_highlight(f"Low relevance ({relevance_score}) for {url}, skipping")
                continue
            
            # Get extracted facts based on category
            if category == "market_dynamics":
                facts = extraction_result.get("extracted_facts", [])
                
            elif category == "provider_landscape":
                facts = []
                vendors = extraction_result.get("extracted_vendors", [])
                if vendors:
                    facts.extend([
                        {"type": "vendor", "data": vendor} for vendor in vendors
                    ])
                
                relationships = extraction_result.get("vendor_relationships", [])
                if relationships:
                    facts.extend([
                        {"type": "relationship", "data": rel} for rel in relationships
                    ])
                
            elif category == "technical_requirements":
                facts = []
                requirements = extraction_result.get("extracted_requirements", [])
                if requirements:
                    facts.extend([
                        {"type": "requirement", "data": req} for req in requirements
                    ])
                
                standards = extraction_result.get("standards", [])
                if standards:
                    facts.extend([
                        {"type": "standard", "data": std} for std in standards
                    ])
                
            elif category == "regulatory_landscape":
                facts = []
                regulations = extraction_result.get("extracted_regulations", [])
                if regulations:
                    facts.extend([
                        {"type": "regulation", "data": reg} for reg in regulations
                    ])
                
                compliance = extraction_result.get("compliance_requirements", [])
                if compliance:
                    facts.extend([
                        {"type": "compliance", "data": req} for req in compliance
                    ])
                
            elif category == "cost_considerations":
                facts = []
                costs = extraction_result.get("extracted_costs", [])
                if costs:
                    facts.extend([
                        {"type": "cost", "data": cost} for cost in costs
                    ])
                
                models = extraction_result.get("pricing_models", [])
                if models:
                    facts.extend([
                        {"type": "pricing_model", "data": model} for model in models
                    ])
                
            elif category == "best_practices":
                facts = []
                practices = extraction_result.get("extracted_practices", [])
                if practices:
                    facts.extend([
                        {"type": "practice", "data": practice} for practice in practices
                    ])
                
                methods = extraction_result.get("methodologies", [])
                if methods:
                    facts.extend([
                        {"type": "methodology", "data": method} for method in methods
                    ])
                
            elif category == "implementation_factors":
                facts = []
                factors = extraction_result.get("extracted_factors", [])
                if factors:
                    facts.extend([
                        {"type": "factor", "data": factor} for factor in factors
                    ])
                
                challenges = extraction_result.get("challenges", [])
                if challenges:
                    facts.extend([
                        {"type": "challenge", "data": challenge} for challenge in challenges
                    ])
            else:
                # Default for unknown categories
                facts = extraction_result.get("extracted_facts", [])
            
            # Add source information to each fact
            for fact in facts:
                fact["source_url"] = url
                fact["source_title"] = result.get("title", "")
                
            # Only add source and facts if we extracted something
            if facts:
                extracted_facts.extend(facts)
                
                # Add source metadata
                source = {
                    "url": url,
                    "title": result.get("title", ""),
                    "published_date": result.get("published_date"),
                    "fact_count": len(facts),
                    "relevance_score": relevance_score,
                    "quality_score": result.get("quality_score", 0.5)
                }
                sources.append(source)
                
                info_highlight(f"Extracted {len(facts)} facts from {url}")
            else:
                info_highlight(f"No relevant facts found in {url}")
                
        except Exception as e:
            warning_highlight(f"Error extracting from {url}: {str(e)}")
            continue
    
    # Update the category state
    category_state["extracted_facts"] = extracted_facts
    category_state["sources"] = sources
    
    # Determine if extraction was successful
    thresholds = SEARCH_QUALITY_THRESHOLDS.get(category, {})
    min_facts = thresholds.get("min_facts", 3)
    min_sources = thresholds.get("min_sources", 2)
    
    if len(extracted_facts) >= min_facts and len(sources) >= min_sources:
        category_state["status"] = "extracted"
        category_state["complete"] = True
        
        # Calculate quality score for this category
        quality_score = calculate_category_quality_score(
            category=category,
            extracted_facts=extracted_facts,
            sources=sources,
            thresholds=thresholds
        )
        category_state["quality_score"] = quality_score
        
        info_highlight(f"Successfully extracted information for {category} (quality: {quality_score:.2f})")
    else:
        category_state["status"] = "extraction_incomplete"
        info_highlight(f"Incomplete extraction for {category}: {len(extracted_facts)} facts from {len(sources)} sources")
        
        # If we've retried too many times, mark as complete anyway
        if category_state["retry_count"] >= 3:
            category_state["complete"] = True
            warning_highlight(f"Maximum retries reached for {category}, accepting partial results")
    
    return {"categories": categories}

# Helper function to calculate quality score for a category
def calculate_category_quality_score(
    category: str,
    extracted_facts: List[Dict[str, Any]],
    sources: List[Dict[str, Any]],
    thresholds: Dict[str, Any]
) -> float:
    """Calculate quality score for a category based on extracted data."""
    # Base score starts at 0.3
    score = 0.3
    
    # Add points for number of facts
    min_facts = thresholds.get("min_facts", 3)
    fact_ratio = min(1.0, len(extracted_facts) / (min_facts * 2))
    score += fact_ratio * 0.2
    
    # Add points for number of sources
    min_sources = thresholds.get("min_sources", 2)
    source_ratio = min(1.0, len(sources) / (min_sources * 1.5))
    score += source_ratio * 0.2
    
    # Add points for authoritative sources
    auth_ratio = thresholds.get("authoritative_source_ratio", 0.5)
    authoritative_sources = [
        s for s in sources 
        if s.get("quality_score", 0.0) > 0.7 or
        any(domain in s.get("url", "") for domain in ['.edu', '.gov', '.org'])
    ]
    auth_source_ratio = len(authoritative_sources) / len(sources) if sources else 0
    if auth_source_ratio >= auth_ratio:
        score += 0.2
    else:
        score += (auth_source_ratio / auth_ratio) * 0.1
    
    # Add points for recency
    recency_threshold = thresholds.get("recency_threshold_days", 365)
    recent_sources = 0
    for source in sources:
        date_str = source.get("published_date")
        if not date_str:
            continue
            
        try:
            from dateutil import parser
            from datetime import datetime, timezone
            date = parser.parse(date_str)
            now = datetime.now(timezone.utc)
            days_old = (now - date).days
            if days_old <= recency_threshold:
                recent_sources += 1
        except Exception:
            pass
    
    if sources:
        recency_ratio = recent_sources / len(sources)
        score += recency_ratio * 0.1
    
    return min(1.0, score)

async def execute_research_for_categories(
    state: ResearchState,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Execute research for all categories in parallel."""
    log_step("Executing research for all categories", 5, 10)
    
    categories = state["categories"]
    research_tasks = []
    
    for category, category_state in categories.items():
        if category_state["complete"]:
            info_highlight(f"Category {category} already complete, skipping")
            continue
            
        # Execute search followed by extraction
        info_highlight(f"Adding research task for {category}")
        
        # Create async task for this category
        async def process_category(cat: str) -> None:
            search_result = await execute_category_search(state, cat, config)
            # Only extract if search was successful
            if cat in search_result.get("categories", {}) and search_result["categories"][cat]["status"] == "searched":
                await extract_category_information(state, cat, config)
        
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
# 4. Synthesis and validation nodes
# --------------------------------------------------------------------

async def synthesize_research(
    state: ResearchState,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Synthesize all research data into a comprehensive result."""
    log_step("Synthesizing research results", 7, 10)
    
    categories = state["categories"]
    original_query = state["original_query"]
    
    # Prepare research data for synthesis
    research_data = {}
    for category, category_state in categories.items():
        research_data[category] = {
            "facts": category_state["extracted_facts"],
            "sources": category_state["sources"],
            "quality_score": category_state["quality_score"]
        }
    
    # Generate prompt
    synthesis_prompt = SYNTHESIS_PROMPT.format(
        query=original_query,
        research_json=json.dumps(research_data, indent=2)
    )
    
    try:
        synthesis_result = await call_model_json(
            messages=[{"role": "human", "content": synthesis_prompt}],
            config=ensure_config(config)
        )
        
        log_dict(
            {
                "synthesis_sections": list(synthesis_result.get("synthesis", {}).keys()),
                "confidence_score": synthesis_result.get("confidence_assessment", {}).get("overall_score")
            },
            title="Synthesis Overview"
        )
        
        overall_score = synthesis_result.get("confidence_assessment", {}).get("overall_score", 0.0)
        info_highlight(f"Research synthesis complete with confidence score: {overall_score:.2f}")
        
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
        
        # Log validation results
        log_dict(
            {
                "is_valid": is_valid,
                "validation_score": validation_score,
                "critical_issues": validation_results.get("critical_issues", [])
            },
            title="Validation Results"
        )
        
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
    """Prepare the final response with the research results."""
    log_step("Preparing final response", 9, 10)
    
    synthesis = state.get("synthesis", {})
    validation = state.get("validation_result", {})
    original_query = state["original_query"]
    
    # Create a summary message for the user
    synthesis_content = synthesis.get("synthesis", {}) if synthesis else {}
    confidence = synthesis.get("confidence_assessment", {}) if synthesis else {}
    
    # Build an executive summary
    executive_summary = "# Research Results: " + original_query + "\n\n"
    
    # Add confidence information
    confidence_score = confidence.get("overall_score", 0.0)
    executive_summary += f"**Confidence Score:** {confidence_score:.2f}/1.0\n\n"
    
    # Add key findings from most complete sections
    executive_summary += "## Key Findings\n\n"
    
    # Get the most complete sections
    section_scores = confidence.get("section_scores", {})
    sorted_sections = sorted(
        section_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Add content from top 3 sections
    for section_name, score in sorted_sections[:3]:
        if section_name in synthesis_content and synthesis_content[section_name]["content"]:
            section_content = synthesis_content[section_name]["content"]
            executive_summary += f"### {section_name.replace('_', ' ').title()}\n\n"
            executive_summary += f"{section_content}\n\n"
    
    # Add limitations
    knowledge_gaps = confidence.get("knowledge_gaps", [])
    if knowledge_gaps:
        executive_summary += "## Limitations\n\n"
        for gap in knowledge_gaps:
            executive_summary += f"- {gap}\n"
    
    # Create the response message
    response_message = AIMessage(content=executive_summary)
    
    return {
        "messages": [response_message],
        "status": "complete",
        "complete": True
    }

# --------------------------------------------------------------------
# 5. Conditional routing functions
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
    categories_to_retry = []
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
    if is_valid := validation_results.get("is_valid", False):
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
# 6. Error handling
# --------------------------------------------------------------------

async def handle_error(state: ResearchState) -> Dict[str, Any]:
    """Handle errors gracefully and return a helpful message."""
    error = state.get("error", {})
    phase = error.get("phase", "unknown") if error else "unknown"
    message = error.get("message", "An unknown error occurred") if error else "An unknown error occurred"

    error_highlight(f"Handling error in phase {phase}: {message}")

    # Create a helpful error message
    error_response = f"I encountered an issue while researching your query: {message}"
    error_response += "\n\nHere's what I was able to find before the error occurred:"

    # Include any partial results we have
    categories = state.get("categories", {})
    if complete_categories := [
        category
        for category, category_state in categories.items()
        if category_state.get("complete", False)
    ]:
        error_response += f"\n\nI completed research on: {', '.join(complete_categories)}"

        # Include facts from complete categories
        for category in complete_categories:
            category_state = categories[category]
            facts = category_state.get("extracted_facts", [])
            if facts:
                error_response += f"\n\n## {category.replace('_', ' ').title()}\n"
                for i, fact in enumerate(facts[:3]):  # Show up to 3 facts
                    if isinstance(fact, dict):
                        if "data" in fact and isinstance(fact["data"], dict):
                            # Handle structured facts
                            if "fact" in fact["data"]:
                                error_response += f"\n- {fact['data']['fact']}"
                            elif "requirement" in fact["data"]:
                                error_response += f"\n- {fact['data']['requirement']}"
                            elif "vendor_name" in fact["data"]:
                                error_response += f"\n- {fact['data']['vendor_name']}: {fact['data'].get('description', '')}"
                            else:
                                # Just get the first string value we can find
                                for k, v in fact["data"].items():
                                    if isinstance(v, str) and v:
                                        error_response += f"\n- {v}"
                                        break
                        elif "fact" in fact:
                            error_response += f"\n- {fact['fact']}"
    else:
        error_response += "\n\nUnfortunately, I wasn't able to complete any research categories before the error occurred."

    error_response += "\n\nWould you like me to try again with a more specific query?"

    # Create the error message
    error_message = AIMessage(content=error_response)

    return {
        "messages": [error_message],
        "status": "error",
        "complete": True
    }

# --------------------------------------------------------------------
# 7. Build the graph
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
    graph.add_node("validate_synthesis", validate_synthesis)
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
    graph.add_edge(START, "initialize")

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
    graph.add_edge("process_clarification", "analyze_query")

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

    # Synthesis and validation
    graph.add_conditional_edges(
        "synthesize_research",
        handle_error_or_continue,
        {
            "handle_error": "handle_error",
            "continue": "validate_synthesis"
        }
    )

    graph.add_conditional_edges(
        "validate_synthesis",
        handle_error_or_continue,
        {
            "handle_error": "handle_error",
            "continue": "validate_or_complete"
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

    return graph.compile(interrupt_before=["process_clarification"])

# Create the graph instance
research_graph = create_research_graph()

# def create_category_graph(category: str) -> CompiledStateGraph:
#     """Create a specialized graph for a single research category."""
#     graph = StateGraph(ResearchState)
    
#     # Add nodes for single category research
#     graph.add_node("initialize", initialize_research)
#     graph.add_node("analyze_query", analyze_query)
#     graph.add_node("execute_category_search", lambda state, config: execute_category_search(state, category, config))
#     graph.add_node("extract_category_information", lambda state, config: extract_category_information(state, category, config))
#     graph.add_node("prepare_final_response", prepare_final_response)
#     graph.add_node("handle_error", handle_error)
    
#     # Add edges
#     graph.add_edge(START, "initialize")
#     graph.add_edge("initialize", "analyze_query")
#     graph.add_edge("analyze_query", "execute_category_search")
#     graph.add_edge("execute_category_search", "extract_category_information")
#     graph.add_edge("extract_category_information", "prepare_final_response")
#     graph.add_edge("prepare_final_response", END)
#     graph.add_edge("handle_error", END)
    
#     return graph.compile()