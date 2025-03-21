"""
Combined Research Flow with Typed State, Human-in-the-Loop, and Activity Nodes
-----------------------------------------------------------------------------
This module demonstrates a unified approach that merges the flow logic from
the first snippet with the typed state and ephemeral (human-input) pattern
from the second snippet.
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Literal, Sequence, Annotated, Tuple, Union
from typing_extensions import TypedDict
from datetime import datetime
from langgraph.constants import START, END
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_core.documents import Document

from react_agent.utils.validations import is_valid_url
from react_agent.utils.llm import call_model, call_model_json
from react_agent.tools.jina import search as jina_search
from react_agent.prompts import (
    CLARIFICATION_PROMPT,
    QUERY_ANALYSIS_PROMPT,
    RESEARCH_AGENT_PROMPT,
    RESEARCH_BASE_PROMPT,
)
from react_agent.utils.logging import get_logger, log_dict, info_highlight, warning_highlight, error_highlight, log_step

# Initialize logger
logger = get_logger(__name__)

# --------------------------------------------------------------------
# 1. Define merge strategies for state updates
# --------------------------------------------------------------------
def replace_search_results(existing: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replace existing search results with new ones."""
    return new

def merge_urls(existing: List[str], new: List[str]) -> List[str]:
    """Merge URL lists, removing duplicates."""
    return list(set(existing + new)) if existing else new

def append_status(existing: List[str], new: List[str]) -> List[str]:
    """Append new status updates to existing ones."""
    return existing + new if existing else new

def merge_extracted_info(existing: Dict[str, List[Any]], new: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """Merge extracted information, combining lists for matching keys."""
    result = existing.copy()
    for key, values in new.items():
        if key in result:
            result[key].extend(values)
        else:
            result[key] = values
    return result

# --------------------------------------------------------------------
# 2. Define the typed state with merge strategies
# --------------------------------------------------------------------
class ResearchState(TypedDict):
    """A typed state capturing all necessary data for the research flow."""
    # Basic conversation data with message handling
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Original query and enriched query
    original_query: str
    enriched_query: Optional[Dict[str, Any]]

    # Flags/fields for missing context and clarifications
    missing_context: List[str]
    needs_clarification: bool
    clarification_request: Optional[str]
    human_feedback: Optional[str]

    # Search-phase data with merge strategies
    search_results: Annotated[List[Dict[str, Any]], replace_search_results]
    visited_urls: Annotated[List[str], merge_urls]
    
    # Extraction-phase data with merge strategies
    extracted_info: Annotated[Dict[str, List[Any]], merge_extracted_info]
    sources: List[Dict[str, Any]]

    # Synthesis-phase data
    synthesis: Optional[Dict[str, Any]]
    synthesis_complete: bool

    # Validation-phase data
    confidence_score: float
    validation_status: Dict[str, Any]
    validation_passed: bool

    # Additional contextual fields
    industry_context: Optional[str]
    search_priority: List[str]

    # Status updates with append strategy
    status_updates: Annotated[List[str], append_status]

    # Quality control and feedback fields
    search_quality_score: float
    extraction_quality_score: float
    search_retry_count: int
    extraction_retry_count: int
    last_search_query: Optional[str]
    last_extraction_query: Optional[str]
    search_feedback: Optional[Dict[str, Any]]
    extraction_feedback: Optional[Dict[str, Any]]
    extraction_status: Optional[str]
    extraction_message: Optional[str]
    extraction_attempt: int

    # Search strategy flags
    targeted_search: bool
    alternate_search_strategy: bool
    alternate_search: bool
    alternate_extraction: bool

# --------------------------------------------------------------------
# 2. Utility function to push status updates to the user (UI, logs, etc.)
# --------------------------------------------------------------------
def push_status_to_user(status_message: str) -> None:
    """Send status updates to the user interface (e.g., via WebSocket or SSE)."""
    print(f"[STATUS UPDATE] {status_message}")
    # Implement your real UI push here if desired.


# --------------------------------------------------------------------
# 3. Node (step) definitions. Below, we adapt your original logic
#    to use the typed ResearchState and the ephemeral/HITL pattern.
# --------------------------------------------------------------------

async def initialize(state: ResearchState) -> Dict[str, Any]:
    """Initialize the research flow with the user's query."""
    log_step("Initializing research flow", 1, 7)
    
    if not state["messages"]:
        warning_highlight("No messages found in state")
        return {}
    
    last_message = state["messages"][-1]
    info_highlight("Research process initiated")

    return {
        "messages": state["messages"],
        "original_query": last_message.content if isinstance(last_message, BaseMessage) else ""
    }


async def enrich_query_node(state: ResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Analyze and enrich the user's query using UNSPSC taxonomy categories
    instead of requesting overly specific details.
    """
    log_step("Enriching query", 2, 10)

    query = state["original_query"].strip()
    if not query:
        warning_highlight("No query found to enrich")
        return {}
    
    # Check for feedback from previous iterations
    human_feedback = state.get("human_feedback", "")
    if human_feedback:
        info_highlight(f"Incorporating user feedback: {human_feedback}")
        query = f"{query} [Additional context: {human_feedback}]"
    
    info_highlight(f"Analyzing query: {query}")
    
    # Use UNSPSC taxonomy categories for enrichment
    enrichment_prompt = f"""
    Analyze this research query and identify relevant procurement categories based on UNSPSC taxonomy:
    
    QUERY: {query}
    
    1. Identify the top 3 most relevant UNSPSC categories (Segment, Family, or Class level)
    2. Generate search terms for market analysis in each category
    3. Determine if this is about products, services, or both
    4. Identify if any geographical focus is mentioned or implied
    
    Return a JSON object with:
    {{
        "unspsc_categories": [list of categories with code and name],
        "search_terms": {{
            "market_dynamics": [],
            "key_suppliers": [],
            "specifications": [],
            "regulations": []
        }},
        "primary_keywords": [],
        "product_vs_service": "",
        "geographical_focus": "",
        "missing_context": []
    }}
    
    Note: Don't require excessive specific details for the search - use available information
    to make reasonable assumptions where possible. Only mark as "missing_context" if it's
    absolutely critical for meaningful research.
    """
    
    data = await call_model_json(
        messages=[{"role": "human", "content": enrichment_prompt}],
        config=ensure_config(config)
    )

    # Log the enrichment results
    log_dict(data, title="Query Enrichment Results Using UNSPSC Taxonomy")

    # Extract search terms from structured analysis
    search_terms = []
    if "search_terms" in data:
        for category, terms in data["search_terms"].items():
            if isinstance(terms, list):
                search_terms.extend(terms)
                info_highlight(f"Added {len(terms)} terms from {category}")

    primary_keywords = data.get("primary_keywords", [])
    all_terms = list(set(search_terms + primary_keywords))
    enhanced_query = " OR ".join([term for term in all_terms if term]) if all_terms else query

    # Check for missing context - but be more lenient
    missing_context = data.get("missing_context", [])
    # Only consider clarification if multiple critical elements are missing
    needs_clarification = len(missing_context) > 2

    # Build final enriched query object
    enriched_query = {
        "enhanced_query": enhanced_query,
        "unspsc_categories": data.get("unspsc_categories", []),
        "product_vs_service": data.get("product_vs_service", ""),
        "geographical_focus": data.get("geographical_focus", ""),
        "primary_keywords": primary_keywords,
        "missing_context": missing_context,
        "original_query": query,
        "search_terms": data.get("search_terms", {})
    }

    info_highlight(
        f"Query enriched with {len(primary_keywords)} primary keywords and {len(search_terms)} search terms. "
        f"Using UNSPSC taxonomy for categorization."
    )

    return {
        "enriched_query": enriched_query,
        "product_vs_service": data.get("product_vs_service", ""),
        "geographical_focus": data.get("geographical_focus", ""),
        "missing_context": missing_context,
        "needs_clarification": needs_clarification
    }

async def request_clarification(state: ResearchState) -> Dict[str, Any]:
    """
    Human-in-the-loop node that interrupts execution to request user clarification.
    Uses LangGraph's interrupt mechanism to pause execution and wait for user input.
    """
    log_step("Requesting clarification", 3, 10)  # Updated total steps

    # Format missing sections for the prompt
    missing_sections = "\n".join([f"- {section}" for section in state["missing_context"]])
    
    # Get enriched query data safely
    enriched_query = state.get("enriched_query") or {}
    primary_keywords = enriched_query.get("primary_keywords", [])
    
    # Use the structured clarification prompt - use double curly braces to escape
    try:
        # First verify the template string
        clarif_template = CLARIFICATION_PROMPT
        # Ensure template uses proper format syntax (single braces)
        if "{{" in clarif_template:
            clarif_template = clarif_template.replace("{{", "{").replace("}}", "}")
            
        # Format the template with actual values
        clarif_msg = clarif_template.format(
            query=state["original_query"],
            industry_context=state.get("industry_context", "Unknown industry"),
            primary_keywords=", ".join(primary_keywords),
            missing_sections=missing_sections
        )
        
        info_highlight(f"Generated clarification request")
    except Exception as e:
        error_highlight(f"Error formatting clarification message: {str(e)}")
        # Fallback to direct message if template formatting fails
        clarif_msg = f"I need additional information about your query: '{state['original_query']}'\n\n"
        clarif_msg += f"Specifically, I need details about:\n{missing_sections}\n\n"
        clarif_msg += "This will help me provide more accurate and relevant research results."

    return {
        "clarification_request": clarif_msg,
        "__interrupt__": {
            "value": {
                "question": clarif_msg,
                "missing_context": state["missing_context"]
            },
            "resumable": True,
            "ns": ["request_clarification"],
            "when": "during"
        }
    }


async def process_clarification(state: ResearchState) -> Dict[str, Any]:
    """
    Process user-provided clarification and incorporate it as feedback
    for the next research iteration.
    """
    log_step("Processing clarification", 4, 10)

    if not state["messages"]:
        warning_highlight("No messages found for clarification")
        return {}
    
    last_message = state["messages"][-1]
    clarification_content = str(last_message.content).strip()
    
    # Store as human feedback that can be used in subsequent steps
    info_highlight(f"Storing user feedback: {clarification_content}")
    
    # Don't append [Clarification: ] to the query to avoid repetition
    return {
        "messages": state["messages"],
        "original_query": state["original_query"],  # Keep original query
        "human_feedback": clarification_content,  # Store feedback separately
        "needs_clarification": False,
        "missing_context": []
    }

def extract_key_terms_from_results(search_results: List[Document]) -> List[str]:
    """Extract important terms from initial search results to guide further searches."""
    if not search_results:
        return []
        
    # Extract text from search results
    all_text = " ".join(doc.page_content for doc in search_results if doc.page_content)
    
    # Initialize term groups
    term_groups = []
    
    # Look for capitalized phrases which often indicate specific terms
    import re
    cap_phrases = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', all_text)
    if cap_phrases:
        term_groups.append(" ".join(cap_phrases[:3]))  # Use top 3 capitalized phrases
        
    # Extract terms following "including", "such as", etc.
    list_items = []
    list_matches = re.findall(r'(?:include|including|such as|like|e\.g\.|i\.e\.|namely|for example)[:]?\s+([^.;!?]+)', all_text)
    for match in list_matches:
        items = re.split(r',\s+|\s+and\s+|\s+or\s+', match)
        list_items.extend([item.strip() for item in items if len(item.strip()) > 3])
    
    if list_items:
        term_groups.append(" ".join(list_items[:3]))
        
    return term_groups

async def perform_search(state: ResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Performs search with progressive fallback strategies if results are insufficient."""
    log_step("Performing search with fallback strategies", 5, 10)

    if not state["enriched_query"]:
        warning_highlight("No enriched query available for search")
        return {"search_results": []}

    eq = state["enriched_query"]
    enhanced_query = eq["enhanced_query"]
    original_query = eq["original_query"]

    try:
        # First attempt: Try enhanced query
        info_highlight(f"Attempting search with enhanced query: {enhanced_query}")
        results = await jina_search(enhanced_query, config=ensure_config(config))
        
        # Check if results are sufficient
        if results and len(results) >= 3:
            info_highlight(f"Enhanced query successful with {len(results)} results")
        else:
            # Fallback 1: Try original query
            warning_highlight("Insufficient results with enhanced query, trying original query")
            results = await jina_search(original_query, config=ensure_config(config))
            
            # Check if original query got results
            if results and len(results) >= 2:
                info_highlight(f"Original query successful with {len(results)} results")
            else:
                # Fallback 2: Try simplified query - just use primary keywords
                primary_keywords = eq.get("primary_keywords", [])
                if primary_keywords:
                    simplified_query = " ".join(primary_keywords)
                    warning_highlight(f"Trying simplified query: {simplified_query}")
                    results = await jina_search(simplified_query, config=ensure_config(config))
                    
                    # If still insufficient, try very basic query
                    if not results or len(results) < 2:
                        # Fallback 3: Try broadest possible query
                        broadest_term = original_query.split()[0] if " " in original_query else original_query
                        warning_highlight(f"Trying broadest possible query: {broadest_term}")
                        results = await jina_search(broadest_term, config=ensure_config(config))
    except Exception as e:
        warning_highlight(f"Error with search, using fallback: {str(e)}")
        try:
            results = await jina_search(original_query, config=ensure_config(config))
            info_highlight(f"Fallback search completed. Processing {len(results) if results else 0} results")
        except Exception as e:
            error_highlight(f"Search failed completely: {str(e)}")
            return {"search_results": []}

    if not results:
        warning_highlight("No valid search results found after multiple attempts")
        return {"search_results": []}

    # Convert Document objects to the expected format
    formatted_results = []
    try:
        formatted_results.extend(
            {
                "url": doc.metadata.get("url", ""),
                "title": doc.metadata.get("title", ""),
                "snippet": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "published_date": doc.metadata.get("published_date"),
            }
            for doc in results
        )
        info_highlight(f"Successfully formatted {len(formatted_results)} search results")
        
        # Log sample of formatted results
        if formatted_results:
            log_dict(
                formatted_results[0],
                title="Sample Formatted Result"
            )
    except Exception as e:
        error_highlight(f"Error formatting results: {str(e)}")
        return {"search_results": []}

    return {"search_results": formatted_results}

async def process_search_results(state: ResearchState) -> Dict[str, Any]:
    """
    Prioritize search results and pick top documents. 
    Same as 'process_search_results' from snippet #1, adapted to typed state.
    """
    log_step("Processing search results", 6, 7)

    search_results = state["search_results"]
    visited_urls = set(state["visited_urls"] or [])
    if not search_results:
        warning_highlight("No search results to process")
        return {}

    info_highlight("Scoring and ranking search results")
    query = state["original_query"].lower()
    keywords = query.split()

    scored_results = []
    for result in search_results:
        url = result.get("url", "")
        text = f"{result.get('title','')} {result.get('snippet','')}".lower()

        score = sum(kw in text for kw in keywords)
        # Small boost for certain domains
        if any(d in url for d in ["wikipedia.org", ".gov", ".edu"]):
            score += 2
            info_highlight(f"Applied domain boost for {url}")

        # Penalize visited
        if url in visited_urls:
            score -= 10
            info_highlight(f"Applied visited penalty for {url}")

        scored_results.append((score, result))

    sorted_results = sorted(scored_results, key=lambda x: x[0], reverse=True)
    top_results = []
    newly_visited = []

    for score, doc in sorted_results[:3]:
        url = doc.get("url")
        if url and url not in visited_urls:
            top_results.append(doc)
            newly_visited.append(url)
            info_highlight(f"Selected result: {url} (score: {score})")

    info_highlight(f"Selected {len(top_results)} most relevant results")
    return {
        "search_results": top_results,
        "visited_urls": list(visited_urls.union(newly_visited))
    }

def verify_extraction_result(extraction_result: Dict[str, Any], original_content: str) -> bool:
    """
    Verify that extracted results are based on actual content and not fabricated.
    """
    if not isinstance(extraction_result, dict):
        return False
        
    # Check for expected structure
    if "extracted_facts" not in extraction_result:
        return False
        
    # Verify that extracted facts exist in original content
    facts = extraction_result.get("extracted_facts", [])
    if not facts:
        return True  # No facts to verify
        
    for fact_item in facts:
        if not isinstance(fact_item, dict):
            continue
            
        source_text = fact_item.get("source_text", "")
        if not source_text or source_text not in original_content:
            return False
    
    # Check for unrealistic confidence scores
    confidence_score = extraction_result.get("confidence_score", 0.0)
    if confidence_score > 0.9 and len(original_content) < 500:
        return False  # Suspicious high confidence on small content
        
    return True

def calculate_data_quality(valid_urls: int, invalid_urls: int, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate overall data quality score based on source validity and extraction results.
    """
    total_urls = valid_urls + invalid_urls
    
    if total_urls == 0:
        return {"score": 0.0, "reason": "No sources processed"}
        
    # Calculate base score from valid URL ratio
    base_score = valid_urls / total_urls if total_urls > 0 else 0
    
    # Check extraction comprehensiveness
    fact_count = len(extracted_info.get("extracted_facts", []))
    has_market_data = bool(extracted_info.get("market_data", {}).get("items", []))
    has_vendor_info = bool(extracted_info.get("vendor_info", {}).get("items", []))
    has_specifications = bool(extracted_info.get("specifications", {}).get("items", []))
    
    # Adjust score based on content richness
    content_score = min(1.0, (fact_count / 10) + 0.1 * sum([has_market_data, has_vendor_info, has_specifications]))
    
    # Calculate final score
    final_score = min(0.95, (base_score * 0.6) + (content_score * 0.4))
    
    reason = "High quality data" if final_score > 0.7 else \
             "Moderate quality data" if final_score > 0.4 else \
             "Low quality data - insufficient verified information"
             
    return {
        "score": round(final_score, 2),
        "reason": reason,
        "stats": {
            "valid_urls": valid_urls,
            "invalid_urls": invalid_urls,
            "fact_count": fact_count,
            "has_market_data": has_market_data,
            "has_vendor_info": has_vendor_info,
            "has_specifications": has_specifications
        }
    }


async def extract_key_information(state: ResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Extract key info with better error handling and minimum data requirements."""
    log_step("Extracting key information", 7, 10)

    # Get search results with default empty list if None
    search_results = state["search_results"] or []
    
    if not search_results:
        warning_highlight("No search results to extract from")
        return {
            "extracted_info": {},
            "sources": [],
            "extraction_status": "failed",
            "extraction_message": "No search results available"
        }

    info_highlight(f"Processing {len(search_results)} documents for information extraction")
    extracted_info = {}
    sources = []
    extraction_count = 0

    try:
        for idx, doc in enumerate(search_results, 1):
            if not isinstance(doc, dict):
                warning_highlight(f"Skipping invalid document format at index {idx}")
                continue
                
            url = doc.get("url", "")
            content = doc.get("snippet", "") or doc.get("content", "") or ""
            
            if not url or not content or len(content.strip()) < 50:
                warning_highlight(f"Skipping document with insufficient content at index {idx}")
                continue
                
            info_highlight(f"Extracting information from document {idx}: {url}")
            
            try:
                # Enhanced extraction prompt with strict guidelines
                extraction_prompt = f"""
                Extract factual information from this content about {state.get('original_query', 'the research topic')}.
                
                URL: {url}
                
                CRITICAL INSTRUCTIONS:
                1. Only extract VERIFIED facts explicitly stated in the content
                2. Format each fact with:
                   - The fact statement
                   - Direct quote from the content supporting the fact
                   - Category label (overview, standards, providers, specs, costs, etc.)
                3. If the document doesn't contain relevant information, indicate this
                
                CONTENT:
                {content}
                
                FORMAT YOUR RESPONSE AS JSON:
                {{
                  "extracted_facts": [
                    {{
                      "fact": "Clear factual statement",
                      "source_text": "Direct quote from content",
                      "category": "Category label"
                    }}
                  ],
                  "relevance_score": 0.0  // 0-1 score of how relevant this content is
                }}
                """
                
                extraction_result = await call_model_json(
                    messages=[{
                        "role": "human", 
                        "content": extraction_prompt
                    }],
                    config=ensure_config(config)
                )
                
                # Check if any facts were extracted
                facts = extraction_result.get("extracted_facts", [])
                if not facts:
                    warning_highlight(f"No relevant facts extracted from document {idx}")
                    continue
                
                # Add each fact to appropriate category
                for fact in facts:
                    category = fact.get("category", "general")
                    if category not in extracted_info:
                        extracted_info[category] = []
                    
                    # Include source URL with each fact
                    fact["source_url"] = url
                    extracted_info[category].append(fact)
                
                # Track sources
                sources.append({
                    "url": url,
                    "title": doc.get("title", ""),
                    "fact_count": len(facts),
                    "categories": list(set(fact.get("category", "general") for fact in facts))
                })
                
                extraction_count += 1
                info_highlight(f"Successfully extracted {len(facts)} facts from document {idx}")
                
            except Exception as e:
                warning_highlight(f"Failed to process document {idx}: {str(e)}")
                continue

        # Check if we have minimum viable data
        total_facts = sum(len(facts) for facts in extracted_info.values())
        
        if extraction_count == 0 or total_facts == 0:
            warning_highlight("CRITICAL: No useful information extracted from any documents")
            return {
                "extracted_info": {},
                "sources": [],
                "extraction_status": "failed",
                "extraction_message": "No useful information could be extracted"
            }
        
        info_highlight(f"Successfully extracted {total_facts} facts from {extraction_count} sources")
        return {
            "extracted_info": extracted_info,
            "sources": sources,
            "extraction_status": "success",
            "extraction_message": f"Extracted {total_facts} facts from {extraction_count} sources"
        }

    except Exception as e:
        error_highlight(f"Error during information extraction: {str(e)}")
        return {
            "extracted_info": {},
            "sources": [],
            "extraction_status": "error",
            "extraction_message": f"Error: {str(e)}"
        }

def verify_extraction_quality(extraction_result, original_content):
    """Verify that extracted information is genuinely from the content."""
    # Check if the extraction result is a dictionary
    if not isinstance(extraction_result, dict):
        return {"passed": False, "reason": "Extraction result is not a dictionary"}
    
    # Check if any categories have content
    has_content = False
    for category, items in extraction_result.items():
        if isinstance(items, list) and items:
            has_content = True
            break
    
    if not has_content:
        return {"passed": False, "reason": "No content extracted from any category"}
    
    # Verify source references are present in original content
    for category, items in extraction_result.items():
        if not isinstance(items, list):
            continue
            
        for item in items:
            if not isinstance(item, dict):
                continue
                
            source_text = item.get("source_text", "")
            # Check if source text is in the original content
            if source_text and source_text not in original_content:
                return {"passed": False, "reason": f"Source text not found in original content: {source_text[:50]}..."}
    
    return {"passed": True}

def deduplicate_extracted_info(extracted_info):
    """Remove duplicate or very similar information."""
    result = {category: [] for category in extracted_info}
    
    for category, items in extracted_info.items():
        seen_texts = set()
        for item in items:
            # Create a simplified version of the text for comparison
            text = item.get("text", "").lower()
            simplified = ' '.join([word for word in text.split() if len(word) > 3])
            
            # Check if we've seen something very similar
            is_duplicate = False
            for seen in seen_texts:
                # Simple similarity check
                if len(set(simplified.split()) & set(seen.split())) / max(len(simplified.split()), len(seen.split())) > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_texts.add(simplified)
                result[category].append(item)
    
    return result

async def aggregate_research_findings(state: ResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Aggregate research findings into a comprehensive report with structured categories and enhanced validation.
    Ensures complete coverage of all required aspects and maintains high quality standards.
    """
    log_step("Aggregating research findings into final report", 9, 10)
    
    synthesis = state.get("synthesis", {}) or {}
    validation_status = synthesis.get("validation_status", {})
    confidence_score = synthesis.get("confidence_score", 0.0)
    
    if not synthesis or confidence_score < 0.4:
        warning_highlight("Insufficient synthesis quality for aggregation")
        return {
            "report": None,
            "aggregation_complete": False,
            "validation_status": {
                "is_valid": False,
                "issues": ["Insufficient synthesis quality"]
            }
        }
    
    info_highlight("Starting research aggregation")
    
    try:
        # Prepare the aggregation prompt with clear structure
        aggregation_prompt = {
            "role": "human",
            "content": f"""
            Create a comprehensive research report based on the synthesized findings.
            
            REPORT STRUCTURE:
            1. Executive Summary
               - Key findings and recommendations
               - Confidence assessment
               - Coverage overview
            
            2. Detailed Analysis
               {json.dumps(synthesis.get("analysis", {}), indent=2)}
            
            3. Market Analysis
               - Market dynamics and trends
               - Competitive landscape
               - Growth opportunities
            
            4. Technical Assessment
               - Requirements and specifications
               - Implementation considerations
               - Risk factors
            
            5. Cost Analysis
               - Market basket analysis
               - Budget considerations
               - ROI factors
            
            6. Recommendations
               - Strategic recommendations
               - Implementation roadmap
               - Risk mitigation strategies
            
            7. Appendices
               - Detailed market basket
               - Source citations
               - Data quality assessment
            
            VALIDATION STATUS:
            {json.dumps(validation_status, indent=2)}
            
            CONFIDENCE SCORE: {confidence_score}
            
            FORMAT RESPONSE AS JSON:
            {{
              "report": {{
                "executive_summary": {{
                  "key_findings": [],
                  "recommendations": [],
                  "confidence_assessment": {{
                    "score": 0.0,
                    "factors": []
                  }}
                }},
                "detailed_analysis": {{
                  "sections": [],
                  "coverage": {{
                    "complete": [],
                    "partial": [],
                    "missing": []
                  }}
                }},
                "market_analysis": {{
                  "dynamics": "",
                  "landscape": "",
                  "opportunities": []
                }},
                "technical_assessment": {{
                  "requirements": [],
                  "implementation": [],
                  "risks": []
                }},
                "cost_analysis": {{
                  "market_basket_summary": "",
                  "budget_factors": [],
                  "roi_considerations": []
                }},
                "recommendations": {{
                  "strategic": [],
                  "implementation": [],
                  "risk_mitigation": []
                }},
                "appendices": {{
                  "market_basket": [],
                  "citations": [],
                  "quality_assessment": {{
                    "score": 0.0,
                    "factors": []
                  }}
                }}
              }},
              "metadata": {{
                "generated_at": "",
                "version": "2.0",
                "confidence_score": 0.0,
                "coverage_score": 0.0
              }}
            }}
            """
        }
        
        # Generate the report
        report_response = await call_model_json(
            messages=[aggregation_prompt],
            config=ensure_config(config)
        )
        
        # Log the report generation
        log_dict(
            report_response,
            title="Research Report Generation"
        )
        
        # Validate the report
        report_validation = validate_report_quality(report_response)
        
        # Update metadata
        report_response["metadata"].update({
            "generated_at": datetime.now().isoformat(),
            "confidence_score": confidence_score,
            "coverage_score": validation_status.get("completeness", {}).get("coverage_percentage", 0.0)
        })
        
        info_highlight(
            f"Report generation complete with confidence score: {confidence_score}"
        )
        
        return {
            "report": report_response,
            "aggregation_complete": True,
            "validation_status": report_validation
        }
        
    except Exception as e:
        error_highlight(f"Error during report aggregation: {str(e)}")
        return {
            "report": None,
            "aggregation_complete": False,
            "validation_status": {
                "is_valid": False,
                "issues": [f"Aggregation error: {str(e)}"]
            }
        }

def validate_report_quality(report_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the quality and completeness of the generated report.
    """
    validation = {
        "is_valid": True,
        "issues": [],
        "quality_metrics": {}
    }
    
    report = report_response.get("report", {})
    
    # Check executive summary
    exec_summary = report.get("executive_summary", {})
    if not exec_summary.get("key_findings"):
        validation["issues"].append("Missing key findings in executive summary")
    if not exec_summary.get("recommendations"):
        validation["issues"].append("Missing recommendations in executive summary")
        
    # Check detailed analysis
    analysis = report.get("detailed_analysis", {})
    if not analysis.get("sections"):
        validation["issues"].append("Missing detailed analysis sections")
    
    # Check market analysis
    market = report.get("market_analysis", {})
    if not all([market.get("dynamics"), market.get("landscape"), market.get("opportunities")]):
        validation["issues"].append("Incomplete market analysis")
    
    # Check technical assessment
    tech = report.get("technical_assessment", {})
    if not all([tech.get("requirements"), tech.get("implementation"), tech.get("risks")]):
        validation["issues"].append("Incomplete technical assessment")
    
    # Check recommendations
    recommendations = report.get("recommendations", {})
    if not all([
        recommendations.get("strategic"),
        recommendations.get("implementation"),
        recommendations.get("risk_mitigation")
    ]):
        validation["issues"].append("Missing recommendations")
    
    # Check appendices
    appendices = report.get("appendices", {})
    if not appendices.get("citations"):
        validation["issues"].append("Missing citations in appendices")
    
    # Calculate quality metrics
    section_scores = {
        "executive_summary": 1.0 if not any("executive summary" in issue for issue in validation["issues"]) else 0.5,
        "detailed_analysis": 1.0 if not any("detailed analysis" in issue for issue in validation["issues"]) else 0.5,
        "market_analysis": 1.0 if not any("market analysis" in issue for issue in validation["issues"]) else 0.5,
        "technical_assessment": 1.0 if not any("technical assessment" in issue for issue in validation["issues"]) else 0.5,
        "recommendations": 1.0 if not any("recommendations" in issue for issue in validation["issues"]) else 0.5
    }
    
    validation["quality_metrics"] = {
        "section_scores": section_scores,
        "overall_score": sum(section_scores.values()) / len(section_scores)
    }
    
    validation["is_valid"] = len(validation["issues"]) == 0
    
    return validation

async def synthesize_research(state: ResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Synthesize research with structured requirements to ensure comprehensive coverage regardless of domain.
    Uses extracted information to generate insights with citation tracking.
    """
    log_step("Synthesizing comprehensive research findings", 8, 10)

    extracted_info = state.get("extracted_info", {})
    sources = state.get("sources", [])
    
    if not extracted_info or not any(extracted_info.values()):
        warning_highlight("No extracted information available for synthesis")
        return {
            "synthesis": {
                "analysis": {},
                "market_basket": [],
                "confidence_score": 0.0,
                "validation_status": {"is_valid": False, "issues": ["No data to synthesize"]}
            },
            "synthesis_complete": False,
            "confidence_score": 0.0,
            "validation_status": {"is_valid": False, "issues": ["No data to synthesize"]}
        }

    info_highlight("Starting comprehensive research synthesis")
    
    # Structure the synthesis requirements based on domain-agnostic categories
    synthesis_categories = [
        "domain_overview",          # Overview of the specific domain/industry
        "market_dynamics",          # Market trends, size, growth rate
        "regulatory_landscape",     # Regulations, standards, compliance requirements
        "best_practices",          # Industry best practices and methodologies
        "provider_landscape",       # Key vendors, suppliers, service providers
        "technical_requirements",   # Technical specifications, compatibility
        "implementation_factors",   # Resources, timeline, constraints
        "cost_considerations"       # Pricing models, TCO, budget considerations
    ]
    
    # Map our extracted information to these synthesis categories
    category_mapping = {
        "industry_overview": "domain_overview",
        "standards_regulations": "regulatory_landscape",
        "best_practices": "best_practices",
        "key_providers": "provider_landscape",
        "product_specifications": "technical_requirements",
        "implementation_considerations": "implementation_factors",
        "cost_factors": "cost_considerations"
    }
    
    # Prepare the synthesis prompt with clear expectations
    try:
        synthesis_prompt = {
            "role": "human",
            "content": f"""
            Create a comprehensive synthesis of research findings for: {state.get('original_query', 'the requested topic')}
            
            REQUIREMENTS:
            1. Create a structured analysis covering ALL of the following categories:
               - Domain Overview: Key characteristics and context
               - Market Dynamics: Trends, growth rates, market size
               - Regulatory Landscape: Compliance requirements, standards
               - Best Practices: Methodologies, recommended approaches
               - Provider Landscape: Key suppliers, vendors, service providers
               - Technical Requirements: Specifications, compatibility
               - Implementation Factors: Resources, processes, constraints
               - Cost Considerations: Pricing models, budget factors
            
            2. For EACH category:
               - Synthesize insights from multiple sources where available
               - Identify consistencies and discrepancies across sources
               - Include citation links to source URLs for every fact
               - If information is missing for any category, explicitly note that
            
            3. For market basket items (if available):
               - Only include items with verified information
               - Include manufacturer, item details, and pricing
               - Provide citation for each market basket item
            
            CONTEXT:
            - Original query: {state.get('original_query', '')}
            - Domain context: {state.get('product_vs_service', '')}
            - Geographical focus: {state.get('geographical_focus', '')}
            
            AVAILABLE INFORMATION:
            {json.dumps(extracted_info, indent=2)}
            
            SOURCES:
            {json.dumps(sources, indent=2)}
            
            FORMAT RESPONSE AS JSON:
            {{
              "analysis": {{
                "domain_overview": {{ "content": "", "citations": [] }},
                "market_dynamics": {{ "content": "", "citations": [] }},
                "regulatory_landscape": {{ "content": "", "citations": [] }},
                "best_practices": {{ "content": "", "citations": [] }},
                "provider_landscape": {{ "content": "", "citations": [] }},
                "technical_requirements": {{ "content": "", "citations": [] }},
                "implementation_factors": {{ "content": "", "citations": [] }},
                "cost_considerations": {{ "content": "", "citations": [] }}
              }},
              "market_basket": [
                {{
                  "manufacturer": "",
                  "item_number": "",
                  "item_description": "",
                  "uom": "",
                  "estimated_qty": 0,
                  "unit_cost": 0,
                  "citation": ""
                }}
              ],
              "confidence_score": 0.0,
              "validation_status": {{
                "is_valid": false,
                "issues": [],
                "completeness": {{
                  "complete_categories": [],
                  "incomplete_categories": [],
                  "missing_categories": []
                }}
              }}
            }}
            """
        }
        
        synthesis_response = await call_model_json(
            messages=[synthesis_prompt],
            config=ensure_config(config)
        )
        
        # Log synthesis results
        log_dict(
            synthesis_response,
            title="Comprehensive Research Synthesis"
        )
        
        # Evaluate synthesis quality
        coverage_analysis = evaluate_synthesis_coverage(synthesis_response, synthesis_categories)
        confidence_score = calculate_synthesis_confidence(synthesis_response, coverage_analysis)
        
        # Update validation status
        validation_status = synthesis_response.get("validation_status", {})
        validation_status["completeness"] = coverage_analysis
        
        is_valid = (confidence_score >= 0.7 and 
                   len(coverage_analysis.get("complete_categories", [])) >= 5 and
                   len(coverage_analysis.get("missing_categories", [])) <= 1)
        
        validation_status["is_valid"] = is_valid
        
        # Put everything together
        final_synthesis = {
            "analysis": synthesis_response.get("analysis", {}),
            "market_basket": synthesis_response.get("market_basket", []),
            "confidence_score": confidence_score,
            "validation_status": validation_status
        }
        
        info_highlight(f"Synthesis complete with confidence score: {confidence_score}")
        info_highlight(f"Complete categories: {len(coverage_analysis.get('complete_categories', []))}, "
                     f"Incomplete: {len(coverage_analysis.get('incomplete_categories', []))}, "
                     f"Missing: {len(coverage_analysis.get('missing_categories', []))}")
        
        return {
            "synthesis": final_synthesis,
            "synthesis_complete": True,
            "confidence_score": confidence_score,
            "validation_status": validation_status
        }
        
    except Exception as e:
        error_highlight(f"Error during synthesis: {str(e)}")
        return {
            "synthesis": {
                "analysis": {},
                "market_basket": [],
                "confidence_score": 0.0,
                "validation_status": {"is_valid": False, "issues": [f"Synthesis error: {str(e)}"]}
            },
            "synthesis_complete": False,
            "confidence_score": 0.0,
            "validation_status": {"is_valid": False, "issues": [f"Synthesis error: {str(e)}"]}
        }

def evaluate_synthesis_coverage(synthesis, required_categories):
    """Evaluate how well the synthesis covers the required categories."""
    complete_categories = []
    incomplete_categories = []
    missing_categories = []
    
    analysis = synthesis.get("analysis", {})
    
    for category in required_categories:
        category_data = analysis.get(category, {})
        content = category_data.get("content", "")
        citations = category_data.get("citations", [])
        
        if not content:
            missing_categories.append(category)
        elif len(content) > 200 and citations:
            complete_categories.append(category)
        else:
            incomplete_categories.append(category)
    
    return {
        "complete_categories": complete_categories,
        "incomplete_categories": incomplete_categories,
        "missing_categories": missing_categories,
        "coverage_percentage": len(complete_categories) / len(required_categories) if required_categories else 0
    }

def calculate_synthesis_confidence(synthesis, coverage_analysis):
    """Calculate confidence score based on synthesis quality and coverage."""
    # Base confidence from coverage
    coverage_score = coverage_analysis.get("coverage_percentage", 0)
    
    # Check citation quality
    citation_count = 0
    analysis = synthesis.get("analysis", {})
    
    for category, data in analysis.items():
        citation_count += len(data.get("citations", []))
    
    citation_score = min(1.0, citation_count / 15)  # Expect around 15 citations for full score
    
    # Check market basket quality
    market_basket = synthesis.get("market_basket", [])
    basket_score = 0
    if market_basket:
        valid_items = [item for item in market_basket 
                      if all(item.get(field) for field in ["manufacturer", "item_description", "citation"])]
        basket_score = len(valid_items) / len(market_basket) if market_basket else 0
    
    # Calculate weighted confidence score
    final_score = (coverage_score * 0.5) + (citation_score * 0.3) + (basket_score * 0.2)
    
    # Cap at 0.95 to allow for some uncertainty
    return min(0.95, final_score)

def is_query_complex_enough(query: str) -> bool:
    """Check if the query has enough complexity to warrant detailed validation."""
    word_count = len(query.split())
    contains_industry_terms = any(term in query.lower() for term in ["industry", "market", "sector", "vendors", "suppliers", "procurement"])
    
    return word_count >= 3 or contains_industry_terms

async def validate_research_output(state: ResearchState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Validate the synthesized output with additional checks for fabricated data."""
    log_step("Validating research output", 9, 10)

    if not state["synthesis"]:
        warning_highlight("No synthesis to validate")
        return {
            "confidence_score": 0.0,
            "validation_status": {
                "is_valid": False,
                "issues": ["No synthesis available for validation"],
                "coverage": {
                    "topics_covered": [],
                    "missing_topics": state.get("search_priority", []),
                    "coverage_score": 0.0
                },
                "quality": {
                    "depth_score": 0.0,
                    "relevance_score": 0.0,
                    "source_quality_score": 0.0
                },
                "recommendations": ["Perform initial research to generate synthesis"]
            },
            "validation_passed": False,
            "should_retry_search": True
        }

    info_highlight("Starting validation of research synthesis")
    
    # Get sources from the state
    sources = state.get("sources", [])
    source_urls = [source.get("url", "") for source in sources if source.get("url")]
    
    # Extract URLs from the synthesis
    synthesis = state.get("synthesis", {}) or {}
    market_basket = synthesis.get("market_basket", [])
    
    # Check for fabricated URLs in market basket
    fabricated_urls = []
    for item in market_basket:
        citation = item.get("citation", "")
        if citation and citation not in source_urls and not is_valid_url(citation):
            fabricated_urls.append(citation)
    
    if fabricated_urls:
        warning_highlight(f"Found fabricated URLs in market basket: {fabricated_urls}")
    
    try:
        # Add validation for fabricated data
        validation_prompt = {
            "role": "human",
            "content": f"""Validate this research synthesis against these criteria:

1. Data Authenticity: Check for fabricated or made-up data
2. Coverage: Check if all required topics {json.dumps(state.get('search_priority', []))} are addressed
3. Quality: Assess depth, relevance, and source quality
4. Confidence: Rate overall confidence in the findings

CRITICAL VALIDATION REQUIREMENTS:
- Flag any fabricated URLs, especially example.com or test.com domains
- Verify that market basket items have legitimate sources
- Confirm that no data appears to be fabricated or hallucinated
- Check that confidence scores are realistic given the data quality
- Ensure all claims have proper citations to source URLs

Known source URLs: {json.dumps(source_urls)}
Potentially fabricated URLs detected: {json.dumps(fabricated_urls)}

Provide validation results as JSON:
{{
    "confidence_score": float,  # 0.0 to 1.0
    "validation_status": {{
        "is_valid": bool,
        "issues": list[str],
        "fabricated_data_detected": bool,
        "fabricated_elements": list[str],
        "coverage": {{
            "topics_covered": list[str],
            "missing_topics": list[str],
            "coverage_score": float  # 0.0 to 1.0
        }},
        "quality": {{
            "depth_score": float,  # 0.0 to 1.0
            "relevance_score": float,  # 0.0 to 1.0
            "source_quality_score": float  # 0.0 to 1.0
        }},
        "recommendations": list[str]
    }}
}}

Synthesis to validate: {json.dumps(state['synthesis'])}"""
        }

        validation_result = await call_model_json(
            messages=[validation_prompt],
            config=ensure_config(config)
        )

        # Log validation results
        log_dict(
            validation_result,
            title="Validation Results"
        )

        # Extract validation components
        confidence_score = validation_result.get("confidence_score", 0.0)
        vstatus = validation_result.get("validation_status", {})
        
        # Adjust for fabricated data
        fabricated_data_detected = vstatus.get("fabricated_data_detected", bool(fabricated_urls))
        if fabricated_data_detected:
            warning_highlight("Fabricated data detected in research synthesis")
            confidence_score = min(confidence_score, 0.3)  # Cap confidence when fabricated data exists
            if "issues" in vstatus:
                vstatus["issues"].append("Fabricated data detected - results unreliable")
            vstatus["is_valid"] = False
        
        # Calculate validation result
        quality_scores = vstatus.get("quality", {})
        coverage = vstatus.get("coverage", {})
        
        avg_quality_score = sum([
            quality_scores.get("depth_score", 0.0),
            quality_scores.get("relevance_score", 0.0),
            quality_scores.get("source_quality_score", 0.0)
        ]) / 3.0
        
        coverage_score = coverage.get("coverage_score", 0.0)
        
        # Validation passes ONLY if:
        # 1. No fabricated data detected
        # 2. Confidence score >= 0.6
        # 3. Average quality score >= 0.6
        # 4. Coverage score >= 0.6
        validation_passed = (
            not fabricated_data_detected and
            confidence_score >= 0.6 and
            avg_quality_score >= 0.6 and
            coverage_score >= 0.6 and
            not any("critical" in issue.lower() for issue in vstatus.get("issues", []))
        )

        # Determine if we should retry the search
        should_retry_search = (
            not validation_passed and
            (coverage_score < 0.5 or len(coverage.get("missing_topics", [])) > len(coverage.get("topics_covered", [])))
        )

        msg = (
            f"Validation {'passed' if validation_passed else 'failed'} "
            f"(confidence: {confidence_score:.2f}, quality: {avg_quality_score:.2f}, coverage: {coverage_score:.2f})"
        )
        info_highlight(msg)

        if not validation_passed:
            warning_highlight(f"Validation failed. Issues: {vstatus.get('issues', [])}")
            if fabricated_data_detected:
                error_highlight("CRITICAL: Fabricated data detected in results")

        return {
            "confidence_score": confidence_score,
            "validation_status": vstatus,
            "validation_passed": validation_passed,
            "should_retry_search": should_retry_search
        }

    except Exception as e:
        error_highlight(f"Error during validation: {str(e)}")
        return {
            "confidence_score": 0.0,
            "validation_status": {
                "is_valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "fabricated_data_detected": True,  # Assume problem when validation fails
                "coverage": {
                    "topics_covered": [],
                    "missing_topics": state.get("search_priority", []),
                    "coverage_score": 0.0
                },
                "quality": {
                    "depth_score": 0.0,
                    "relevance_score": 0.0,
                    "source_quality_score": 0.0
                },
                "recommendations": ["Error during validation - retry validation"]
            },
            "validation_passed": False,
            "should_retry_search": True
        }
    

# --------------------------------------------------------------------
# 4. Conditional branching logic
# --------------------------------------------------------------------

def should_clarify(state: ResearchState) -> str:
    """Decide if we must request clarification from the user with more relaxed criteria."""
    info_highlight("Checking if clarification is needed")
    
    # Get missing context
    missing_context = state.get("missing_context", [])
    has_feedback = bool(state.get("human_feedback"))
    
    # Only request clarification for critical missing pieces and if no feedback exists
    if len(missing_context) > 2 and not has_feedback and state.get("needs_clarification", False):
        info_highlight("Clarification required - multiple critical context elements missing")
        return "request_clarification"
    
    info_highlight("Proceeding with search using available context")
    return "perform_search"

def continue_flow_after_clarification(state: ResearchState) -> str:
    """After clarification, return to query enrichment."""
    info_highlight("Processing completed clarification - returning to query enrichment")
    return "enrich_query_node"

def handle_validation_result(state: ResearchState) -> str:
    """Route to appropriate next step based on validation results."""
    info_highlight("Determining next step based on validation results")

    # Get validation status with safe defaults
    validation_status = state.get("validation_status", {}) or {}
    coverage = validation_status.get("coverage", {}) or {}
    missing_topics = coverage.get("missing_topics", []) or []
    confidence_score = state.get("confidence_score", 0.0)

    if state.get("validation_passed", False):
        info_highlight("Validation passed - ending research flow")
        return END

    # If we have missing topics, we should retry the search
    if missing_topics:
        warning_highlight(f"Validation failed - retrying search with missing topics: {missing_topics}")
        # Clear previous results to ensure full reprocessing
        state["search_results"] = []
        state["extracted_info"] = {}
        state["synthesis"] = None
        state["search_priority"] = missing_topics
        return "perform_search"

    # If validation failed but no missing topics, check confidence
    if confidence_score < 0.6:
        warning_highlight(f"Validation failed with low confidence ({confidence_score:.2f}) - retrying search")
        state["search_results"] = []
        state["extracted_info"] = {}
        state["synthesis"] = None
        return "perform_search"

    info_highlight("Validation failed but no missing topics - ending research flow")
    return END


# TODO Rename this here and in `handle_validation_result`
def _extracted_from_handle_validation_result_15(missing_topics, state):
    warning_highlight(f"Validation failed - retrying search with missing topics: {missing_topics}")
    # Clear previous results to ensure full reprocessing
    state["search_results"] = []
    state["extracted_info"] = {}
    state["synthesis"] = None
    state["search_priority"] = missing_topics
    return "perform_search"

# --------------------------------------------------------------------
# 5. Build the final unified graph
# --------------------------------------------------------------------
def create_research_graph() -> CompiledStateGraph:
    """Creates a graph with improved error handling and data validation."""
    info_highlight("Creating research graph with enhanced error handling")
    graph = StateGraph(ResearchState)

    # 1) Graph nodes
    info_highlight("Adding graph nodes")
    graph.add_node("initialize", initialize)
    graph.add_node("enrich_query_node", enrich_query_node)
    graph.add_node("request_clarification", request_clarification)
    graph.add_node("process_clarification", process_clarification)
    graph.add_node("perform_search", perform_search)  # Enhanced with fallbacks
    graph.add_node("process_search_results", process_search_results)
    graph.add_node("extract_key_information", extract_key_information)  # Improved data validation
    graph.add_node("synthesize_research", synthesize_research)  # Better handling of limited data
    graph.add_node("validate_research_output", validate_research_output)
    graph.add_node("aggregate_research_findings", aggregate_research_findings)

    # 2) Edges
    info_highlight("Configuring graph edges")
    # Entry → Enrich
    graph.add_edge("__start__", "initialize")
    graph.add_edge("initialize", "enrich_query_node")

    # Enrich → Clarify or Perform search
    graph.add_conditional_edges(
        "enrich_query_node",
        should_clarify,
        {
            "request_clarification": "request_clarification",
            "perform_search": "perform_search"
        }
    )

    # Clarification flow
    graph.add_edge("request_clarification", "process_clarification")
    graph.add_conditional_edges(
        "process_clarification",
        continue_flow_after_clarification,
        {
            "enrich_query_node": "enrich_query_node"
        }
    )

    # Normal search flow
    graph.add_edge("perform_search", "process_search_results")
    
    # Add data validation edge after search results processing
    graph.add_conditional_edges(
        "process_search_results",
        check_search_results_quality,  # New function to validate search results
        {
            "extract_key_information": "extract_key_information",
            "perform_search": "perform_search"  # Loop back if results insufficient
        }
    )
    
    graph.add_edge("extract_key_information", "synthesize_research")
    
    # Add data validation edge after information extraction
    graph.add_conditional_edges(
        "extract_key_information",
        check_extraction_quality,  # New function to validate extraction results
        {
            "synthesize_research": "synthesize_research",
            "perform_search": "perform_search"  # Loop back if extraction failed
        }
    )
    
    graph.add_edge("synthesize_research", "validate_research_output")
    
    # Add conditional edges from validate_research_output with improved handler
    graph.add_conditional_edges(
        "validate_research_output",
        handle_validation_result,  # Enhanced handler with better error handling
        {
            "perform_search": "perform_search",
            "aggregate_research_findings": "aggregate_research_findings",
            END: END
        }
    )
    
    # Add edge from aggregation to end
    graph.add_edge("aggregate_research_findings", END)

    # Configure interrupt points
    info_highlight("Configuring graph interrupt points")
    graph = graph.compile(
        interrupt_before=["process_clarification"]
    )

    info_highlight("Research graph creation complete")
    return graph

# New validation functions for conditional edges
def check_search_results_quality(state: ResearchState) -> str:
    """Validate search results quality before proceeding to extraction."""
    search_results = state.get("search_results", []) or []
    
    if not search_results:
        warning_highlight("No search results found - retry search with broadened terms")
        # Modify search strategy for retry
        enriched_query = state.get("enriched_query", {}) or {}
        if enriched_query:
            # Simplify the query to get more results
            primary_keywords = enriched_query.get("primary_keywords", []) or []
            if primary_keywords and len(primary_keywords) > 1:
                # Use just the first keyword for broader results
                enriched_query["enhanced_query"] = primary_keywords[0]
                state["enriched_query"] = enriched_query
                info_highlight(f"Broadened search to: {primary_keywords[0]}")
        
        return "perform_search"
    
    if len(search_results) < 2:
        warning_highlight("Insufficient search results - retry with alternative strategy")
        # Set flag for alternate search approach
        state["alternate_search"] = True
        return "perform_search"
    
    info_highlight(f"Search produced {len(search_results)} results - proceeding to information extraction")
    return "extract_key_information"

def check_extraction_quality(state: ResearchState) -> str:
    """Validate extraction quality before proceeding to synthesis."""
    extracted_info = state.get("extracted_info", {})
    extraction_status = state.get("extraction_status", "")
    
    if extraction_status in ["failed", "error"]:
        warning_highlight(f"Extraction failed: {state.get('extraction_message', 'Unknown error')}")
        # Try different search strategy
        if not state.get("alternate_extraction", False):
            state["alternate_extraction"] = True
            return "perform_search"
    
    if not extracted_info or not any(extracted_info.values()):
        warning_highlight("No information extracted - retry with different strategy")
        # Set flag for alternate extraction approach
        state["alternate_extraction"] = True
        return "perform_search"
    
    # Check if we have minimum viable data (at least 3 facts)
    total_facts = sum(len(facts) for facts in extracted_info.values())
    if total_facts < 3:
        warning_highlight(f"Insufficient data extracted ({total_facts} facts) - retry search")
        return "perform_search"
    
    info_highlight(f"Extraction produced {total_facts} facts - proceeding to synthesis")
    return "synthesize_research"

# --------------------------------------------------------------------
# 6. Example usage
# --------------------------------------------------------------------
research_graph = create_research_graph()

# You can now invoke this graph by passing an initial HumanMessage:
# example_message = HumanMessage(content="I'd like to learn about quantum computing. Need details?")
# result_state = research_graph.invoke(example_message)
# 
# The flow will proceed through:
#   initialize → enrich_query_node → request_clarification (since "details?" triggers missing context) ...
#   (Waits for user clarification) → process_clarification → enrich_query_node → perform_search → ...
#
# In a real application, you'll feed the user's clarification as the next invocation:
# clarified_message = HumanMessage(content="Specifically, I want more details on quantum entanglement.")
# result_state = research_graph.invoke(clarified_message, state=result_state)
#
# And so on...