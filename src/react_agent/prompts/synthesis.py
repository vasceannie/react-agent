"""Enhanced synthesis and output module.

This module provides improved synthesis and output formatting capabilities
to create more thorough, insightful and verbose research reports with 
better statistics integration and citation handling.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import json
from datetime import datetime, timezone
import re

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from react_agent.utils.logging import get_logger, info_highlight, warning_highlight, error_highlight
from react_agent.utils.llm import call_model_json
from react_agent.tools import StatisticsExtractionTool
from react_agent.utils.defaults import get_default_extraction_result
from react_agent.utils.cache import ProcessorCache, create_checkpoint, load_checkpoint
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Initialize logger
logger = get_logger(__name__)

# Initialize processor cache for synthesis
synthesis_cache = ProcessorCache(thread_id="synthesis")

# Initialize memory saver for caching
memory_saver = MemorySaver()

# Initialize tools
statistics_tool = StatisticsExtractionTool()

# Enhanced synthesis prompt template
ENHANCED_SYNTHESIS_PROMPT = """Create a comprehensive synthesis of research findings for: {query}

REQUIREMENTS:
1. Structure your synthesis with these EXACT sections:
   - Executive Summary: Concise overview of key findings with critical statistics
   - Domain Overview: Essential context and background with industry statistics
   - Market Dynamics: Size, growth, trends, competition, procurement patterns with market statistics
   - Provider Landscape: Key vendors, manufacturers, distributors, and their positioning with market share data
   - Technical Requirements: Specifications, standards, and procurement requirements with technical statistics
   - Regulatory Landscape: Compliance, legal requirements, and procurement regulations with compliance data
   - Implementation Factors: Resources, process, challenges, and procurement considerations with implementation statistics
   - Cost Analysis: Pricing, ROI, financial factors, volume discounts, and contract terms with financial metrics
   - Best Practices: Recommended approaches for procurement and sourcing with adoption statistics
   - Contract & Procurement Strategy: Contract terms, negotiation strategies, and procurement processes with benchmarks

2. For EACH section:
   - Synthesize insights from multiple sources when available
   - Prioritize STATISTICAL information and NUMERICAL data
   - Include specific PERCENTAGES, AMOUNTS, and METRICS
   - Address inconsistencies and note knowledge gaps
   - Include SPECIFIC facts with proper citations
   - Present balanced perspectives where there are differences
   - Prioritize VERIFIED information from authoritative sources
   - If information is missing for any section, explicitly note this

3. For "Confidence Assessment":
   - Evaluate information completeness for each section
   - Identify potential biases in the sources
   - Note limitations in the research
   - Assign justified confidence scores by section
   - Provide specific reasons for confidence ratings

AVAILABLE RESEARCH:
{research_json}

FORMAT RESPONSE AS JSON:
{{
  "synthesis": {{
    "executive_summary": {{ "content": "", "citations": [], "statistics": [] }},
    "domain_overview": {{ "content": "", "citations": [], "statistics": [] }},
    "market_dynamics": {{ "content": "", "citations": [], "statistics": [] }},
    "provider_landscape": {{ "content": "", "citations": [], "statistics": [] }},
    "technical_requirements": {{ "content": "", "citations": [], "statistics": [] }},
    "regulatory_landscape": {{ "content": "", "citations": [], "statistics": [] }},
    "implementation_factors": {{ "content": "", "citations": [], "statistics": [] }},
    "cost_considerations": {{ "content": "", "citations": [], "statistics": [] }},
    "best_practices": {{ "content": "", "citations": [], "statistics": [] }},
    "contract_procurement_strategy": {{ "content": "", "citations": [], "statistics": [] }}
  }},
  "confidence_assessment": {{
    "overall_score": 0.0-1.0,
    "section_scores": {{
      "executive_summary": 0.0-1.0,
      "domain_overview": 0.0-1.0,
      "market_dynamics": 0.0-1.0,
      "provider_landscape": 0.0-1.0,
      "technical_requirements": 0.0-1.0,
      "regulatory_landscape": 0.0-1.0,
      "implementation_factors": 0.0-1.0,
      "cost_considerations": 0.0-1.0,
      "best_practices": 0.0-1.0,
      "contract_procurement_strategy": 0.0-1.0
    }},
    "limitations": [],
    "knowledge_gaps": [],
    "confidence_justifications": {{
      "executive_summary": "",
      "domain_overview": "",
      "market_dynamics": "",
      "provider_landscape": "",
      "technical_requirements": "",
      "regulatory_landscape": "",
      "implementation_factors": "",
      "cost_considerations": "",
      "best_practices": "",
      "contract_procurement_strategy": ""
    }}
  }}
}}

REMEMBER:
- Prioritize statistical data and numerical findings
- Include specific numbers, percentages, and metrics
- Emphasize recent studies, surveys, and market reports
- Highlight data from industry-leading sources
- Only include claims that are supported by the research
- Use clear, concise language focused on business impact
- Highlight conflicting information when present
- Pay special attention to procurement and sourcing-related insights
"""

# Enhanced report template with better statistics and citations
ENHANCED_REPORT_TEMPLATE = """
# {title}

## Executive Summary
{executive_summary}

## Key Findings & Statistics
{key_statistics}

## Domain Overview
{domain_overview}

## Market Analysis
### Market Size & Growth
{market_size}

### Competitive Landscape
{competitive_landscape}

### Trends & Developments
{market_trends}

## Provider Landscape
### Key Vendors
{key_vendors}

### Vendor Comparison
{vendor_comparison}

## Technical Requirements
{technical_requirements}

## Regulatory Considerations
{regulatory_landscape}

## Implementation Strategy
{implementation_factors}

## Cost Analysis
### Cost Structure
{cost_structure}

### Pricing Models
{pricing_models}

### ROI Considerations
{roi_considerations}

## Best Practices
{best_practices}

## Procurement Strategy
{procurement_strategy}

## Recommendations
{recommendations}

## Sources & Citations
{sources}

---
**Research Confidence:** {confidence_score}/1.0  
**Date Generated:** {generation_date}  
{confidence_notes}
"""

def extract_all_statistics(synthesis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all statistics from synthesis results."""
    all_stats = []
    
    for section_name, section_data in synthesis.items():
        if isinstance(section_data, dict) and "statistics" in section_data:
            stats = section_data.get("statistics", [])
            if stats and isinstance(stats, list):
                for stat in stats:
                    if isinstance(stat, dict):
                        # Add section name to statistic
                        stat["section"] = section_name
                        all_stats.append(stat)
    
    # Sort by quality score if available
    sorted_stats = sorted(
        all_stats,
        key=lambda x: x.get("quality_score", 0),
        reverse=True
    )
    
    return sorted_stats

def format_citation(citation: Dict[str, Any]) -> str:
    """Format a citation for inclusion in the report."""
    if not citation or not isinstance(citation, dict):
        return ""
        
    title = citation.get("title", "")
    source = citation.get("source", "")
    url = citation.get("url", "")
    date = citation.get("date", "")
    
    if title and source:
        return f"{title} ({source}{', ' + date if date else ''})"
    elif title:
        return title
    elif source:
        return source
    elif url:
        return url
    else:
        return "Unnamed source"

def format_statistic(stat: Dict[str, Any]) -> str:
    """Format a statistic for inclusion in the report."""
    if not stat or not isinstance(stat, dict):
        return ""
        
    text = stat.get("text", "")
    citation = ""
    
    # Add citation if available
    citations = stat.get("citations", [])
    if citations and isinstance(citations, list) and len(citations) > 0:
        first_citation = citations[0]
        if isinstance(first_citation, dict):
            source = first_citation.get("source", "")
            if source:
                citation = f" ({source})"
    
    return f"{text}{citation}"

def highlight_statistics_in_content(content: str, statistics: List[Dict[str, Any]]) -> str:
    """Highlight statistics in content with bold formatting."""
    if not content or not statistics:
        return content
        
    highlighted_content = content
    
    for stat in statistics:
        if isinstance(stat, dict) and "text" in stat:
            text = stat.get("text", "")
            if text and text in highlighted_content:
                # Highlight the statistic with bold formatting
                highlighted_content = highlighted_content.replace(text, f"**{text}**")
    
    return highlighted_content

async def synthesize_research(
    state: Dict[str, Any],
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Synthesize all research data into a comprehensive result with enhanced statistics focus."""
    info_highlight("Synthesizing research results with enhanced statistics focus")
    
    try:
        # Check cache with TTL
        cache_key = f"synthesize_research_{hash(str(state))}"
        if cached_state := synthesis_cache.get(cache_key):
            if cached_state.get("data"):
                return cached_state["data"]
        
        categories = state["categories"]
        original_query = state["original_query"]
        
        # Prepare research data for synthesis
        research_data = {}
        for category, category_state in categories.items():
            # Extract statistics from facts
            statistics = []
            for fact in category_state.get("extracted_facts", []):
                if "statistics" in fact:
                    statistics.extend(fact["statistics"])
                elif "source_text" in fact:
                    # Extract statistics from source text using the tool
                    extracted_stats = statistics_tool.run(
                        text=fact["source_text"],
                        url=fact.get("source_url", ""),
                        source_title=fact.get("source_title", "")
                    )
                    statistics.extend(extracted_stats)
            
            research_data[category] = {
                "facts": category_state["extracted_facts"],
                "sources": category_state["sources"],
                "quality_score": category_state["quality_score"],
                "statistics": statistics  # Add extracted statistics
            }
        
        # Generate prompt
        synthesis_prompt = ENHANCED_SYNTHESIS_PROMPT.format(
            query=original_query,
            research_json=json.dumps(research_data, indent=2)
        )
        
        # Call model for synthesis
        synthesis_result = await call_model_json(
            messages=[{"role": "human", "content": synthesis_prompt}],
            config=config
        )
        
        # Extract key statistics for reference
        synthesis_sections = synthesis_result.get("synthesis", {})
        all_statistics = extract_all_statistics(synthesis_sections)
        synthesis_result["key_statistics"] = all_statistics[:10]  # Top 10 statistics
        
        overall_score = synthesis_result.get("confidence_assessment", {}).get("overall_score", 0.0)
        info_highlight(f"Research synthesis complete with confidence score: {overall_score:.2f}")
        
        result = {
            "synthesis": synthesis_result,
            "status": "synthesized"
        }
        
        # Cache result with TTL
        synthesis_cache.put(
            cache_key,
            {
                "data": result,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            ttl=3600  # 1 hour TTL
        )
        
        return result
        
    except Exception as e:
        error_highlight(f"Error in research synthesis: {str(e)}")
        return {"error": {"message": f"Error in research synthesis: {str(e)}", "phase": "synthesis"}}

async def prepare_enhanced_response(
    state: Dict[str, Any],
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Prepare an enhanced final response with detailed statistics and citations."""
    info_highlight("Preparing enhanced final response")
    
    synthesis = state.get("synthesis", {})
    validation = state.get("validation_result", {})
    original_query = state["original_query"]
    
    # Get synthesis content
    synthesis_content = synthesis.get("synthesis", {}) if synthesis else {}
    confidence = synthesis.get("confidence_assessment", {}) if synthesis else {}
    
    # Get all statistics
    all_statistics = synthesis.get("key_statistics", []) if synthesis else []
    
    # Format the title with proper capitalization
    title = "Research Results: " + original_query.capitalize()
    
    # Format each section with highlighted statistics
    sections = {}
    for section_name, section_data in synthesis_content.items():
        if isinstance(section_data, dict) and "content" in section_data:
            content = section_data.get("content", "")
            statistics = section_data.get("statistics", [])
            
            # Highlight statistics in content
            highlighted_content = highlight_statistics_in_content(content, statistics)
            
            # Add citations
            citations = section_data.get("citations", [])
            if citations and isinstance(citations, list) and len(citations) > 0:
                citation_text = "\n\n**Sources:** " + ", ".join(
                    format_citation(citation) for citation in citations if citation
                )
                sections[section_name] = highlighted_content + citation_text
            else:
                sections[section_name] = highlighted_content
    
    # Format key statistics section
    key_stats_formatted = []
    for stat in all_statistics:
        formatted_stat = format_statistic(stat)
        if formatted_stat:
            key_stats_formatted.append(f"- {formatted_stat}")
    
    key_statistics_section = "\n".join(key_stats_formatted) if key_stats_formatted else "No key statistics available."
    
    # Format sources section
    sources_set = set()
    for section_data in synthesis_content.values():
        if isinstance(section_data, dict) and "citations" in section_data:
            citations = section_data.get("citations", [])
            for citation in citations:
                if isinstance(citation, dict):
                    formatted = format_citation(citation)
                    if formatted:
                        sources_set.add(formatted)
    
    sources_list = sorted(list(sources_set))
    sources_section = "\n".join(f"- {source}" for source in sources_list) if sources_list else "No sources available."
    
    # Get confidence information
    confidence_score = confidence.get("overall_score", 0.0)
    limitations = confidence.get("limitations", [])
    knowledge_gaps = confidence.get("knowledge_gaps", [])
    
    # Format confidence notes
    confidence_notes = []
    if limitations:
        confidence_notes.append("**Limitations:** " + ", ".join(limitations))
    if knowledge_gaps:
        confidence_notes.append("**Knowledge Gaps:** " + ", ".join(knowledge_gaps))
    
    confidence_notes_text = "\n".join(confidence_notes)
    
    # Generate recommendations based on synthesis
    recommendations = await generate_recommendations(synthesis_content, original_query, config)
    
    # Fill in the template
    report_content = ENHANCED_REPORT_TEMPLATE.format(
        title=title,
        executive_summary=sections.get("executive_summary", "No executive summary available."),
        key_statistics=key_statistics_section,
        domain_overview=sections.get("domain_overview", "No domain overview available."),
        market_size=sections.get("market_dynamics", "No market dynamics information available."),
        competitive_landscape=sections.get("provider_landscape", "No provider landscape information available."),
        market_trends=sections.get("market_dynamics", "No market trends information available."),
        key_vendors=sections.get("provider_landscape", "No vendor information available."),
        vendor_comparison=sections.get("provider_landscape", "No vendor comparison available."),
        technical_requirements=sections.get("technical_requirements", "No technical requirements information available."),
        regulatory_landscape=sections.get("regulatory_landscape", "No regulatory information available."),
        implementation_factors=sections.get("implementation_factors", "No implementation information available."),
        cost_structure=sections.get("cost_considerations", "No cost structure information available."),
        pricing_models=sections.get("cost_considerations", "No pricing models information available."),
        roi_considerations=sections.get("cost_considerations", "No ROI considerations available."),
        best_practices=sections.get("best_practices", "No best practices information available."),
        procurement_strategy=sections.get("contract_procurement_strategy", "No procurement strategy information available."),
        recommendations=recommendations,
        sources=sources_section,
        confidence_score=f"{confidence_score:.2f}",
        generation_date=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        confidence_notes=confidence_notes_text
    )
    
    # Create the response message
    response_message = AIMessage(content=report_content)
    
    return {
        "messages": [response_message],
        "status": "complete",
        "complete": True
    }

async def generate_recommendations(
    synthesis_content: Dict[str, Any],
    query: str,
    config: Optional[RunnableConfig] = None
) -> str:
    """Generate recommendations based on synthesis content."""
    info_highlight("Generating recommendations based on synthesis")
    
    if not synthesis_content:
        return "No recommendations available due to insufficient data."
    
    # Create a prompt for recommendations
    recommendations_prompt = f"""
    Based on the research synthesis for "{query}", generate 5-7 specific, actionable recommendations.
    Each recommendation should:
    1. Be specific and actionable
    2. Reference relevant statistics or findings when available
    3. Address a key need or gap identified in the research
    4. Be practical and implementable
    5. Include expected benefits or outcomes
    
    Synthesis data:
    {json.dumps(synthesis_content, indent=2)}
    
    FORMAT:
    Return a markdown list of recommendations with brief explanations.
    """
    
    try:
        response = await call_model_json(
            messages=[{"role": "human", "content": recommendations_prompt}],
            config=config
        )
        
        if isinstance(response, dict) and "recommendations" in response:
            return response["recommendations"]
        elif isinstance(response, dict) and "content" in response:
            return response["content"]
        elif isinstance(response, str):
            return response
        else:
            # Default format if response structure is unexpected
            return "Recommendations could not be generated due to unexpected response format."
    except Exception as e:
        error_highlight(f"Error generating recommendations: {str(e)}")
        return "Recommendations could not be generated due to an error."

__all__ = [
    "chunk_text",
    "preprocess_content",
    "estimate_tokens",
    "should_skip_content",
    "merge_chunk_results",
    "validate_content",
    "detect_content_type"
]