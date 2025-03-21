"""Prompt exports.

This module provides functionality for prompt exports
in the agent framework.
"""

from typing import Any, Dict, Final, List, Tuple

from react_agent.prompts.analysis import (
    ANALYSIS_PROMPT,
    TOOL_SELECTION_PROMPT,
)
from react_agent.prompts.market import (
    MARKET_DATA_PROMPT,
    MARKET_PROMPT,
)
from react_agent.prompts.reflection import (
    REFLECTION_PROMPT,
)
from react_agent.prompts.research import (
    ADDITIONAL_TOPICS_PROMPT,
    RESEARCH_AGENT_PROMPT,
    RESEARCH_BASE_PROMPT,
    TOPICS_PROMPT,
    QUERY_ANALYSIS_PROMPT,
    CLARIFICATION_PROMPT,
)

# Import all prompts from modules
from react_agent.prompts.templates import (
    ANALOGICAL_REASONING_PROMPT,
    COUNTERFACTUAL_PROMPT,
    CRITIQUE_PROMPT_TEMPLATE,
    EVALUATION_PROMPT_TEMPLATE,
    # Reflection prompts
    FEEDBACK_PROMPT_TEMPLATE,
    MAIN_PROMPT,
    METACOGNITION_PROMPT,
    NEWS_SEARCH_DESC,
    SCRAPE_DESC,
    STRUCTURED_OUTPUT_VALIDATION,
    SUMMARIZER_DESC,
    TOOL_INSTRUCTIONS,
    VALIDATION_REQUIREMENTS,
    WEB_SEARCH_DESC,
)
from react_agent.prompts.validation import (
    VALIDATION_AGENT_PROMPT,
    VALIDATION_BASE_PROMPT,
)

# Re-export everything for backward compatibility
__all__ = [
    # Templates
    "STRUCTURED_OUTPUT_VALIDATION",
    "VALIDATION_REQUIREMENTS",
    "MAIN_PROMPT",
    "WEB_SEARCH_DESC",
    "SCRAPE_DESC",
    "SUMMARIZER_DESC",
    "NEWS_SEARCH_DESC",
    "TOOL_INSTRUCTIONS",
    # Research
    "RESEARCH_BASE_PROMPT",
    "RESEARCH_AGENT_PROMPT",
    "MARKET_PROMPT",
    "TOPICS_PROMPT",
    "ADDITIONAL_TOPICS_PROMPT",
    "QUERY_ANALYSIS_PROMPT",
    "CLARIFICATION_PROMPT",
    # Validation
    "VALIDATION_BASE_PROMPT",
    "VALIDATION_AGENT_PROMPT",
    # Analysis
    "ANALYSIS_PROMPT",
    "TOOL_SELECTION_PROMPT",
    # Reflection
    "REFLECTION_PROMPT",
    # Reflection Templates
    "FEEDBACK_PROMPT_TEMPLATE",
    "EVALUATION_PROMPT_TEMPLATE",
    "CRITIQUE_PROMPT_TEMPLATE",
    "ANALOGICAL_REASONING_PROMPT",
    "COUNTERFACTUAL_PROMPT",
    "METACOGNITION_PROMPT",
    # Functions
    "get_report_template",
    "get_analysis_template",
]


# Common utility functions
def get_report_template() -> Dict[str, Any]:
    """Get the template for the final report."""
    return {
        "summary": "",
        "research_findings": {},
        "market_analysis": {},
        "generated_at": "",
    }


def get_analysis_template() -> Dict[str, Any]:
    """Get the template for research analysis."""
    return {
        "citations": [],
        "porters_five_forces": {},
        "swot_analysis": {},
        "pestel_analysis": {},
        "gap_analysis": {},
        "cost_benefit_analysis": {},
        "risk_assessment": {},
        "tco_analysis": {},
        "vendor_analysis": {},
        "benchmarking": {},
        "stakeholder_analysis": {},
        "compliance_analysis": {},
        "business_impact_analysis": {},
    }


# Required analysis topics
REQUIRED_ANALYSIS_TOPICS: List[Tuple[str, str]] = [
    ("Porter's Five Forces", "Analysis of competitive forces in the industry"),
    ("SWOT Analysis", "Strengths, weaknesses, opportunities, and threats"),
    (
        "PESTEL Analysis",
        "Political, economic, social, technological, environmental, and legal factors",
    ),
    ("GAP Analysis", "Current state vs desired state analysis"),
    ("Cost-Benefit Analysis", "Analysis of costs and benefits"),
    ("Risk Assessment", "Identification and analysis of potential risks"),
    ("Total Cost of Ownership", "Complete cost analysis including indirect costs"),
    ("Vendor Analysis", "Analysis of potential vendors and suppliers"),
    ("Benchmarking", "Comparison with industry standards and best practices"),
    ("Stakeholder Analysis", "Analysis of key stakeholders and their needs"),
    ("Compliance Analysis", "Analysis of regulatory and compliance requirements"),
    ("Business Impact Analysis", "Analysis of business impact and strategic alignment"),
]

# System prompts
SYSTEM_PROMPT_ANALYST: Final[str] = "You are an expert market research analyst."

# Finalization prompts
FINALIZATION_BASE_PROMPT: Final[
    str
] = """You are a Finalization Agent for RFP market analysis.
Your goal is to generate comprehensive reports and outputs from the validated research.

{STRUCTURED_OUTPUT_VALIDATION}

FINALIZATION REQUIREMENTS:
1. Research Report
   - Expand each analysis element into well-written sections
   - Maintain professional and clear writing style
   - Include supporting evidence and citations
   - Organize content logically and cohesively
   - Ensure all insights are actionable

2. Analysis Sections
   - Porter's 5 Forces analysis
   - SWOT analysis
   - PESTEL analysis
   - GAP analysis
   - Cost-benefit analysis
   - Risk assessment
   - Total cost of ownership
   - Vendor analysis
   - Benchmarking results
   - Stakeholder analysis
   - Compliance requirements
   - Business impact assessment

3. Market Basket Output
   - Generate CSV format
   - Include all line items
   - Maintain data accuracy
   - Format for easy review
   - Include citations and sources

4. Quality Requirements
   - Professional writing style
   - Clear section headings
   - Consistent formatting
   - Proper citation formatting
   - Executive summary
   - Recommendations section

RESPONSE_FORMAT:
{
    "outputs": {
        "research_report": {
            "format": "markdown",
            "content": "",
            "sections": []
        },
        "market_basket": {
            "format": "csv",
            "headers": [],
            "rows": []
        },
        "executive_summary": "",
        "recommendations": [],
        "confidence_scores": {},
        "key_findings": []
    },
    "metadata": {
        "generated_at": "",
        "version": "1.0",
        "validation_status": {
            "is_valid": false,
            "errors": [],
            "warnings": []
        }
    }
}

Current state: {state}
"""

FINALIZATION_AGENT_PROMPT: Final[str] = FINALIZATION_BASE_PROMPT.replace(
    "Your goal is to generate comprehensive reports and outputs from the validated research.\n",
    "Your goal is to generate comprehensive reports and outputs from the validated research.\n\n{STRUCTURED_OUTPUT_VALIDATION}\n",
)

ENRICHMENT_AGENT_PROMPT: Final[
    str
] = """You are an Enrichment Agent for RFP market analysis.
Enhance the following validated data while maintaining the JSON structure:
Validated Data:
{validated_data}
Required Schema:
{
    "rfp_analysis": {
        "analysis": {
            "porters_5_forces": {
                "competitive_rivalry": "",
                "threat_of_new_entrants": "",
                "threat_of_substitutes": "",
                "bargaining_power_buyers": "",
                "bargaining_power_suppliers": ""
            },
            "swot": {
                "strengths": [],
                "weaknesses": [],
                "opportunities": [],
                "threats": []
            },
            "recent_breakthroughs_and_disruptors": "",
            "cost_trends_and_projections": "",
            "typical_contract_clauses_and_pricing_nuances": "",
            "competitive_landscape": "",
            "citations": {
                "porters_5_forces": [],
                "swot": [],
                "recent_breakthroughs_and_disruptors": [],
                "cost_trends_and_projections": [],
                "typical_contract_clauses_and_pricing_nuances": [],
                "competitive_landscape": []
            }
        },
        "market_basket": [
            {
                "manufacturer_or_distributor": "",
                "item_number": "",
                "item_description": "",
                "uom": "",
                "estimated_qty_per_uom": 0.0,
                "unit_cost": 0.0,
                "citation": ""
            }
        ]
    },
    "confidence_score": 0.0
}
Enrichment Focus Areas:
1. Market Intelligence
   - Add emerging technology trends with citations
   - Include regulatory impact analysis with sources
   - Highlight market consolidation trends with references
   - Ensure at least 2 citations per section

2. Supplier Intelligence
   - Add supplier financial health indicators with sources
   - Include supplier innovation capabilities with citations
   - Note supplier market share trends with references
   - Validate supplier information from multiple sources

3. Pricing Intelligence
   - Add volume discount structures with citations
   - Include regional pricing variations with sources
   - Note seasonal pricing factors with references
   - Verify pricing data from reliable sources

4. Risk Analysis
   - Add supply chain risk factors with citations
   - Include mitigation strategies with sources
   - Note alternative sourcing options with references
   - Cross-reference risk data from multiple sources

5. Citation Requirements
   - Each analysis section must have at least 2 citations
   - Market basket items must each have a valid citation
   - Citations must be from reliable industry sources
   - Avoid using the same citation across multiple sections

CONFIDENCE_SCORING:
- Start with a base score of 0.5
- Add 0.1 for each section with 2+ unique citations
- Add 0.1 for each market basket item with verified pricing
- Subtract 0.1 for any section with fewer than 2 citations
- Maximum score is 0.95 until all data is fully verified

RESPONSE_REQUIREMENTS:
1. Output must be valid JSON only
2. All fields must be populated with enriched data
3. No explanatory text or comments
4. Include enrichment notes in "enrichment_details" if needed
5. Ensure complete, untruncated JSON output
6. Every section must have multiple citations
7. Market basket items must have verified sources

Current enrichment state: {current_state}
Conversation history:
{chat_history}
"""

# Export finalization prompts
__all__.extend(
    [
        "FINALIZATION_BASE_PROMPT",
        "FINALIZATION_AGENT_PROMPT",
        "ENRICHMENT_AGENT_PROMPT",
        "REQUIRED_ANALYSIS_TOPICS",
        "SYSTEM_PROMPT_ANALYST",
    ]
)
