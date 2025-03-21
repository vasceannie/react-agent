"""Enhanced research-specific prompts.

This module provides specialized prompts for different research categories
to improve extraction quality and relevance.
"""

from typing import Final, Dict, List, Any

# Base templates for common validation requirements
STRUCTURED_OUTPUT_VALIDATION: Final[str] = """CRITICAL: All responses MUST:
1. Be valid JSON only - no additional text or comments
2. Follow the exact schema provided
3. Never return empty or null values - use empty strings or arrays instead
4. Include all required fields
5. Use proper data types (strings, numbers, arrays)
6. Maintain proper JSON syntax
7. Include citations for all data points
8. Pass JSON schema validation

Any response that fails these requirements will be rejected."""

# Enhanced query analysis prompt with improved categorization and structure
QUERY_ANALYSIS_PROMPT: Final[str] = """Analyze the following research query to generate targeted search terms.

Query: {query}

TASK:
Break down this query into precise search components following these rules:
1. Use the UNSPSC taxonomy to identify relevant procurement categories
2. Extract no more than 3-5 focused keywords per category
3. Prioritize specificity over quantity
4. Identify the specific industry verticals, markets, and sectors
5. Determine geographical scope if relevant

FORMAT YOUR RESPONSE AS JSON:
{{
    "unspsc_categories": [
        {{"code": "code", "name": "category name", "relevance": 0.0-1.0}}
    ],
    "search_components": {{
        "primary_topic": "", 
        "industry": "",
        "product_type": "",
        "geographical_focus": ""
    }},
    "search_terms": {{
        "market_dynamics": [],
        "provider_landscape": [],
        "technical_requirements": [],
        "regulatory_landscape": [],
        "cost_considerations": [],
        "best_practices": [],
        "implementation_factors": []
    }},
    "boolean_query": "",
    "missing_context": []
}}

IMPORTANT: 
- Keep each category to a MAXIMUM of 5 focused search terms
- Only include truly essential items in "missing_context" - make reasonable assumptions
- For "boolean_query" create a precise search string using AND/OR operators
- Assign relevance scores (0.0-1.0) to each UNSPSC category
- Your response must be valid JSON with all fields present
- Do not include any comments or additional text in the JSON response
"""

# Specialized extraction prompts for different research categories
EXTRACTION_PROMPTS: Dict[str, str] = {
    "market_dynamics": """Extract factual information about MARKET DYNAMICS from this content about {query}.

URL: {url}

INSTRUCTIONS:
1. ONLY extract VERIFIED facts about market size, growth rates, trends, forecasts, and competitive dynamics
2. Format each fact with:
   - The fact statement
   - Direct quote from the content supporting the fact
   - Confidence rating (high/medium/low)
3. If the document doesn't contain relevant market data, indicate this

CONTENT:
{content}

FORMAT YOUR RESPONSE AS JSON:
{
  "extracted_facts": [
    {
      "fact": "Clear factual statement about market dynamics",
      "source_text": "Direct quote from content",
      "confidence": "high/medium/low",
      "data_type": "market_size/growth_rate/trend/forecast/competitive"
    }
  ],
  "market_metrics": {
    "market_size": null,  // Include if available with units
    "growth_rate": null,  // Include if available with time period
    "forecast_period": null  // Include if available
  },
  "relevance_score": 0.0-1.0
}
""",

    "provider_landscape": """Extract factual information about PROVIDERS/VENDORS from this content about {query}.

URL: {url}

INSTRUCTIONS:
1. ONLY extract VERIFIED facts about vendors, suppliers, service providers, and market players
2. Format each fact with:
   - The vendor name and specific details
   - Direct quote from the content supporting the fact
   - Confidence rating (high/medium/low)
3. If no vendor information is found, indicate this

CONTENT:
{content}

FORMAT YOUR RESPONSE AS JSON:
{
  "extracted_vendors": [
    {
      "vendor_name": "Name of vendor",
      "description": "What they provide",
      "market_position": "leader/challenger/niche",
      "source_text": "Direct quote from content",
      "confidence": "high/medium/low"
    }
  ],
  "vendor_relationships": [
    {
      "relationship_type": "partnership/competition/acquisition",
      "entities": ["vendor1", "vendor2"],
      "source_text": "Direct quote from content",
      "confidence": "high/medium/low"
    }
  ],
  "relevance_score": 0.0-1.0
}
""",

    "technical_requirements": """Extract factual information about TECHNICAL REQUIREMENTS from this content about {query}.

URL: {url}

INSTRUCTIONS:
1. ONLY extract VERIFIED facts about specifications, standards, technologies, and requirements
2. Format each fact with:
   - The technical requirement or specification
   - Direct quote from the content supporting the fact
   - Confidence rating (high/medium/low)
3. If no technical information is found, indicate this

CONTENT:
{content}

FORMAT YOUR RESPONSE AS JSON:
{
  "extracted_requirements": [
    {
      "requirement": "Specific technical requirement",
      "category": "hardware/software/compliance/integration/performance",
      "source_text": "Direct quote from content",
      "confidence": "high/medium/low"
    }
  ],
  "standards": [
    {
      "standard_name": "Name of standard or protocol",
      "description": "Brief description",
      "source_text": "Direct quote from content",
      "confidence": "high/medium/low"
    }
  ],
  "relevance_score": 0.0-1.0
}
""",

    "regulatory_landscape": """Extract factual information about REGULATIONS & COMPLIANCE from this content about {query}.

URL: {url}

INSTRUCTIONS:
1. ONLY extract VERIFIED facts about regulations, laws, compliance requirements, and standards
2. Format each regulation with:
   - The regulation name and jurisdiction
   - Direct quote from the content supporting the fact
   - Confidence rating (high/medium/low)
3. If no regulatory information is found, indicate this

CONTENT:
{content}

FORMAT YOUR RESPONSE AS JSON:
{
  "extracted_regulations": [
    {
      "regulation": "Name of regulation/law/standard",
      "jurisdiction": "Geographical or industry scope",
      "description": "Brief description of requirement",
      "source_text": "Direct quote from content",
      "confidence": "high/medium/low"
    }
  ],
  "compliance_requirements": [
    {
      "requirement": "Specific compliance requirement",
      "description": "What must be done",
      "source_text": "Direct quote from content", 
      "confidence": "high/medium/low"
    }
  ],
  "relevance_score": 0.0-1.0
}
""",

    "cost_considerations": """Extract factual information about COSTS & PRICING from this content about {query}.

URL: {url}

INSTRUCTIONS:
1. ONLY extract VERIFIED facts about pricing, costs, budgets, TCO, ROI, and financial considerations
2. Format each fact with:
   - The specific cost information
   - Direct quote from the content supporting the fact
   - Confidence rating (high/medium/low)
3. If no cost information is found, indicate this

CONTENT:
{content}

FORMAT YOUR RESPONSE AS JSON:
{
  "extracted_costs": [
    {
      "cost_item": "Specific cost element",
      "amount": null,  // Include if available with currency
      "context": "Description of pricing context",
      "source_text": "Direct quote from content",
      "confidence": "high/medium/low"
    }
  ],
  "pricing_models": [
    {
      "model_type": "subscription/one-time/usage-based/etc",
      "description": "How the pricing works",
      "source_text": "Direct quote from content",
      "confidence": "high/medium/low"
    }
  ],
  "relevance_score": 0.0-1.0
}
""",

    "best_practices": """Extract factual information about BEST PRACTICES from this content about {query}.

URL: {url}

INSTRUCTIONS:
1. ONLY extract VERIFIED best practices, methodologies, approaches, and success factors
2. Format each best practice with:
   - The practice description
   - Direct quote from the content supporting it
   - Confidence rating (high/medium/low)
3. If no best practices are found, indicate this

CONTENT:
{content}

FORMAT YOUR RESPONSE AS JSON:
{
  "extracted_practices": [
    {
      "practice": "Description of best practice",
      "benefit": "What benefit it provides",
      "source_text": "Direct quote from content",
      "confidence": "high/medium/low"
    }
  ],
  "methodologies": [
    {
      "methodology": "Name of methodology or approach",
      "description": "How it works",
      "source_text": "Direct quote from content",
      "confidence": "high/medium/low"
    }
  ],
  "relevance_score": 0.0-1.0
}
""",

    "implementation_factors": """Extract factual information about IMPLEMENTATION FACTORS from this content about {query}.

URL: {url}

INSTRUCTIONS:
1. ONLY extract VERIFIED facts about implementation considerations, requirements, challenges, and success factors
2. Format each factor with:
   - The implementation factor
   - Direct quote from the content supporting the fact
   - Confidence rating (high/medium/low)
3. If no implementation information is found, indicate this

CONTENT:
{content}

FORMAT YOUR RESPONSE AS JSON:
{
  "extracted_factors": [
    {
      "factor": "Description of implementation factor",
      "category": "resource/timeline/risk/organizational/technical",
      "source_text": "Direct quote from content",
      "confidence": "high/medium/low"
    }
  ],
  "challenges": [
    {
      "challenge": "Description of implementation challenge",
      "mitigation": "How to address it (if mentioned)",
      "source_text": "Direct quote from content",
      "confidence": "high/medium/low"
    }
  ],
  "relevance_score": 0.0-1.0
}
"""
}

# Enhanced synthesis prompt with better structure
SYNTHESIS_PROMPT: Final[str] = """Create a comprehensive synthesis of research findings for: {query}

REQUIREMENTS:
1. Structure your synthesis with these EXACT sections:
   - Domain Overview: Essential context and background
   - Market Dynamics: Size, growth, trends, competition
   - Provider Landscape: Key vendors and their positioning
   - Technical Requirements: Specifications and standards
   - Regulatory Landscape: Compliance and legal requirements
   - Implementation Factors: Resources, process, challenges
   - Cost Considerations: Pricing, ROI, financial factors
   - Best Practices: Recommended approaches

2. For EACH section:
   - Synthesize insights from multiple sources when available
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

AVAILABLE RESEARCH:
{research_json}

FORMAT RESPONSE AS JSON:
{{
  "synthesis": {{
    "domain_overview": {{ "content": "", "citations": [] }},
    "market_dynamics": {{ "content": "", "citations": [] }},
    "provider_landscape": {{ "content": "", "citations": [] }},
    "technical_requirements": {{ "content": "", "citations": [] }},
    "regulatory_landscape": {{ "content": "", "citations": [] }},
    "implementation_factors": {{ "content": "", "citations": [] }},
    "cost_considerations": {{ "content": "", "citations": [] }},
    "best_practices": {{ "content": "", "citations": [] }}
  }},
  "confidence_assessment": {{
    "overall_score": 0.0-1.0,
    "section_scores": {{
      "domain_overview": 0.0-1.0,
      "market_dynamics": 0.0-1.0,
      "provider_landscape": 0.0-1.0,
      "technical_requirements": 0.0-1.0,
      "regulatory_landscape": 0.0-1.0,
      "implementation_factors": 0.0-1.0,
      "cost_considerations": 0.0-1.0,
      "best_practices": 0.0-1.0
    }},
    "limitations": [],
    "knowledge_gaps": []
  }}
}}

REMEMBER:
- Prioritize factual accuracy over comprehensiveness
- Only include claims that are supported by the research
- Use clear, concise language focused on business impact
- Highlight conflicting information when present
"""

# Enhanced validation prompt with adaptive thresholds
VALIDATION_PROMPT: Final[str] = """Validate the research synthesis against these criteria:

VALIDATION CRITERIA:
1. Factual Accuracy
   - Does each claim have proper citation?
   - Are the citations from credible sources?
   - Are claims consistent with the source material?

2. Comprehensive Coverage
   - Are all required sections populated?
   - Is the depth appropriate for each section?
   - Are there any significant knowledge gaps?

3. Source Quality
   - Are sources diverse and authoritative?
   - Are recent sources used where appropriate?
   - Is there over-reliance on any single source?

4. Overall Quality
   - Is confidence assessment realistic?
   - Are limitations properly acknowledged?
   - Is the synthesis balanced and objective?

RESEARCH SYNTHESIS TO VALIDATE:
{synthesis_json}

FORMAT RESPONSE AS JSON:
{{
  "validation_results": {{
    "is_valid": true/false,
    "validation_score": 0.0-1.0,
    "section_validations": {{
      "domain_overview": {{ "is_valid": true/false, "issues": [] }},
      "market_dynamics": {{ "is_valid": true/false, "issues": [] }},
      "provider_landscape": {{ "is_valid": true/false, "issues": [] }},
      "technical_requirements": {{ "is_valid": true/false, "issues": [] }},
      "regulatory_landscape": {{ "is_valid": true/false, "issues": [] }},
      "implementation_factors": {{ "is_valid": true/false, "issues": [] }},
      "cost_considerations": {{ "is_valid": true/false, "issues": [] }},
      "best_practices": {{ "is_valid": true/false, "issues": [] }}
    }},
    "critical_issues": [],
    "improvement_suggestions": []
  }},
  "adaptive_threshold": {{
    "minimum_valid_sections": 0-8,
    "required_sections": [],
    "section_weights": {{
      "domain_overview": 0.0-1.0,
      "market_dynamics": 0.0-1.0,
      "provider_landscape": 0.0-1.0,
      "technical_requirements": 0.0-1.0,
      "regulatory_landscape": 0.0-1.0,
      "implementation_factors": 0.0-1.0,
      "cost_considerations": 0.0-1.0,
      "best_practices": 0.0-1.0
    }}
  }}
}}

IMPORTANT:
- Calculate "minimum_valid_sections" based on query complexity and available data
- Identify critical sections as "required_sections" based on query intent
- Assign weights to sections based on importance to the query
- A synthesis can be valid even with some sections incomplete if priority sections are solid
- Flag fabricated or unsupported claims as critical issues
"""

# Enhanced report template with executive summary format
REPORT_TEMPLATE: Final[str] = """
# Research Report: {query}

## Executive Summary
{executive_summary}

## Key Findings
{key_findings}

## Detailed Analysis

### Market Dynamics
{market_dynamics}

### Provider Landscape
{provider_landscape}

### Technical Requirements
{technical_requirements}

### Regulatory Landscape
{regulatory_landscape}

### Implementation Considerations
{implementation_factors}

### Cost Analysis
{cost_considerations}

### Best Practices
{best_practices}

## Recommendations
{recommendations}

## Sources and Citations
{sources}

---
Confidence Score: {confidence_score}
Generated: {generation_date}
"""

# Enhanced clarity request prompt
CLARIFICATION_PROMPT: Final[str] = """I'm analyzing your research request: "{query}"

Based on my initial analysis, I need some additional context to provide you with the most relevant research.

What I understand so far:
- Product/Service Focus: {product_vs_service}
- Industry Context: {industry_context}
- Geographical Scope: {geographical_focus}

To deliver more precise and comprehensive research, I need clarification on:

{missing_sections}

Could you please provide these additional details? This will help me focus the research on your specific needs rather than making assumptions.

Even with partial clarification, I can begin the research process and refine as we go."""

# Category-specific search quality thresholds
SEARCH_QUALITY_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "market_dynamics": {
        "min_sources": 3,
        "min_facts": 5,
        "recency_threshold_days": 180,  # Market data needs to be recent
        "authoritative_source_ratio": 0.5  # At least half from authoritative sources
    },
    "provider_landscape": {
        "min_sources": 3,
        "min_facts": 3,
        "recency_threshold_days": 365,
        "authoritative_source_ratio": 0.3
    },
    "technical_requirements": {
        "min_sources": 2,
        "min_facts": 3,
        "recency_threshold_days": 730,  # Technical specs can be older
        "authoritative_source_ratio": 0.7  # Need highly authoritative sources
    },
    "regulatory_landscape": {
        "min_sources": 2,
        "min_facts": 2,
        "recency_threshold_days": 730,
        "authoritative_source_ratio": 0.8  # Regulatory info needs official sources
    },
    "cost_considerations": {
        "min_sources": 2,
        "min_facts": 3,
        "recency_threshold_days": 365,  # Pricing should be recent
        "authoritative_source_ratio": 0.4
    },
    "best_practices": {
        "min_sources": 2,
        "min_facts": 3,
        "recency_threshold_days": 730,
        "authoritative_source_ratio": 0.5
    },
    "implementation_factors": {
        "min_sources": 2,
        "min_facts": 3,
        "recency_threshold_days": 730,
        "authoritative_source_ratio": 0.4
    }
}

# Helper function for creating category-specific search prompts
def get_extraction_prompt(category: str, query: str, url: str, content: str) -> str:
    """Get the appropriate extraction prompt for a specific category."""
    if category in EXTRACTION_PROMPTS:
        return EXTRACTION_PROMPTS[category].format(
            query=query,
            url=url,
            content=content
        )
    else:
        # Fallback to general extraction prompt
        return EXTRACTION_PROMPTS["market_dynamics"].format(
            query=query,
            url=url,
            content=content
        )

# Prompt for identifying additional research topics
ADDITIONAL_TOPICS_PROMPT: Final[str] = """Based on the current research findings, identify additional topics that would enhance the analysis.

Current Research:
{current_research}

Consider:
1. Related market segments or industries
2. Emerging technologies or trends
3. Regulatory or compliance areas
4. Implementation considerations
5. Cost factors
6. Best practices

Format your response as JSON:
{
    "additional_topics": [
        {
            "topic": "Topic name",
            "relevance": 0.0-1.0,
            "rationale": "Why this topic is important"
        }
    ],
    "priority_order": ["topic1", "topic2", ...],
    "estimated_effort": {
        "topic1": "high/medium/low",
        "topic2": "high/medium/low",
        ...
    }
}"""

# Base research prompt
RESEARCH_BASE_PROMPT: Final[str] = """You are a Research Agent focused on gathering comprehensive market intelligence.

Your task is to analyze the following query and provide detailed research findings.

Query: {query}

INSTRUCTIONS:
1. Break down the query into research components
2. Identify key areas for investigation
3. Gather relevant market data
4. Analyze trends and patterns
5. Synthesize findings

RESPONSE FORMAT:
{
    "research_components": ["Component 1", "Component 2"],
    "key_findings": ["Finding 1", "Finding 2"],
    "sources": ["Source 1", "Source 2"],
    "confidence_score": 0.0
}"""

# Research agent prompt
RESEARCH_AGENT_PROMPT: Final[str] = """You are an advanced Research Agent specialized in market analysis.

Your task is to conduct comprehensive research on the following topic.

Topic: {topic}

INSTRUCTIONS:
1. Identify key research areas
2. Gather market intelligence
3. Analyze trends and patterns
4. Evaluate sources and credibility
5. Synthesize findings

RESPONSE FORMAT:
{
    "research_areas": ["Area 1", "Area 2"],
    "findings": ["Finding 1", "Finding 2"],
    "sources": ["Source 1", "Source 2"],
    "confidence_score": 0.0
}"""

# Topics prompt
TOPICS_PROMPT: Final[str] = """Analyze the following query to identify key research topics.

Query: {query}

INSTRUCTIONS:
1. Break down the query into main topics
2. Identify subtopics for each main topic
3. Prioritize topics by relevance
4. Consider industry context
5. Note any specialized areas

RESPONSE FORMAT:
{
    "main_topics": ["Topic 1", "Topic 2"],
    "subtopics": {
        "Topic 1": ["Subtopic 1", "Subtopic 2"],
        "Topic 2": ["Subtopic 1", "Subtopic 2"]
    },
    "priority_order": ["Topic 1", "Topic 2"],
    "specialized_areas": ["Area 1", "Area 2"]
}"""

# Export all prompts and utilities
__all__ = [
    "STRUCTURED_OUTPUT_VALIDATION",
    "QUERY_ANALYSIS_PROMPT",
    "EXTRACTION_PROMPTS",
    "SYNTHESIS_PROMPT",
    "VALIDATION_PROMPT",
    "REPORT_TEMPLATE",
    "CLARIFICATION_PROMPT",
    "SEARCH_QUALITY_THRESHOLDS",
    "get_extraction_prompt",
    "ADDITIONAL_TOPICS_PROMPT",
    "RESEARCH_BASE_PROMPT",
    "RESEARCH_AGENT_PROMPT",
    "TOPICS_PROMPT"
]