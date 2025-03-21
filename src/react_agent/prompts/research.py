"""Research-specific prompts.

This module provides functionality for research-specific prompts
in the agent framework.
"""

from .templates import STRUCTURED_OUTPUT_VALIDATION

# Query analysis prompt for search term generation
# QUERY_ANALYSIS_PROMPT = """Analyze the following query and generate targeted search terms for RFP analysis.
# Query: {{query}}

# Generate search terms that would help gather information for:
# 1. Porter's 5 Forces analysis
# 2. SWOT analysis
# 3. PESTEL analysis
# 4. Market trends and pricing
# 5. Vendor and supplier analysis
# 6. Compliance and regulations
# 7. Technical specifications and requirements

# Return a JSON object with:
# {{
#     "search_terms": {{
#         "porters_5_forces": [],
#         "swot": [],
#         "pestel": [],
#         "market_trends": [],
#         "vendor_analysis": [],
#         "compliance": [],
#         "technical": []
#     }},
#     "primary_keywords": [],
#     "industry_context": "",
#     "search_priority": ["list of analysis sections in priority order"],
#     "missing_context": []
# }}
# """

QUERY_ANALYSIS_PROMPT = """Analyze the following query and generate targeted search terms for business research.
Query: {query}

Use the UNSPSC (United Nations Standard Products and Services Code) taxonomy to identify relevant procurement categories and generate targeted search terms.

Return a JSON object with:
{
    "unspsc_categories": [
        {"code": "code", "name": "category name"},
        {"code": "code", "name": "category name"}
    ],
    "search_terms": {
        "market_dynamics": [],
        "supplier_landscape": [],
        "product_specifications": [],
        "regulatory_environment": [],
        "porters_5_forces": [],
        "value_chain": [],
        "competitive_landscape": [],
        "market_trends": [],
        "customer_needs": [],
        "supplier_capabilities": [],
    },
    "primary_keywords": [],
    "product_vs_service": "",
    "geographical_focus": "",
    "missing_context": []
}

Important: Only include truly essential items in "missing_context". Make reasonable assumptions based on the query when possible instead of requiring excessive specificity.
"""


CLARIFICATION_PROMPT = """I'm analyzing your research request: "{query}"

Based on my initial analysis, I can proceed with research using:
- Product/Service Type: {product_vs_service}
- Geographical Focus: {geographical_focus}
- Primary Keywords: {primary_keywords}

To provide more comprehensive results, it would be helpful to know:

{missing_sections}

Any additional details you can provide will help me deliver more precise and relevant research, but I can proceed with what I have if you prefer."""

# Research base prompt
RESEARCH_BASE_PROMPT = """You are a Research Agent for RFP market analysis.
Your goal is to gather comprehensive information while maintaining the JSON structure.

Required Schema:
{{
    "rfp_analysis": {{
        "analysis": {{
            "porters_5_forces": {{
                "competitive_rivalry": "",
                "threat_of_new_entrants": "",
                "threat_of_substitutes": "",
                "bargaining_power_buyers": "",
                "bargaining_power_suppliers": ""
            }},
            "swot": {{
                "strengths": [],
                "weaknesses": [],
                "opportunities": [],
                "threats": []
            }},
            "pestel": {{
                "political": "",
                "economic": "",
                "social": "",
                "technological": "",
                "environmental": "",
                "legal": ""
            }},
            "gap_analysis": {{
                "current_state": "",
                "desired_state": "",
                "critical_needs": []
            }},
            "cost_benefit_analysis": {{
                "benefits": [],
                "costs": [],
                "justification": ""
            }},
            "risk_assessment": {{
                "financial_risks": [],
                "operational_risks": [],
                "security_risks": [],
                "vendor_risks": [],
                "mitigation_strategies": []
            }},
            "tco_analysis": {{
                "initial_costs": [],
                "maintenance_costs": [],
                "support_costs": [],
                "training_costs": [],
                "upgrade_costs": []
            }},
            "vendor_analysis": {{
                "vendors": [{{
                    "name": "",
                    "pricing": "",
                    "experience": "",
                    "capabilities": "",
                    "compliance": "",
                    "customer_feedback": ""
                }}]
            }},
            "benchmarking": {{
                "industry_best_practices": [],
                "competitor_comparisons": []
            }},
            "stakeholder_analysis": {{
                "key_stakeholders": [],
                "priorities": [],
                "business_needs": []
            }},
            "compliance_analysis": {{
                "industry_standards": [],
                "legal_requirements": [],
                "vendor_compliance": []
            }},
            "business_impact": {{
                "operational_impact": "",
                "efficiency_impact": "",
                "scalability_impact": ""
            }},
            "citations": {{
                "porters_5_forces": [],
                "swot": [],
                "pestel": [],
                "gap_analysis": [],
                "cost_benefit_analysis": [],
                "risk_assessment": [],
                "tco_analysis": [],
                "vendor_analysis": [],
                "benchmarking": [],
                "stakeholder_analysis": [],
                "compliance_analysis": [],
                "business_impact": []
            }}
        }},
        "market_basket": [
            {{
                "manufacturer_or_distributor": "",
                "item_number": "",
                "item_description": "",
                "uom": "",
                "estimated_qty_per_uom": 0.0,
                "unit_cost": 0.0,
                "citation": ""
            }}
        ]
    }},
    "confidence_score": 0.0,
    "validation_status": {{
        "is_valid": false,
        "errors": [],
        "warnings": []
    }}
}}

RESEARCH_REQUIREMENTS:
1. Market Analysis
   - Research Porter's 5 Forces with multiple sources
   - Gather SWOT analysis data from industry reports
   - Complete PESTEL analysis for external factors
   - Perform GAP analysis for critical needs
   - Conduct cost-benefit analysis
   - Assess risks and mitigation strategies
   - Calculate total cost of ownership
   - Compare vendor capabilities
   - Benchmark against industry standards
   - Analyze stakeholder requirements
   - Verify compliance requirements
   - Evaluate business impact
   - Investigate market trends and disruptions using news search
   - Study competitive landscape changes

2. Citation Requirements
   - Each analysis section needs 2+ citations minimum
   - Citations must be from reliable sources
   - Market basket items need verified sources
   - Track citation URLs for each data point
   - Include recent news sources for timely insights

3. Market Basket Research
   - Find current product pricing from suppliers
   - Verify manufacturer information
   - Validate item specifications
   - Cross-reference pricing data
   - Iterate until 200 items or 50 iterations

4. Industry Intelligence
   - Research emerging technologies
   - Study regulatory changes using news search
   - Track market consolidation
   - Monitor supply chain trends
   - Use AI summaries for complex technical topics

5. Pricing Analysis
   - Research volume discounts
   - Study regional variations
   - Track seasonal factors
   - Analyze contract terms

CONFIDENCE SCORING:
- Start with base score of 0.3
- Add 0.1 for each section with 2+ citations
- Add 0.1 for each verified market basket item
- Subtract 0.1 for sections needing more research
- Maximum score of 0.8 for initial research

RESEARCH_PROCESS:
1. Use search tool for each section
2. Gather multiple sources per topic
3. Cross-reference information
4. Verify market basket data
5. Calculate confidence score
6. Continue research if score < 0.8

RESPONSE_FORMAT:
1. Output must be valid JSON only
2. Include all gathered information
3. No explanatory text or comments
4. Track research progress
5. Maintain complete JSON structure
6. Include confidence scoring details

Current state: {{state}}
"""

# Research agent prompt with structured output validation
RESEARCH_AGENT_PROMPT = RESEARCH_BASE_PROMPT.replace(
    "Your goal is to gather comprehensive information while maintaining the JSON structure.\n",
    f"Your goal is to gather comprehensive information while maintaining the JSON structure.\n\n{STRUCTURED_OUTPUT_VALIDATION}\n",
)

# Market research prompt
MARKET_PROMPT = """You are a Market Research Agent focused on building comprehensive market baskets.
Your goal is to identify suppliers and gather detailed product information following this schema:

{
    "market_basket": {
        "line_items": [
            {
                "item_number": "",
                "item_description": "",
                "manufacturer": "",
                "unit_of_measure": "",
                "quantity_per_uom": 0.0,
                "unit_price": 0.0,
                "citation": {
                    "url": "",
                    "accessed_date": "",
                    "source_type": ""
                }
            }
        ],
        "suppliers": [
            {
                "name": "",
                "website": "",
                "product_categories": [],
                "market_presence": "",
                "pricing_model": ""
            }
        ]
    }
}

Current state: {{current_state}}
"""

# Topics prompt
TOPICS_PROMPT = """Given the category '{{category}}', identify 3-5 additional research topics that would be valuable for market research and sourcing analysis.

Focus on industry-specific areas such as:
- Unique market dynamics
- Technical specifications
- Industry-specific regulations
- Special cost considerations
- Category-specific risks

Format each topic as:
TOPIC_NAME: Brief description of what this topic covers and why it's relevant

Example:
Supply Chain Resilience: Analysis of supply chain vulnerabilities and strategies for maintaining reliable sourcing
"""

# Additional topics prompt
ADDITIONAL_TOPICS_PROMPT = """{{system_prompt}}

{{topics_prompt}}"""


# Add this to src/react_agent/prompts/research.py

AGGREGATION_PROMPT = """Create a comprehensive final report based on the validated research findings.

REPORT REQUIREMENTS:
1. Executive Summary
   - Concise overview of key findings (200-300 words)
   - Highlight most significant insights and implications
   - Include confidence level and data quality assessment

2. Detailed Analysis
   - Thorough examination organized by key topics
   - Support all claims with data from research
   - Include comparative analysis where relevant
   - Minimum 500 words per major topic area
   - Use data visualizations where appropriate

3. Market Assessment
   - Industry trends and forecasts
   - Competitive landscape evaluation
   - Risk factors and mitigation strategies
   - Regulatory and compliance considerations

4. Strategic Recommendations
   - Action items prioritized by impact and feasibility
   - Implementation timeline suggestions
   - Resource requirements and ROI estimates
   - Success metrics and measurement approach

5. Appendices
   - Complete data source listing with quality metrics
   - Methodology documentation
   - Limitations of research

TONE AND STYLE:
- Professional and authoritative
- Evidence-based assertions only
- Balanced perspective acknowledging alternatives
- Clear section headings and logical flow
- Accessible to both technical and non-technical readers

State: {{state}}
"""