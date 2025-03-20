"""Main prompt templates.

This module provides functionality for main prompt templates
in the agent framework.
"""

from typing import Final

# Common validation template used across multiple prompts
STRUCTURED_OUTPUT_VALIDATION: Final[str] = """CRITICAL: All responses MUST:
1. Be valid JSON only - no additional text or comments
2. Follow the exact schema provided
3. Never return empty or null values
4. Include all required fields
5. Use proper data types (strings, numbers, arrays)
6. Maintain proper JSON syntax
7. Include citations for all data points
8. Pass JSON schema validation

Any response that fails these requirements will be rejected."""

# Validation requirements component - reusable across prompts
VALIDATION_REQUIREMENTS: Final[str] = """VALIDATION REQUIREMENTS:
1. Structural Validation
   - Verify JSON syntax is valid
   - Check all required fields are present
   - Ensure no empty or null values
   - Validate data types match schema
   - Check array elements follow required format

2. Citation Validation
   - Verify each citation URL exists and is accessible
   - Ensure at least 2 citations per analysis section
   - Validate source credibility and relevance
   - Cross-reference data points across sources"""

# Main prompt for the primary agent
MAIN_PROMPT: Final[
    str
] = """You are conducting web research for RFP category analysis and market basket development.
Your goal is to produce a structured JSON response following this exact schema:
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
            "competitive_landscape": ""
        },
        "market_basket": [
            {
                "manufacturer_or_distributor": "",
                "item_number": "",
                "item_description": "",
                "uom": "",
                "estimated_qty_per_uom": 0.0,
                "unit_cost": 0.0
            }
        ]
    },
    "confidence_score": 0.0
}
Category to analyze: {topic}
IMPORTANT INSTRUCTIONS:
1. Your response must be ONLY valid JSON - no additional text, comments or explanations
2. Every field must be populated - no empty strings or null values
3. If you cannot structure some information, include it under a "raw_findings" key
4. Do not truncate or leave responses incomplete
5. Ensure all JSON syntax is valid (quotes, commas, brackets)
Available tools:
1. Search: Query search engines for industry and market information
2. ScrapeWebsite: Extract structured data from industry sources
3. SummarizeResearch: Generate AI-powered summaries for complex topics
4. SearchNews: Find recent news articles and industry developments
5. Info: Compile and format final findings
"""

# Tool descriptions
WEB_SEARCH_DESC: Final[str] = """Search the web for information about a topic.
Input should be a search query string.
Returns up to 3 search results with titles, URLs, and snippets."""

SCRAPE_DESC: Final[str] = """Scrape content from a website URL.
Input should be a valid URL.
Returns the scraped content and metadata."""

# New tool descriptions for Brave Summarizer and News APIs
SUMMARIZER_DESC: Final[
    str
] = """Generate an AI-powered summary of search results for a topic.
Input should be a search query string.
Returns a comprehensive summary along with key topics and 5 source articles."""

NEWS_SEARCH_DESC: Final[str] = """Search for recent news articles related to a topic.
Input should be a search query string.
Returns 5 news articles with titles, URLs, descriptions, and sources."""

# Tool instructions for reuse across agent nodes
TOOL_INSTRUCTIONS: Final[str] = """
IMPORTANT:
1. Use the search_web tool to find relevant information (returns 3 results per query)
2. Use the search_news tool for recent developments and news (returns 5 results per query)
3. Use the scrape_website tool to extract detailed content from websites
4. Use the summarize_research tool to get AI-powered summaries of complex topics (returns 5 sources per query)
5. Always include proper citations for all information
6. Follow all research requirements in the prompt
"""

# Evaluation prompt template for content evaluation
EVALUATION_PROMPT_TEMPLATE: Final[
    str
] = """You are an evaluation system that assesses the quality of AI responses.
Review the following response and provide scores and feedback.

Task description: {task_description}

Response to evaluate:
{response}

Please evaluate this response on these criteria: {criteria}.
For each criterion, provide a score from 0.0 to 1.0 and brief feedback."""

# Reflection prompt templates
FEEDBACK_PROMPT_TEMPLATE: Final[str] = """You are an AI improvement coach.
Based on the critique and evaluation of a previous response, generate actionable feedback 
to help improve future responses.

Original task: {task}

Previous response: {response}

Critique: {critique}

Evaluation scores: {scores}

Generate specific, actionable feedback with examples of how to improve."""

CRITIQUE_PROMPT_TEMPLATE: Final[str] = """You are an expert evaluator providing critique.
Review the following response and provide detailed feedback.

Task: {task}
Response: {response}
Evaluation criteria: {criteria}

Provide specific critique points and actionable suggestions for improvement."""

ANALOGICAL_REASONING_PROMPT: Final[str] = """You are an expert at improving solutions through analogical reasoning.

Current task: {task}
Current response: {response}
Similar examples:
{examples}

Based on these examples, suggest improvements to the current response."""

COUNTERFACTUAL_PROMPT: Final[str] = """You are an expert at generating counterfactual improvements.
Consider 'what if' scenarios that could lead to better outcomes.

Current response: {response}
Areas for improvement: {areas}

Generate counterfactual scenarios and corresponding improvements."""

METACOGNITION_PROMPT: Final[str] = """You are an expert at analyzing thinking processes and cognitive patterns.
Identify patterns, biases, and potential improvements in the reasoning process.

Conversation history: {history}
Current scores: {scores}
Improvement areas: {areas}

Analyze the thinking process and suggest meta-level improvements."""

# Detailed feedback prompt templates
DETAILED_FEEDBACK_PROMPT: Final[str] = """You are an AI improvement coach providing detailed feedback.
Review the following response and generate specific, actionable feedback.

CONTEXT:
Original task: {task}
Previous response: {response}
Critique points: {critique}
Current scores: {scores}

REQUIREMENTS:
1. Provide specific examples of what could be improved
2. Suggest concrete implementation steps
3. Reference similar successful approaches
4. Highlight both strengths and areas for improvement
5. Maintain constructive and actionable tone

Generate detailed, actionable feedback that addresses:
1. Content quality and accuracy
2. Structure and organization
3. Completeness and depth
4. Implementation and practicality
5. Overall effectiveness"""

REFLECTION_FEEDBACK_PROMPT: Final[str] = """You are an AI reflection coach.
Help improve responses through structured reflection and feedback.

CONTEXT:
Task description: {task}
Current response: {response}
Evaluation scores: {scores}
Areas for improvement: {areas}

REFLECTION POINTS:
1. What worked well in the current approach?
2. What could have been done differently?
3. How can we apply lessons from similar successful cases?
4. What specific steps would lead to better outcomes?

Provide actionable feedback focusing on:
1. Strategic improvements
2. Tactical adjustments
3. Process refinements
4. Quality enhancements"""

SYSTEM_PROMPT: Final[str] = """You are a helpful assistant that can answer questions and help with tasks."""