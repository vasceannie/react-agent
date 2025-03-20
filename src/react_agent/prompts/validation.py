"""Validation-specific prompts.

This module provides functionality for validation-specific prompts
in the agent framework.
"""

from typing import Final

# Validation base prompt
VALIDATION_BASE_PROMPT: Final[
    str
] = """You are a Validation Agent for RFP market analysis.
Your goal is to prevent hallucinations and ensure data quality.

{VALIDATION_REQUIREMENTS}

3. Content Validation
   - Verify all required fields are populated
   - Check for data consistency across sections
   - Validate numerical data and calculations
   - Ensure analysis conclusions are supported by data

4. Market Basket Validation
   - Verify product information accuracy
   - Cross-check pricing against multiple sources
   - Validate manufacturer/distributor details
   - Ensure proper unit of measure conversions

5. Analysis Quality
   - Verify PESTEL factors are comprehensive
   - Check GAP analysis identifies clear needs
   - Validate cost-benefit calculations
   - Review risk assessment completeness
   - Cross-check TCO components
   - Verify vendor analysis objectivity
   - Check benchmarking methodology
   - Validate stakeholder identification
   - Ensure compliance requirements are current
   - Verify business impact assessments

CONFIDENCE SCORING:
- Start with base score of 0.4
- Add 0.1 for each validated section with 2+ citations
- Add 0.1 for each verified market basket item
- Add 0.1 for comprehensive analysis coverage
- Subtract 0.1 for each validation failure
- Reject if final score < 0.98

RESPONSE_FORMAT:
{
    "validation_results": {
        "is_valid": false,
        "errors": [],
        "warnings": [],
        "confidence_score": 0.0,
        "section_scores": {
            "structural": 0.0,
            "citations": 0.0,
            "content": 0.0,
            "market_basket": 0.0
        },
        "failed_validations": [],
        "required_fixes": []
    }
}

Current state: {state}
"""

# Validation agent prompt with structured output validation
VALIDATION_AGENT_PROMPT: Final[str] = VALIDATION_BASE_PROMPT.replace(
    "Your goal is to prevent hallucinations and ensure data quality.\n",
    "Your goal is to prevent hallucinations and ensure data quality.\n\n{STRUCTURED_OUTPUT_VALIDATION}\n",
)

# Prompt for generating validation criteria
VALIDATION_CRITERIA_PROMPT: Final[str] = """
Content Type: {content_type}
Generate appropriate validation criteria for content of this type.
The criteria should be comprehensive and tailored to the specific content type.
For example:
- For research content: factual accuracy, source credibility, logical consistency
- For analysis content: methodological soundness, statistical validity, interpretative accuracy
- For code: functional correctness, efficiency, security, readability

Format your response as a JSON object with these fields:
- primary_criteria: List of primary validation criteria (string[])
- secondary_criteria: List of secondary validation criteria (string[])
- critical_requirements: List of must-have elements (string[])
- disqualifying_factors: List of automatic disqualifiers (string[])
- scoring_weights: Dictionary mapping criteria to weights (0.0 to 1.0)
"""

# Prompt for fact checking
FACT_CHECK_CLAIMS_PROMPT: Final[str] = """
Content Type: {content_type}
Analyze the following content for factual accuracy of claims:
{content}

1. Identify claims that are factual, opinion-based, unclear, or contradictory.
2. Provide source citations for each claim.
3. Evaluate the credibility of sources.
4. Verify the accuracy of each claim.
5. Determine if the content as a whole is factually accurate.
6. Identify any potential biases or conflicts of interest.
7. Note any areas where more research is needed.

Respond in JSON format with these fields:
- factually_accurate_claims: string[] (list of factual claims)
- opinion_based_claims: string[] (list of opinion-based claims)
- unclear_claims: string[] (list of unclear claims)
- source_citations: string[] (list of source URLs for each claim)
- source_credibility: string[] (list of source credibility scores)
- verification_results: string[] (list of verification results)
- overall_accuracy: number from 0-10
- potential_biases: string[] (list of potential biases)
- areas_for_future_research: string[] (list of areas for future research)
- issues: string[] (summarizing all critical issues)
"""

# Prompt for validating individual claims
VALIDATE_CLAIM_PROMPT: Final[str] = """
Fact check the following claim:
CLAIM: {claim}

Respond in JSON format with these fields:
- accuracy: number from 0-10
- confidence: number from 0-10
- issues: string[] (empty if no issues)
- verification_notes: string
"""

# Prompt for logic validation
LOGIC_VALIDATION_PROMPT: Final[str] = """
Content Type: {content_type}
Validate the logical consistency, reasoning quality, and argument structure of the following content:
{content}

Analyze for:
1. Valid argument structure (premises, conclusions)
2. Logical fallacies (e.g., circular reasoning, false cause)
3. Consistency between claims
4. Quality of evidence and reasoning
5. Appropriate conclusions

Respond in JSON format with these fields:
- logical_structure_score: number from 0-10
- fallacies_found: string[] (empty if none)
- consistency_issues: string[] (empty if none)
- reasoning_quality: number from 0-10
- conclusion_validity: number from 0-10
- overall_score: number from 0-10
- issues: string[] (summarizing all critical issues)
"""

# Prompt for consistency checking
CONSISTENCY_CHECK_PROMPT: Final[str] = """
Content Type: {content_type}
Check the internal consistency and coherence of the following content:
{content}

Analyze for:
1. Consistency between different sections
2. Coherence of narrative or explanation
3. Presence of contradictions
4. Logical flow and structure
5. Completeness (no missing pieces in the reasoning)

Respond in JSON format with these fields:
- section_consistency: number from 0-10
- coherence_score: number from 0-10
- contradictions: string[] (empty if none)
- flow_quality: number from 0-10
- completeness: number from 0-10
- overall_score: number from 0-10
- issues: string[] (summarizing all critical issues)
- needs_human_review: boolean (true if human review is recommended)
"""

# Prompt for human feedback request
HUMAN_FEEDBACK_PROMPT: Final[str] = """
Content Type: {content_type}
Based on automated validation, the following issues were identified:
{issues}

Generate 3-5 specific questions for human reviewers to address these issues.
Questions should be clear, focused, and help improve the quality of the content.
Additionally, suggest specific sections or aspects that need human attention.

Format your response as JSON with these fields:
- questions: string[] (list of questions)
- focus_areas: string[] (specific aspects needing review)
- content_summary: string (brief summary of the content)
"""

