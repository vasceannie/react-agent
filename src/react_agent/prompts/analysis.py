"""Analysis-specific prompts.

This module provides functionality for analysis-specific prompts
in the agent framework.
"""

from typing import Final

# Prompt for tool selection
TOOL_SELECTION_PROMPT: Final[str] = (
    """What information do we need to research about {current_topic}?"""
)

# Prompt for analysis of tool results
ANALYSIS_PROMPT: Final[str] = """
Based on the following research about {current_topic}, provide a comprehensive analysis:

{formatted_results}

Your analysis should include:
1. Key insights from the research
2. Patterns or trends identified
3. Implications for the business
4. Recommendations based on the findings
"""

# Prompt for analysis plan formulation
ANALYSIS_PLAN_PROMPT: Final[str] = """
Analysis Task: {task}
Available Data:
{data_summary}

Create a comprehensive plan for analyzing this data that will address the task.
Your plan should include:
1. Data preparation steps needed (e.g., cleaning, transformation)
2. Analysis methods to apply (e.g., descriptive statistics, correlation, regression)
3. Visualizations to create (e.g., histograms, scatter plots, bar charts)
4. Hypotheses to test (if applicable)
5. Statistical methods to use (e.g., t-tests, ANOVA, chi-squared)
6. Expected insights (what do you expect to learn from the analysis?)

Format your response as a JSON object with these sections.
"""

# Prompt for results interpretation
INTERPRET_RESULTS_PROMPT: Final[str] = """
Analysis Task: {task}
Analysis Results:
{analysis_results}
Analysis Plan:
{analysis_plan}

Interpret these results in the context of the original task.
Your interpretation should include:
1. Key findings and insights
2. Patterns and trends identified
3. Anomalies or unexpected results
4. Limitations of the analysis
5. Answers to specific questions in the task (if any)
6. Business implications (if applicable)

Format your response as a JSON object with these sections.
"""

# Prompt for report compilation
COMPILE_REPORT_PROMPT: Final[str] = """
Analysis Task: {task}
Analysis Results:
{analysis_results}
Interpretations:
{interpretations}
Visualizations:
{visualization_metadata}

Compile a comprehensive analysis report addressing the original task.
The report should:
1. Start with an executive summary of key findings.
2. Include an introduction explaining the context and objectives.
3. Describe the methodology and data sources.
4. Present the detailed findings with references to visualizations.
5. Discuss implications and recommendations.
6. Note limitations and potential future analysis.

Format the report as markdown with proper headings, lists, and sections.
"""
