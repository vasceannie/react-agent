# Derivation Tools

This document explains the refactored derivation tools that integrate functions from `extraction.py` and `statistics.py` into consolidated tools within `derivation.py`.

## Overview

The refactoring consolidates multiple utility functions into three versatile tools:

1. **ExtractionTool** - Combines statistics, citations, category extraction, and fact enrichment
2. **StatisticsAnalysisTool** - Combines quality scoring, source assessment, confidence calculation
3. **ResearchValidationTool** - Combines content validation, fact cross-validation, source quality

Each tool uses an `operation` parameter to route to the appropriate functionality, reducing the number of separate tools while maintaining a clean, type-safe implementation.

## Tool Usage

### ExtractionTool

The ExtractionTool combines various extraction operations:

```python
from react_agent.tools.derivation import ExtractionTool

# Initialize the tool
extraction_tool = ExtractionTool()

# Extract statistics
statistics = extraction_tool.run(
    operation="statistics",
    text="According to a recent survey, 75% of enterprises adopted cloud computing.",
    url="https://example.com",
    source_title="Cloud Report"
)

# Extract citations
citations = extraction_tool.run(
    operation="citations",
    text="According to a recent survey by TechCorp, 75% of enterprises adopted cloud computing."
)

# Extract category information (requires async execution or event loop)
from react_agent.utils.llm import get_extraction_model
extraction_model = get_extraction_model()

# Using event loop for synchronous execution
import asyncio
loop = asyncio.get_event_loop()
category_result = loop.run_until_complete(extraction_tool._arun(
    operation="category",
    text="Cloud computing market is growing rapidly...",
    url="https://example.com",
    source_title="Cloud Report",
    category="market_dynamics",
    original_query="cloud computing market trends",
    extraction_model=extraction_model
))

# Enrich a fact
fact = {"text": "75% of enterprises adopted cloud computing in 2023"}
enriched_fact = extraction_tool.run(
    operation="enrich",
    fact=fact,
    url="https://example.com",
    source_title="Cloud Report"
)
```

### StatisticsAnalysisTool

The StatisticsAnalysisTool combines various statistical analysis operations:

```python
from react_agent.tools.derivation import StatisticsAnalysisTool

# Initialize the tool
statistics_tool = StatisticsAnalysisTool()

# Calculate quality score
quality_score = statistics_tool.run(
    operation="quality",
    category="market_dynamics",
    facts=[{"text": "Market grew by 25%"}],
    sources=[{"url": "https://example.com"}],
    thresholds={"min_facts": 1}
)

# Assess authoritative sources
authoritative_sources = statistics_tool.run(
    operation="sources",
    sources=[{"url": "https://example.com"}, {"url": "https://example.gov"}]
)

# Calculate confidence
confidence = statistics_tool.run(
    operation="confidence",
    category_scores={"market_dynamics": 0.8, "provider_landscape": 0.7},
    synthesis_quality=0.85,
    validation_score=0.9
)

# Assess synthesis quality
synthesis_quality = statistics_tool.run(
    operation="synthesis",
    synthesis={"synthesis": {"section1": {"content": "Detailed analysis..."}}}
)

# Validate statistics
validation_result = statistics_tool.run(
    operation="validate",
    facts=[{"text": "Market grew by 25%", "source_text": "The market grew by 25% in 2023"}]
)

# Assess fact consistency
consistency_score = statistics_tool.run(
    operation="consistency",
    facts=[{"text": "Market grew by 25%"}, {"text": "Growth was 25% in 2023"}]
)
```

### ResearchValidationTool

The ResearchValidationTool combines various validation operations:

```python
from react_agent.tools.derivation import ResearchValidationTool

# Initialize the tool
validation_tool = ResearchValidationTool()

# Validate content
content_validation = validation_tool.run(
    operation="content",
    content="Valid research content...",
    url="https://example.com"
)

# Validate facts
facts_validation = validation_tool.run(
    operation="facts",
    facts=[{"text": "Fact 1"}, {"text": "Fact 2"}]
)

# Assess source quality
sources_validation = validation_tool.run(
    operation="sources",
    sources=[{"url": "https://example.com", "quality_score": 0.8}]
)

# Calculate overall score
overall_score = validation_tool.run(
    operation="overall",
    scores={"quality": 0.8, "recency": 0.7, "consistency": 0.9}
)
```

## Integration with LangGraph

The tools can be easily integrated with LangGraph using the provided factory function:

```python
from langgraph.graph import StateGraph
from react_agent.tools.derivation import create_derivation_toolnode

# Create a state graph
workflow = StateGraph(YourStateType)

# Add all derivation tools as a node
workflow.add_node("derivation_tools", create_derivation_toolnode())

# Or add specific tools
workflow.add_node("extraction_tools", create_derivation_toolnode(include_tools=["extraction"]))
```

## Backward Compatibility

The original tool implementations are maintained for backward compatibility:

- `StatisticsExtractionTool`
- `CitationExtractionTool`
- `CategoryExtractionTool`

However, they include deprecation notices in their docstrings to guide users to the new consolidated tools.

## Examples

For complete examples of how to use these tools, see:

- `examples/derivation_tools_example.py` - Demonstrates basic usage of all tools
- `tests/unit_tests/tools/test_derivation.py` - Contains unit tests for the tools

## Benefits of Consolidated Approach

1. **Reduced Complexity**: Fewer tools to manage and maintain
2. **Improved Discoverability**: Related functions grouped together
3. **Consistent Interface**: Similar operations use consistent patterns
4. **Easier Integration**: Fewer tools to integrate into workflows
5. **Better Documentation**: Consolidated documentation for related operations