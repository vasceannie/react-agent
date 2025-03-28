"""
Example script demonstrating how to use the consolidated derivation tools.

This script shows how to use the new ExtractionTool, StatisticsAnalysisTool,
and ResearchValidationTool for various operations.
"""

import asyncio
from typing import Dict, Any, List

from react_agent.tools.derivation import (
    ExtractionTool,
    StatisticsAnalysisTool,
    ResearchValidationTool,
    create_derivation_toolnode
)
from react_agent.utils.llm import get_extraction_model


async def main():
    """Run the example script."""
    print("Demonstrating the consolidated derivation tools")
    print("=" * 50)
    
    # Sample text for extraction
    sample_text = """
    According to a recent survey by TechCorp, 75% of enterprises adopted cloud
    computing in 2023, up from 60% in 2022. The global cloud market reached
    $483.3 billion in revenue, with AWS maintaining a 32% market share.
    A separate study by MarketWatch revealed that cybersecurity spending
    increased by 15% year-over-year.
    """
    
    # Initialize the tools
    extraction_tool = ExtractionTool()
    statistics_tool = StatisticsAnalysisTool()
    validation_tool = ResearchValidationTool()
    
    # 1. Extract statistics
    print("\n1. Extracting statistics:")
    statistics = extraction_tool.run(
        operation="statistics",
        text=sample_text,
        url="https://example.com",
        source_title="Cloud Computing Report"
    )
    print(f"Found {len(statistics)} statistics:")
    for stat in statistics:
        print(f"  - {stat.get('text', '')}")
    
    # 2. Extract citations
    print("\n2. Extracting citations:")
    citations = extraction_tool.run(
        operation="citations",
        text=sample_text
    )
    print(f"Found {len(citations)} citations:")
    for citation in citations:
        print(f"  - {citation.get('source', '')}: {citation.get('context', '')}")
    
    # 3. Enrich a fact
    print("\n3. Enriching a fact:")
    fact = {"text": "75% of enterprises adopted cloud computing in 2023"}
    enriched_fact = extraction_tool.run(
        operation="enrich",
        fact=fact,
        url="https://example.com",
        source_title="Cloud Computing Report"
    )
    print(f"Enriched fact: {enriched_fact}")
    
    # 4. Validate content
    print("\n4. Validating content:")
    validation_result = validation_tool.run(
        operation="content",
        content=sample_text,
        url="https://example.com"
    )
    print(f"Content validation result: {validation_result}")
    
    # 5. Assess fact consistency
    print("\n5. Assessing fact consistency:")
    facts = [
        {"text": "75% of enterprises adopted cloud computing in 2023"},
        {"text": "Cloud market reached $483.3 billion in revenue"},
        {"text": "AWS maintains 32% market share"}
    ]
    consistency_result = validation_tool.run(
        operation="facts",
        facts=facts
    )
    print(f"Fact consistency result: {consistency_result}")
    
    # 6. Calculate quality score
    print("\n6. Calculating quality score:")
    sources = [
        {"url": "https://example.com", "quality_score": 0.8},
        {"url": "https://example.org", "quality_score": 0.7}
    ]
    thresholds = {
        "min_facts": 3,
        "min_sources": 2,
        "authoritative_source_ratio": 0.5,
        "recency_threshold_days": 365
    }
    
    # This would normally use the actual calculate_category_quality_score function
    # but for demonstration purposes, we'll just print the parameters
    print("Would calculate quality score with:")
    print(f"  - Category: market_dynamics")
    print(f"  - Facts: {len(facts)} facts")
    print(f"  - Sources: {len(sources)} sources")
    print(f"  - Thresholds: {thresholds}")
    
    # 7. Demonstrate async usage with category extraction
    print("\n7. Demonstrating async category extraction:")
    # Get an extraction model (this is just for demonstration)
    extraction_model = get_extraction_model()
    
    # Run the category extraction asynchronously
    category_result = await extraction_tool._arun(
        operation="category",
        text=sample_text,
        url="https://example.com",
        source_title="Cloud Computing Report",
        category="market_dynamics",
        original_query="cloud computing market trends",
        extraction_model=extraction_model
    )
    print(f"Category extraction result: {category_result}")
    
    # 8. Demonstrate creating a ToolNode
    print("\n8. Creating a ToolNode with all tools:")
    tool_node = create_derivation_toolnode()
    print(f"Created ToolNode with {len(tool_node.tools)} tools:")
    for tool in tool_node.tools:
        print(f"  - {tool.name}: {tool.description}")


if __name__ == "__main__":
    asyncio.run(main())