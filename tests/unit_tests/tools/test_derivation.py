"""Unit tests for the derivation tools module.

This test suite verifies the functionality of the extraction, statistics analysis,
and research validation tools, including their various operations and edge cases.

Examples of test cases:
    - Tool operation validations
    - Parameter validation
    - Error handling
    - Edge cases
"""

from unittest.mock import MagicMock, patch, ANY

import pytest

from react_agent.tools.derivation import (
    ExtractionTool,
    ExtractionInput,
    ResearchValidationTool,
    ResearchValidationInput,
    StatisticsAnalysisTool,
    StatisticsAnalysisInput,
    create_derivation_toolnode,
)


class TestExtractionTool:
    """Tests for the ExtractionTool class.
    
    Test scenarios cover:
    - Statistics extraction
    - Citations extraction
    - Category information extraction
    - Fact enrichment
    - Parameter validation
    - Error handling
    
    Example test cases:
        >>> test_statistics_extraction()
        Verifies statistics extraction operation
        
        >>> test_citations_extraction()
        Verifies citations extraction operation
    """
    
    @pytest.fixture
    def extraction_tool(self):
        """Create an ExtractionTool instance for testing."""
        return ExtractionTool()
    
    def test_statistics_extraction(self):
        """Test statistics extraction operation."""
        # Arrange
        tool = ExtractionTool()
        
        # Mock the extract_statistics function
        with patch('react_agent.tools.derivation.extract_statistics') as mock_extract:
            mock_extract.return_value = [{"text": "75% of enterprises adopted cloud computing"}]
            
            # Act: Call the tool with statistics operation
            tool_input = {
                "operation": "statistics",
                "text": "According to a recent survey, 75% of enterprises adopted cloud computing in 2023.",
                "url": "https://example.com",
                "source_title": "Cloud Report"
            }
            result = tool.run(tool_input)
            
            # Assert: Verify the function was called with correct parameters
            mock_extract.assert_called_once_with(
                "According to a recent survey, 75% of enterprises adopted cloud computing in 2023.",
                "https://example.com",
                "Cloud Report"
            )
            
            # Assert: Verify the result
            assert result == [{"text": "75% of enterprises adopted cloud computing"}]
    
    def test_citations_extraction(self):
        """Test citations extraction operation."""
        # Arrange
        tool = ExtractionTool()
        
        # Mock the extract_citations function
        with patch('react_agent.tools.derivation.extract_citations') as mock_extract:
            mock_extract.return_value = [{"source": "TechCorp", "context": "survey by TechCorp"}]
            
            # Act: Call the tool with citations operation
            tool_input = {
                "operation": "citations",
                "text": "According to a recent survey by TechCorp, 75% of enterprises adopted cloud computing."
            }
            result = tool.run(tool_input)
            
            # Assert: Verify the function was called with correct parameters
            mock_extract.assert_called_once_with(
                "According to a recent survey by TechCorp, 75% of enterprises adopted cloud computing."
            )
            
            # Assert: Verify the result
            assert result == [{"source": "TechCorp", "context": "survey by TechCorp"}]
    
    def test_category_extraction(self):
        """Test category information extraction operation."""
        # Arrange
        tool = ExtractionTool()
        
        # Mock the extract_category_information function and asyncio.get_event_loop
        with patch('react_agent.tools.derivation.extract_category_information') as mock_extract:
            with patch('react_agent.tools.derivation.asyncio.get_event_loop') as mock_loop:
                # Setup mock return values
                mock_extract.return_value = (
                    [{"text": "Market grew by 25%"}],  # facts
                    0.85  # relevance
                )
                mock_loop_instance = MagicMock()
                mock_loop.return_value = mock_loop_instance
                mock_loop_instance.run_until_complete.return_value = (
                    [{"text": "Market grew by 25%"}],  # facts
                    0.85  # relevance
                )
                
                # Act: Call the tool with category operation
                tool_input = {
                    "operation": "category",
                    "text": "The market grew significantly in 2023.",
                    "url": "https://example.com",
                    "source_title": "Market Report",
                    "category": "market_dynamics",
                    "original_query": "market growth 2023",
                    "extraction_model": MagicMock()
                }
                result = tool.run(tool_input)
                
                # Assert: Verify the function was called with correct parameters
                mock_loop_instance.run_until_complete.assert_called_once()
                
                # Assert: Verify the result
                assert result["facts"] == [{"text": "Market grew by 25%"}]
                assert result["relevance"] == 0.85
    
    def test_enrich_fact(self):
        """Test fact enrichment operation."""
        # Arrange
        tool = ExtractionTool()
        
        # Mock the enrich_extracted_fact function
        with patch('react_agent.tools.derivation.enrich_extracted_fact') as mock_enrich:
            mock_enrich.return_value = {
                "text": "75% of enterprises adopted cloud computing",
                "source_url": "https://example.com",
                "source_title": "Cloud Report",
                "confidence_score": 0.9
            }
            
            # Act: Call the tool with enrich operation
            tool_input = {
                "operation": "enrich",
                "fact": {"text": "75% of enterprises adopted cloud computing"},
                "url": "https://example.com",
                "source_title": "Cloud Report"
            }
            result = tool.run(tool_input)
            
            # Assert: Verify the function was called with correct parameters
            mock_enrich.assert_called_once_with(
                {"text": "75% of enterprises adopted cloud computing"},
                "https://example.com",
                "Cloud Report"
            )
            
            # Assert: Verify the result
            assert result["text"] == "75% of enterprises adopted cloud computing"
            assert result["source_url"] == "https://example.com"
            assert result["source_title"] == "Cloud Report"
            assert result["confidence_score"] == 0.9
    
    def test_missing_parameters(self):
        """Test error handling for missing parameters."""
        # Arrange
        tool = ExtractionTool()
        
        # Act & Assert: Test missing text for statistics
        with pytest.raises(ValueError, match="Text is required for statistics extraction"):
            tool.run({"operation": "statistics"})
        
        # Act & Assert: Test missing text for citations
        with pytest.raises(ValueError, match="Text is required for citation extraction"):
            tool.run({"operation": "citations"})
        
        # Act & Assert: Test missing fact for enrichment
        with pytest.raises(ValueError, match="Fact is required for enrichment"):
            tool.run({"operation": "enrich", "url": "https://example.com"})
    
    def test_unknown_operation(self):
        """Test error handling for unknown operation."""
        # Arrange
        tool = ExtractionTool()
        
        # Act & Assert: Test invalid operation
        with pytest.raises(Exception) as excinfo:
            tool.run({"operation": "invalid_op", "text": "Some text"})
        assert "Input should be 'statistics', 'citations', 'category' or 'enrich'" in str(excinfo.value)
    
    def test_async_run(self):
        """Test asynchronous run method."""
        # Arrange
        tool = ExtractionTool()
        
        # Mock the extract_statistics function
        with patch('react_agent.tools.derivation.extract_statistics') as mock_extract:
            mock_extract.return_value = [{"text": "75% of enterprises adopted cloud computing"}]
            
            # Act: Call the async run method
            import asyncio
            
            async def test_async():
                tool_input = {
                    "operation": "statistics",
                    "text": "According to a recent survey, 75% of enterprises adopted cloud computing in 2023.",
                    "url": "https://example.com",
                    "source_title": "Cloud Report"
                }
                return await tool._arun(
                    operation=tool_input["operation"],
                    text=tool_input["text"],
                    url=tool_input["url"],
                    source_title=tool_input["source_title"]
                )
            
            result = asyncio.run(test_async())
            
            # Assert: Verify the result
            assert result == [{"text": "75% of enterprises adopted cloud computing"}]
    
    def test_input_schema(self):
        """Test that the tool has the correct input schema."""
        # Arrange
        tool = ExtractionTool()
        
        # Assert: Verify the input schema
        assert tool.args_schema == ExtractionInput


class TestStatisticsAnalysisTool:
    """Tests for the StatisticsAnalysisTool class.
    
    Test scenarios cover:
    - Quality score calculation
    - Authoritative sources assessment
    - Confidence calculation
    - Synthesis quality assessment
    - Statistical validation
    - Fact consistency assessment
    - Parameter validation
    - Error handling
    
    Example test cases:
        >>> test_quality_score_calculation()
        Verifies quality score calculation operation
        
        >>> test_authoritative_sources()
        Verifies authoritative sources assessment operation
    """
    
    @pytest.fixture
    def statistics_tool(self):
        """Create a StatisticsAnalysisTool instance for testing."""
        return StatisticsAnalysisTool()
    
    def test_quality_score_calculation(self):
        """Test quality score calculation operation."""
        # Arrange
        tool = StatisticsAnalysisTool()
        
        # Mock the calculate_category_quality_score function
        with patch('react_agent.tools.derivation.calculate_category_quality_score') as mock_calc:
            mock_calc.return_value = 0.85
            
            # Act: Call the tool with quality operation
            tool_input = {
                "operation": "quality",
                "category": "market_dynamics",
                "facts": [{"text": "Market grew by 25%"}],
                "sources": [{"url": "https://example.com"}],
                "thresholds": {"min_facts": 1}
            }
            result = tool.run(tool_input)

            # Assert: Verify the function was called with correct parameters
            mock_calc.assert_called_once_with(
                "market_dynamics",
                [{"text": "Market grew by 25%"}],
                [{"url": "https://example.com"}],
                {"min_facts": 1}
            )
            
            # Assert: Verify the result
            assert result == 0.85
    
    def test_authoritative_sources(self):
        """Test authoritative sources assessment operation."""
        # Arrange
        tool = StatisticsAnalysisTool()
        
        # Mock the assess_authoritative_sources function
        with patch('react_agent.tools.derivation.assess_authoritative_sources') as mock_assess:
            mock_assess.return_value = [{"url": "https://example.gov"}]
            
            # Act: Call the tool with sources operation
            tool_input = {
                "operation": "sources",
                "sources": [{"url": "https://example.com"}, {"url": "https://example.gov"}]
            }
            result = tool.run(tool_input)
            
            # Assert: Verify the function was called with correct parameters
            mock_assess.assert_called_once_with(
                [{"url": "https://example.com"}, {"url": "https://example.gov"}]
            )
            
            # Assert: Verify the result
            assert result == [{"url": "https://example.gov"}]
    
    def test_confidence_calculation(self):
        """Test confidence calculation operation."""
        # Arrange
        tool = StatisticsAnalysisTool()
        
        # Mock the calculate_overall_confidence function
        with patch('react_agent.tools.derivation.calculate_overall_confidence') as mock_calc:
            mock_calc.return_value = 0.78
            
            # Act: Call the tool with confidence operation
            tool_input = {
                "operation": "confidence",
                "category_scores": {"market_dynamics": 0.8, "competition": 0.7},
                "synthesis_quality": 0.9,
                "validation_score": 0.85
            }
            result = tool.run(tool_input)
            
            # Assert: Verify the function was called with correct parameters
            mock_calc.assert_called_once_with(
                {"market_dynamics": 0.8, "competition": 0.7},
                0.9,
                0.85
            )
            
            # Assert: Verify the result
            assert result == 0.78
    
    def test_synthesis_quality(self):
        """Test synthesis quality assessment operation."""
        # Arrange
        tool = StatisticsAnalysisTool()
        
        # Mock the assess_synthesis_quality function
        with patch('react_agent.tools.derivation.assess_synthesis_quality') as mock_assess:
            mock_assess.return_value = 0.92
            
            # Act: Call the tool with synthesis operation
            tool_input = {
                "operation": "synthesis",
                "synthesis": {"text": "Comprehensive market analysis", "facts_count": 15}
            }
            result = tool.run(tool_input)
            
            # Assert: Verify the function was called with correct parameters
            mock_assess.assert_called_once_with(
                {"text": "Comprehensive market analysis", "facts_count": 15}
            )
            
            # Assert: Verify the result
            assert result == 0.92
    
    def test_statistical_validation(self):
        """Test statistical validation operation."""
        # Arrange
        tool = StatisticsAnalysisTool()
        
        # Mock the perform_statistical_validation function
        with patch('react_agent.tools.derivation.perform_statistical_validation') as mock_validate:
            mock_validate.return_value = 0.88
            
            # Act: Call the tool with validate operation
            tool_input = {
                "operation": "validate",
                "facts": [{"text": "Market grew by 25%", "confidence": 0.9}]
            }
            result = tool.run(tool_input)
            
            # Assert: Verify the function was called with correct parameters
            mock_validate.assert_called_once_with(
                [{"text": "Market grew by 25%", "confidence": 0.9}]
            )
            
            # Assert: Verify the result
            assert result == 0.88
    
    def test_fact_consistency(self):
        """Test fact consistency assessment operation."""
        # Arrange
        tool = StatisticsAnalysisTool()
        
        # Mock the assess_fact_consistency function
        with patch('react_agent.tools.derivation.assess_fact_consistency') as mock_assess:
            mock_assess.return_value = 0.95
            
            # Act: Call the tool with consistency operation
            tool_input = {
                "operation": "consistency",
                "facts": [
                    {"text": "Market grew by 25% in 2023"},
                    {"text": "Growth was approximately 25% last year"}
                ]
            }
            result = tool.run(tool_input)
            
            # Assert: Verify the function was called with correct parameters
            mock_assess.assert_called_once_with([
                {"text": "Market grew by 25% in 2023"},
                {"text": "Growth was approximately 25% last year"}
            ])
            
            # Assert: Verify the result
            assert result == 0.95
    
    def test_missing_parameters(self):
        """Test error handling for missing parameters."""
        # Arrange
        tool = StatisticsAnalysisTool()
        
        # Act & Assert: Test missing parameters for quality
        with pytest.raises(ValueError, match="Missing required parameters: category, facts, sources, thresholds"):
            tool.run({"operation": "quality"})
        
        # Act & Assert: Test missing parameters for sources
        with pytest.raises(ValueError, match="Missing required parameters: sources"):
            tool.run({"operation": "sources"})
    
    def test_unknown_operation(self):
        """Test error handling for unknown operation."""
        # Arrange
        tool = StatisticsAnalysisTool()
        
        # Act & Assert: Test invalid operation
        with pytest.raises(Exception) as excinfo:
            tool.run({"operation": "invalid_op"})
        assert "Input should be 'quality', 'sources', 'confidence', 'synthesis', 'validate' or 'consistency'" in str(excinfo.value)
    
    def test_async_run(self):
        """Test asynchronous run method."""
        # Arrange
        tool = StatisticsAnalysisTool()
        
        # Mock the calculate_category_quality_score function
        with patch('react_agent.tools.derivation.calculate_category_quality_score') as mock_calc:
            mock_calc.return_value = 0.85
            
            # Act: Call the async run method
            import asyncio
            
            async def test_async():
                tool_input = {
                    "operation": "quality",
                    "category": "market_dynamics",
                    "facts": [{"text": "Market grew by 25%"}],
                    "sources": [{"url": "https://example.com"}],
                    "thresholds": {"min_facts": 1}
                }
                return await tool._arun(
                    operation=tool_input["operation"],
                    category=tool_input["category"],
                    facts=tool_input["facts"],
                    sources=tool_input["sources"],
                    thresholds=tool_input["thresholds"]
                )

            result = asyncio.run(test_async())
            
            # Assert: Verify the result
            assert result == 0.85
    
    def test_input_schema(self):
        """Test that the tool has the correct input schema."""
        # Arrange
        tool = StatisticsAnalysisTool()
        
        # Assert: Verify the input schema
        assert tool.args_schema == StatisticsAnalysisInput


class TestResearchValidationTool:
    """Tests for the ResearchValidationTool class.
    
    Test scenarios cover:
    - Content validation
    - Facts validation
    - Sources quality assessment
    - Overall score calculation
    - Parameter validation
    - Error handling
    
    Example test cases:
        >>> test_content_validation()
        Verifies content validation operation
        
        >>> test_facts_validation()
        Verifies facts validation operation
    """
    
    @pytest.fixture
    def validation_tool(self):
        """Create a ResearchValidationTool instance for testing."""
        return ResearchValidationTool()
    
    def test_content_validation(self):
        """Test content validation operation."""
        # Arrange
        tool = ResearchValidationTool()
        
        # Mock the validate_content and should_skip_content functions
        with patch('react_agent.tools.derivation.validate_content') as mock_validate:
            with patch('react_agent.tools.derivation.should_skip_content') as mock_skip:
                mock_validate.return_value = True
                mock_skip.return_value = False
                
                # Act: Call the tool with content operation
                tool_input = {
                    "operation": "content",
                    "content": "Valid content",
                    "url": "https://example.com"
                }
                result = tool.run(tool_input)
                
                # Assert: Verify the functions were called with correct parameters
                mock_skip.assert_called_once_with("https://example.com")
                mock_validate.assert_called_once_with("Valid content")
                
                # Assert: Verify the result
                assert result == {"valid": True}
    
    def test_facts_validation(self):
        """Test facts validation operation."""
        # Arrange
        tool = ResearchValidationTool()
        
        # Mock the assess_fact_consistency function
        with patch('react_agent.tools.derivation.assess_fact_consistency') as mock_assess:
            mock_assess.return_value = 0.75
            
            # Act: Call the tool with facts operation
            tool_input = {
                "operation": "facts",
                "facts": [{"text": "Fact 1"}, {"text": "Fact 2"}]
            }
            result = tool.run(tool_input)
            
            # Assert: Verify the function was called with correct parameters
            mock_assess.assert_called_once_with([{"text": "Fact 1"}, {"text": "Fact 2"}])
            
            # Assert: Verify the result
            assert result == {"consistency_score": 0.75}
    
    def test_sources_validation(self):
        """Test sources validation operation."""
        # Arrange
        tool = ResearchValidationTool()
        
        # Act: Call the tool with sources operation
        tool_input = {
            "operation": "sources",
            "sources": [
                {"url": "https://example.com", "quality_score": 0.8},
                {"url": "https://example.org", "quality_score": 0.7}
            ]
        }
        result = tool.run(tool_input)
        
        # Assert: Verify the result
        assert "quality_scores" in result
        assert len(result["quality_scores"]) == 2
        assert result["quality_scores"][0]["url"] == "https://example.com"
        assert result["quality_scores"][0]["quality"] == 0.8
    
    def test_overall_validation(self):
        """Test overall score calculation operation."""
        # Arrange
        tool = ResearchValidationTool()
        
        # Act: Call the tool with overall operation
        tool_input = {
            "operation": "overall",
            "scores": {"content": 0.9, "facts": 0.8, "sources": 0.7}
        }
        result = tool.run(tool_input)
        
        # Assert: Verify the result
        assert "overall_score" in result
        assert round(result["overall_score"], 1) == 0.8  # (0.9 + 0.8 + 0.7) / 3
    
    def test_content_validation_skip(self):
        """Test content validation when content should be skipped."""
        # Arrange
        tool = ResearchValidationTool()
        
        # Mock the should_skip_content function to return True
        with patch('react_agent.tools.derivation.should_skip_content') as mock_skip:
            mock_skip.return_value = True
            
            # Act: Call the tool with content operation
            tool_input = {
                "operation": "content",
                "content": "Content to skip",
                "url": "https://example.com/skip-this"
            }
            result = tool.run(tool_input)
            
            # Assert: Verify the result
            assert result == {"valid": False, "reason": "Content type should be skipped"}
    
    def test_missing_parameters(self):
        """Test error handling for missing parameters."""
        # Arrange
        tool = ResearchValidationTool()
        
        # Act & Assert: Test missing parameters for content
        with pytest.raises(ValueError, match="Missing required parameters: content, url"):
            tool.run({"operation": "content"})
        
        # Act & Assert: Test missing parameters for facts
        with pytest.raises(ValueError, match="Missing required parameters: facts"):
            tool.run({"operation": "facts"})
        
        # Act & Assert: Test missing parameters for sources
        with pytest.raises(ValueError, match="Missing required parameters: sources"):
            tool.run({"operation": "sources"})
        
        # Act & Assert: Test missing parameters for overall
        with pytest.raises(ValueError, match="Missing required parameters: scores"):
            tool.run({"operation": "overall"})
    
    def test_unknown_operation(self):
        """Test error handling for unknown operation."""
        # Arrange
        tool = ResearchValidationTool()
        
        # Act & Assert: Test invalid operation
        with pytest.raises(Exception) as excinfo:
            tool.run({"operation": "invalid_op"})
        assert "Input should be 'content', 'facts', 'sources' or 'overall'" in str(excinfo.value)
    
    def test_input_schema(self):
        """Test that the tool has the correct input schema."""
        # Arrange
        tool = ResearchValidationTool()
        
        # Assert: Verify the input schema
        assert tool.args_schema == ResearchValidationInput


class TestToolNodeCreation:
    """Tests for the create_derivation_toolnode function.
    
    Test scenarios cover:
    - Creating a ToolNode with all tools
    - Creating a ToolNode with specific tools
    - Edge cases
    
    Example test cases:
        >>> test_create_derivation_toolnode()
        Verifies creation of ToolNode with all tools
        
        >>> test_create_derivation_toolnode_with_specific_tools()
        Verifies creation of ToolNode with specific tools
    """
    
    def test_create_derivation_toolnode(self):
        """Test creation of ToolNode with all tools."""
        # Arrange & Act
        # Create a ToolNode with all tools
        node = create_derivation_toolnode()
        
        # Assert: Verify the node has all three tools
        tool_names = list(node.tools_by_name.keys())
        assert "extraction_tool" in tool_names
        assert "statistics_analyzer" in tool_names
        assert "research_validator" in tool_names
    
    def test_create_derivation_toolnode_with_specific_tools(self):
        """Test creation of ToolNode with specific tools."""
        # Arrange & Act
        # Create a ToolNode with specific tools
        node = create_derivation_toolnode(include_tools=["extraction", "statistics"])
        
        # Assert: Verify the node has only the specified tools
        tool_names = list(node.tools_by_name.keys())
        assert "extraction_tool" in tool_names
        assert "statistics_analyzer" in tool_names
        assert "research_validator" not in tool_names
    
    def test_create_derivation_toolnode_with_empty_list(self):
        """Test creation of ToolNode with an empty list of tools.
        
        Note: The current implementation returns all tools when an empty list is provided,
        as it treats an empty list the same as None (include all tools).
        """
        # Arrange & Act
        node = create_derivation_toolnode(include_tools=[])
        assert len(node.tools_by_name) == 3  # All tools are included
    
    def test_create_derivation_toolnode_with_invalid_tool(self):
        """Test creation of ToolNode with invalid tool names."""
        # Arrange & Act
        # Create a ToolNode with valid and invalid tool names
        node = create_derivation_toolnode(include_tools=["extraction", "invalid_tool"])
        
        # Assert: Verify only valid tools are included
        tool_names = list(node.tools_by_name.keys())
        assert "extraction_tool" in tool_names
        assert len(tool_names) == 1  # Only one valid tool should be included


# Add a main block to run the tests directly
if __name__ == "__main__":
    import sys
    import pytest
    print("Running tests...")
    sys.exit(pytest.main(["-v", __file__]))