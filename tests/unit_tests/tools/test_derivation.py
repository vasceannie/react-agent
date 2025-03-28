from unittest.mock import MagicMock, patch

import pytest

from react_agent.tools.derivation import (
    ExtractionTool,
    ResearchValidationTool,
    StatisticsAnalysisTool,
    create_derivation_toolnode,
)


class TestExtractionTool:
    def test_statistics_extraction(self):
        """Test statistics extraction operation."""
        tool = ExtractionTool()
        
        # Mock the extract_statistics function
        with patch('react_agent.tools.derivation.extract_statistics') as mock_extract:
            mock_extract.return_value = [{"text": "75% of enterprises adopted cloud computing"}]
            
            # Call the tool with statistics operation
            result = tool.run(
                operation="statistics",
                text="According to a recent survey, 75% of enterprises adopted cloud computing in 2023.",
                url="https://example.com",
                source_title="Cloud Report"
            )
            
            # Verify the function was called with correct parameters
            mock_extract.assert_called_once_with(
                "According to a recent survey, 75% of enterprises adopted cloud computing in 2023.",
                "https://example.com",
                "Cloud Report"
            )
            
            # Verify the result
            assert result == [{"text": "75% of enterprises adopted cloud computing"}]
    
    def test_citations_extraction(self):
        """Test citations extraction operation."""
        tool = ExtractionTool()
        
        # Mock the extract_citations function
        with patch('react_agent.tools.derivation.extract_citations') as mock_extract:
            mock_extract.return_value = [{"source": "TechCorp", "context": "survey by TechCorp"}]
            
            # Call the tool with citations operation
            result = tool.run(
                operation="citations",
                text="According to a recent survey by TechCorp, 75% of enterprises adopted cloud computing."
            )
            
            # Verify the function was called with correct parameters
            mock_extract.assert_called_once_with(
                "According to a recent survey by TechCorp, 75% of enterprises adopted cloud computing."
            )
            
            # Verify the result
            assert result == [{"source": "TechCorp", "context": "survey by TechCorp"}]
    
    def test_enrich_fact(self):
        """Test fact enrichment operation."""
        tool = ExtractionTool()
        
        # Mock the enrich_extracted_fact function
        with patch('react_agent.tools.derivation.enrich_extracted_fact') as mock_enrich:
            mock_enrich.return_value = {
                "text": "75% of enterprises adopted cloud computing",
                "source_url": "https://example.com",
                "source_title": "Cloud Report",
                "confidence_score": 0.9
            }
            
            # Call the tool with enrich operation
            result = tool.run(
                operation="enrich",
                fact={"text": "75% of enterprises adopted cloud computing"},
                url="https://example.com",
                source_title="Cloud Report"
            )
            
            # Verify the function was called with correct parameters
            mock_enrich.assert_called_once_with(
                {"text": "75% of enterprises adopted cloud computing"},
                "https://example.com",
                "Cloud Report"
            )
            
            # Verify the result
            assert result["text"] == "75% of enterprises adopted cloud computing"
            assert result["source_url"] == "https://example.com"
            assert result["source_title"] == "Cloud Report"
            assert result["confidence_score"] == 0.9
    
    def test_missing_parameters(self):
        """Test error handling for missing parameters."""
        tool = ExtractionTool()
        
        # Test missing text for statistics
        with pytest.raises(ValueError, match="Text is required for statistics extraction"):
            tool.run(operation="statistics")
        
        # Test missing text for citations
        with pytest.raises(ValueError, match="Text is required for citation extraction"):
            tool.run(operation="citations")
        
        # Test missing fact for enrichment
        with pytest.raises(ValueError, match="Fact is required for enrichment"):
            tool.run(operation="enrich", url="https://example.com")


class TestStatisticsAnalysisTool:
    def test_quality_score_calculation(self):
        """Test quality score calculation operation."""
        tool = StatisticsAnalysisTool()
        
        # Mock the calculate_category_quality_score function
        with patch('react_agent.tools.derivation.calculate_category_quality_score') as mock_calc:
            mock_calc.return_value = 0.85
            
            # Call the tool with quality operation
            result = tool.run(
                operation="quality",
                category="market_dynamics",
                facts=[{"text": "Market grew by 25%"}],
                sources=[{"url": "https://example.com"}],
                thresholds={"min_facts": 1}
            )
            
            # Verify the function was called with correct parameters
            mock_calc.assert_called_once_with(
                "market_dynamics",
                [{"text": "Market grew by 25%"}],
                [{"url": "https://example.com"}],
                {"min_facts": 1}
            )
            
            # Verify the result
            assert result == 0.85
    
    def test_authoritative_sources(self):
        """Test authoritative sources assessment operation."""
        tool = StatisticsAnalysisTool()
        
        # Mock the assess_authoritative_sources function
        with patch('react_agent.tools.derivation.assess_authoritative_sources') as mock_assess:
            mock_assess.return_value = [{"url": "https://example.gov"}]
            
            # Call the tool with sources operation
            result = tool.run(
                operation="sources",
                sources=[{"url": "https://example.com"}, {"url": "https://example.gov"}]
            )
            
            # Verify the function was called with correct parameters
            mock_assess.assert_called_once_with(
                [{"url": "https://example.com"}, {"url": "https://example.gov"}]
            )
            
            # Verify the result
            assert result == [{"url": "https://example.gov"}]
    
    def test_missing_parameters(self):
        """Test error handling for missing parameters."""
        tool = StatisticsAnalysisTool()
        
        # Test missing parameters for quality
        with pytest.raises(ValueError, match="Missing required parameters: category, facts, sources, thresholds"):
            tool.run(operation="quality")
        
        # Test missing parameters for sources
        with pytest.raises(ValueError, match="Missing required parameters: sources"):
            tool.run(operation="sources")


class TestResearchValidationTool:
    def test_content_validation(self):
        """Test content validation operation."""
        tool = ResearchValidationTool()
        
        # Mock the validate_content and should_skip_content functions
        with patch('react_agent.tools.derivation.validate_content') as mock_validate:
            with patch('react_agent.tools.derivation.should_skip_content') as mock_skip:
                mock_validate.return_value = True
                mock_skip.return_value = False
                
                # Call the tool with content operation
                result = tool.run(
                    operation="content",
                    content="Valid content",
                    url="https://example.com"
                )
                
                # Verify the functions were called with correct parameters
                mock_skip.assert_called_once_with("https://example.com")
                mock_validate.assert_called_once_with("Valid content")
                
                # Verify the result
                assert result == {"valid": True}
    
    def test_facts_validation(self):
        """Test facts validation operation."""
        tool = ResearchValidationTool()
        
        # Mock the assess_fact_consistency function
        with patch('react_agent.tools.derivation.assess_fact_consistency') as mock_assess:
            mock_assess.return_value = 0.75
            
            # Call the tool with facts operation
            result = tool.run(
                operation="facts",
                facts=[{"text": "Fact 1"}, {"text": "Fact 2"}]
            )
            
            # Verify the function was called with correct parameters
            mock_assess.assert_called_once_with([{"text": "Fact 1"}, {"text": "Fact 2"}])
            
            # Verify the result
            assert result == {"consistency_score": 0.75}


class TestToolNodeCreation:
    def test_create_derivation_toolnode(self):
        """Test creation of ToolNode with derivation tools."""
        # Create a ToolNode with all tools
        node = create_derivation_toolnode()
        
        # Verify the node has all three tools
        tool_names = [tool.name for tool in node.tools]
        assert "extraction_tool" in tool_names
        assert "statistics_analyzer" in tool_names
        assert "research_validator" in tool_names
        
        # Create a ToolNode with specific tools
        node = create_derivation_toolnode(include_tools=["extraction", "statistics"])
        
        # Verify the node has only the specified tools
        tool_names = [tool.name for tool in node.tools]
        assert "extraction_tool" in tool_names
        assert "statistics_analyzer" in tool_names
        assert "research_validator" not in tool_names