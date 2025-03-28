"""Tests for enhanced search functionality.

This module contains tests for the enhanced search functionality,
including standardization, optimization, and progressive processing.
"""

import asyncio
import json
import os
import sys
import time
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.react_agent.types import EnhancedSearchResult, SearchResult
from src.react_agent.utils.search import (
    CategorySuccessTracker,
    enhance_search_results,
    get_optimized_query,
    standardize_search_result,
)
from src.react_agent.utils.enhanced_search import (
    execute_progressive_search,
    _search_and_process,
    _process_results_queue,
    _process_search_result,
)


# Test data
SAMPLE_STRING_RESULT = "This is a sample search result."

SAMPLE_DICT_RESULT = {
    "results": [
        {
            "url": "https://example.com/1",
            "title": "Example 1",
            "content": "This is example content 1.",
            "source": "web",
            "quality_score": 0.8,
            "published_date": "2023-01-01"
        },
        {
            "url": "https://example.com/2",
            "title": "Example 2",
            "content": "This is example content 2.",
            "source": "web",
            "quality_score": 0.7,
            "published_date": "2023-02-01"
        }
    ]
}

SAMPLE_LIST_RESULT = [
    {
        "url": "https://example.com/3",
        "title": "Example 3",
        "content": "This is example content 3.",
        "source": "web",
        "quality_score": 0.9,
        "published_date": "2023-03-01"
    },
    {
        "url": "https://example.com/4",
        "title": "Example 4",
        "content": "This is example content 4.",
        "source": "web",
        "quality_score": 0.6,
        "published_date": "2023-04-01"
    },
    "This is a string item in the list."
]


def test_standardize_string_result():
    """Test standardizing a string search result."""
    result = standardize_search_result(SAMPLE_STRING_RESULT, category="test_category", query="test query")
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["snippet"] == SAMPLE_STRING_RESULT
    assert result[0]["content_type"] == "text"


def test_standardize_dict_result():
    """Test standardizing a dictionary search result."""
    result = standardize_search_result(SAMPLE_DICT_RESULT, category="test_category", query="test query")
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["url"] == "https://example.com/1"
    assert result[0]["title"] == "Example 1"
    assert result[0]["snippet"] == "This is example content 1."
    assert result[0]["source"] == "web"
    assert result[0]["quality_score"] == 0.8
    assert result[0]["published_date"] == "2023-01-01"


def test_standardize_list_result():
    """Test standardizing a list search result."""
    result = standardize_search_result(SAMPLE_LIST_RESULT, category="test_category", query="test query")
    
    assert isinstance(result, list)
    assert len(result) == 3  # 2 dict items + 1 string item
    assert result[0]["url"] == "https://example.com/3"
    assert result[0]["title"] == "Example 3"
    assert result[0]["snippet"] == "This is example content 3."
    assert result[0]["source"] == "web"
    assert result[0]["quality_score"] == 0.9
    assert result[0]["published_date"] == "2023-03-01"
    
    # Check the string item
    assert result[2]["snippet"] == "This is a string item in the list."
    assert result[2]["content_type"] == "text"


def test_enhance_search_results():
    """Test enhancing search results with additional metadata."""
    standardized_results = standardize_search_result(SAMPLE_LIST_RESULT, category="test_category", query="test query")
    enhanced_results = enhance_search_results(
        standardized_results,
        category="test_category",
        query="test query content",
        processing_time_ms=100.0
    )
    
    assert isinstance(enhanced_results, list)
    assert len(enhanced_results) == 3
    
    # Check enhanced fields
    for result in enhanced_results:
        assert "retrieval_timestamp" in result
        assert result["category"] == "test_category"
        assert result["query"] == "test query content"
        assert result["success"] is True
        assert result["processing_time_ms"] == 100.0
        assert "source_quality" in result
        assert "relevance_score" in result


def test_get_optimized_query():
    """Test optimized query generation."""
    # Test primary query
    query = get_optimized_query("best_practices", "cloud computing", fallback_level=0)
    assert "cloud computing" in query
    assert "best practices" in query.lower()
    
    # Test fallback query
    query = get_optimized_query("best_practices", "cloud computing", fallback_level=1)
    assert "cloud computing" in query
    assert "guidelines" in query.lower() or "methodologies" in query.lower()
    
    # Test fallback level 2
    query = get_optimized_query("best_practices", "cloud computing", fallback_level=2)
    assert "cloud computing" in query
    assert "implement" in query.lower() or "methodology" in query.lower()
    
    # Test unknown category
    query = get_optimized_query("unknown_category", "cloud computing", fallback_level=0)
    assert query == "cloud computing"


def test_category_success_tracker():
    """Test the CategorySuccessTracker class."""
    tracker = CategorySuccessTracker()
    
    # Track some attempts
    tracker.track_attempt("category1", True)
    tracker.track_attempt("category1", True)
    tracker.track_attempt("category1", False)
    tracker.track_attempt("category2", False)
    tracker.track_attempt("category2", False)
    
    # Check success rates
    assert tracker.success_rates["category1"] == 2/3
    assert tracker.success_rates["category2"] == 0
    
    # Check should_skip_category
    assert not tracker.should_skip_category("category1", threshold=0.5)
    assert tracker.should_skip_category("category2", threshold=0.5)
    
    # Check get_category_priority
    assert tracker.get_category_priority("category1") > tracker.get_category_priority("category2")


@pytest.mark.asyncio
async def test_process_search_result():
    """Test processing a single search result."""
    # Mock dependencies
    with patch("src.react_agent.utils.enhanced_search.load_checkpoint", return_value=None), \
         patch("src.react_agent.utils.enhanced_search.create_checkpoint"), \
         patch("src.react_agent.utils.enhanced_search.extract_category_information", 
               return_value=(["fact1", "fact2"], 0.8)), \
         patch("src.react_agent.utils.enhanced_search.StatisticsAnalysisTool") as mock_stats_tool:
        
        # Configure mock statistics tool
        mock_stats_instance = MagicMock()
        mock_stats_instance.ainvoke = AsyncMock(return_value=["stat1", "stat2"])
        mock_stats_tool.return_value = mock_stats_instance
        
        # Test processing a search result
        result = {
            "url": "https://example.com/test",
            "title": "Test Result",
            "snippet": "This is a test result with good content.",
            "source": "web",
            "quality_score": 0.8,
            "published_date": "2023-01-01",
            "content_type": "text"
        }
        
        facts, sources, statistics = await _process_search_result(
            result=result,
            category="test_category",
            original_query="test query",
            config={}
        )
        
        # Check results
        assert len(facts) == 2
        assert len(sources) == 1
        assert len(statistics) == 2
        assert sources[0]["url"] == "https://example.com/test"
        assert sources[0]["title"] == "Test Result"


@pytest.mark.asyncio
async def test_search_and_process():
    """Test searching and processing results."""
    # This is a more complex test that would require extensive mocking
    # of the search functionality and queue processing
    pass


@pytest.mark.asyncio
async def test_execute_progressive_search():
    """Test the progressive search execution."""
    # This is a more complex test that would require extensive mocking
    # of the search functionality and queue processing
    pass


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])