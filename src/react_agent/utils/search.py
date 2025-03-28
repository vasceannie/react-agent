"""Search result standardization and optimization utilities.

This module provides functions for standardizing search results from various formats,
optimizing search queries for better results, and tracking search performance metrics.
"""

import json
import time
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Union, cast

from react_agent.types import EnhancedSearchResult, SearchResult
from react_agent.utils.logging import get_logger, info_highlight, warning_highlight
from react_agent.utils.content import (
    detect_content_type,
    should_skip_content,
    validate_content,
)

# Initialize logger
logger = get_logger(__name__)


def standardize_search_result(
    result: Any, category: str = "", query: str = "", start_time: Optional[float] = None
) -> List[SearchResult]:
    """Standardize search results from various formats into a consistent structure.

    This function handles different result formats that might be returned by search APIs:
    - String results (plain text)
    - Dictionary with 'results' key containing a list of items
    - List of dictionaries or strings
    - Dictionary with other keys containing relevant content

    Args:
        result: The raw search result in any format
        category: The research category this search was for
        query: The query used for this search
        start_time: Optional start time for performance tracking

    Returns:
        A list of standardized SearchResult objects
    """
    formatted_results: List[SearchResult] = []
    processing_time = 0

    if start_time:
        processing_time = (time.time() - start_time) * 1000  # Convert to ms

    # Log the raw result type for debugging
    logger.debug(f"Raw result type: {type(result)}")
    if isinstance(result, dict):
        logger.debug(f"Raw result keys: {list(result.keys())}")

    # Handle string results
    if isinstance(result, str):
        logger.info(f"Standardizing string result for category: {category}")
        if validate_content(result):
            formatted_results.append(
                SearchResult(
                    content=result,
                    category=category,
                    query=query,
                    processing_time_ms=processing_time,
                )
            )

    # Handle dictionary with results key
    elif isinstance(result, dict):
        logger.info(
            f"Standardizing dictionary result for category: {category}"
        )
        
        # Check for 'results' key first (standard format)
        if "results" in result and isinstance(result["results"], list):
            for item in result["results"]:
                if not isinstance(item, dict):
                    continue

                url = item.get("url", "")
                title = item.get("title", "")
                content = item.get("content", item.get("snippet", ""))

                # Skip problematic content
                if should_skip_content(url) or not validate_content(content):
                    continue

                content_type = detect_content_type(url, content)

                formatted_result: SearchResult = {
                    "url": url,
                    "title": title,
                    "snippet": content,
                    "source": item.get("source", "search"),
                    "quality_score": item.get("quality_score", 0.5),
                    "published_date": item.get("published_date"),
                    "content_type": content_type,
                }
                formatted_results.append(formatted_result)
        
        # Handle Jina API response format (has 'data' key with search results)
        elif "data" in result and isinstance(result["data"], list):
            for item in result["data"]:
                if not isinstance(item, dict):
                    continue

                url = item.get("url", "")
                title = item.get("title", "")
                content = item.get("content", item.get("snippet", item.get("text", "")))

                # Skip problematic content
                if should_skip_content(url) or not validate_content(content):
                    continue

                content_type = detect_content_type(url, content)

                formatted_result: SearchResult = {
                    "url": url,
                    "title": title,
                    "snippet": content,
                    "source": item.get("source", "search"),
                    "quality_score": item.get("quality_score", 0.5),
                    "published_date": item.get("published_date"),
                    "content_type": content_type,
                }
                formatted_results.append(formatted_result)
                
        # Handle case where the dictionary itself contains content
        elif any(key in result for key in ["content", "snippet", "text"]):
            url = result.get("url", "")
            title = result.get("title", "")
            content = result.get("content", result.get("snippet", result.get("text", "")))
            
            # Skip problematic content
            if not should_skip_content(url) and validate_content(content):
                content_type = detect_content_type(url, content)
                
                formatted_result: SearchResult = {
                    "url": url,
                    "title": title or f"Search result for {category}",
                    "snippet": content,
                    "source": result.get("source", "search"),
                    "quality_score": result.get("quality_score", 0.5),
                    "published_date": result.get("published_date"),
                    "content_type": content_type,
                }
                formatted_results.append(formatted_result)

    # Handle list format
    elif isinstance(result, list):
        logger.info(f"Standardizing list result for category: {category}")
        for item in result:
            if isinstance(item, dict):
                url = item.get("url", "")
                title = item.get("title", "")
                content = item.get("content", item.get("text", item.get("snippet", "")))

                # Skip problematic content
                if should_skip_content(url) or not validate_content(content):
                    continue

                content_type = detect_content_type(url, content)

                formatted_result: SearchResult = {
                    "url": url,
                    "title": title,
                    "snippet": content,
                    "source": item.get("source", "search"),
                    "quality_score": item.get("quality_score", 0.5),
                    "published_date": item.get("published_date"),
                    "content_type": content_type,
                }
                formatted_results.append(formatted_result)
            elif isinstance(item, str) and validate_content(item):
                formatted_result: SearchResult = {
                    "url": "",
                    "title": f"Search result {result.index(item) + 1}",
                    "snippet": item,
                    "source": "search",
                    "quality_score": 0.5,
                    "published_date": None,
                    "content_type": "text",
                }
                formatted_results.append(formatted_result)

    # Log the result
    logger.info(
        f"Standardized {len(formatted_results)} results for category: {category}"
    )

    return formatted_results


def enhance_search_results(
    results: List[SearchResult],
    category: str,
    query: str,
    processing_time_ms: float = 0.0,
) -> List[EnhancedSearchResult]:
    """Enhance standardized search results with additional metadata.

    Args:
        results: List of standardized search results
        category: The research category
        query: The search query used
        processing_time_ms: Processing time in milliseconds

    Returns:
        List of enhanced search results with additional metadata
    """
    enhanced_results: List[EnhancedSearchResult] = []
    timestamp = datetime.now(UTC).isoformat()

    for result in results:
        # Calculate source quality based on available metadata
        source_quality = result.get("quality_score", 0.5)

        # Simple relevance calculation based on keyword matching
        # This could be enhanced with more sophisticated methods
        relevance_score = 0.5
        if query and result.get("snippet"):
            query_terms = set(query.lower().split())
            snippet_terms = set(result.get("snippet", "").lower().split())
            if query_terms and snippet_terms:
                overlap = len(query_terms.intersection(snippet_terms))
                relevance_score = min(1.0, max(0.1, overlap / len(query_terms)))

        enhanced_result: EnhancedSearchResult = {
            **result,
            "retrieval_timestamp": timestamp,
            "category": category,
            "query": query,
            "success": True,
            "retry_count": 0,  # Will be updated by the calling function
            "processing_time_ms": processing_time_ms,
            "source_quality": source_quality,
            "relevance_score": relevance_score,
        }
        enhanced_results.append(enhanced_result)

    return enhanced_results


class CategorySuccessTracker:
    """Track search success rates by category.

    This class maintains statistics on search success rates for different categories,
    allowing the system to make informed decisions about which categories to prioritize
    or skip based on historical performance.
    """

    def __init__(self, cache_ttl: int = 86400):  # 24 hour cache by default
        """Initialize the category success tracker.

        Args:
            cache_ttl: Time-to-live for cached data in seconds
        """
        self.success_rates: Dict[str, float] = {}
        self.attempts: Dict[str, int] = {}
        self.successes: Dict[str, int] = {}
        self.cache_key = "category_success_tracker"
        self.cache_ttl = cache_ttl
        self._load_from_cache()

    def track_attempt(self, category: str, success: bool) -> None:
        """Record a search attempt and update metrics.

        Args:
            category: The research category
            success: Whether the search was successful
        """
        if category not in self.attempts:
            self.attempts[category] = 0
            self.successes[category] = 0

        self.attempts[category] += 1
        if success:
            self.successes[category] += 1

        # Update success rate
        if self.attempts[category] > 0:
            self.success_rates[category] = (
                self.successes[category] / self.attempts[category]
            )

        # Save updated metrics to cache
        self._save_to_cache()

    def should_skip_category(self, category: str, threshold: float = 0.2) -> bool:
        """Determine if a category should be skipped based on success rate.

        Args:
            category: The research category
            threshold: Minimum success rate threshold

        Returns:
            True if the category should be skipped, False otherwise
        """
        # Skip if success rate is below threshold and we have enough data
        if category not in self.success_rates:
            return False

        return self.success_rates[category] < threshold and self.attempts[category] >= 5

    def get_category_priority(self, category: str) -> int:
        """Get priority level for category (higher is more important).

        Args:
            category: The research category

        Returns:
            Priority score (higher means higher priority)
        """
        if category not in self.success_rates:
            return 5  # Default medium priority for unknown categories

        # Higher success rate = higher priority
        success_rate = self.success_rates[category]

        if success_rate > 0.8:
            return 10  # High priority
        elif success_rate > 0.5:
            return 7  # Medium-high priority
        elif success_rate > 0.3:
            return 5  # Medium priority
        elif success_rate > 0.1:
            return 3  # Low-medium priority
        else:
            return 1  # Low priority

    def _load_from_cache(self) -> None:
        """Load tracker data from cache."""
        try:
            from react_agent.utils.cache import load_checkpoint

            cached_data = load_checkpoint(self.cache_key)
            if cached_data:
                self.success_rates = cached_data.get("success_rates", {})
                self.attempts = cached_data.get("attempts", {})
                self.successes = cached_data.get("successes", {})
                logger.info(
                    f"Loaded category success tracker from cache with {len(self.success_rates)} categories"
                )
        except Exception as e:
            logger.error(f"Error loading category success tracker from cache: {str(e)}")

    def _save_to_cache(self) -> None:
        """Save tracker data to cache."""
        try:
            from react_agent.utils.cache import create_checkpoint

            cache_data = {
                "success_rates": self.success_rates,
                "attempts": self.attempts,
                "successes": self.successes,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            create_checkpoint(self.cache_key, cache_data, ttl=self.cache_ttl)
            logger.info(
                f"Saved category success tracker to cache with {len(self.success_rates)} categories"
            )
        except Exception as e:
            logger.error(f"Error saving category success tracker to cache: {str(e)}")


# Enhanced query templates for problematic categories
ENHANCED_CATEGORY_TEMPLATES = {
    "best_practices": {
        "primary": "{primary_terms} best practices guidelines recommendations",
        "fallback_1": "{primary_terms} guidelines methodologies frameworks approaches",
        "fallback_2": "{primary_terms} how to implement methodology",
    },
    "regulatory_landscape": {
        "primary": "{primary_terms} regulations compliance legal requirements",
        "fallback_1": "{primary_terms} laws regulations rules standards",
        "fallback_2": "{primary_terms} regulatory authority compliance laws",
    },
    "implementation_factors": {
        "primary": "{primary_terms} implementation considerations factors requirements",
        "fallback_1": "{primary_terms} deployment installation setup process",
        "fallback_2": "{primary_terms} successful implementation case study",
    },
    "provider_landscape": {
        "primary": "{primary_terms} vendors suppliers providers companies",
        "fallback_1": "{primary_terms} market leaders competitors",
        "fallback_2": "{primary_terms} industry players manufacturers",
    },
    "technical_requirements": {
        "primary": "{primary_terms} technical specifications requirements",
        "fallback_1": "{primary_terms} technical documentation standards",
        "fallback_2": "{primary_terms} system requirements architecture",
    },
    "market_dynamics": {
        "primary": "{primary_terms} market trends analysis forecast",
        "fallback_1": "{primary_terms} industry trends market size",
        "fallback_2": "{primary_terms} market growth statistics",
    },
}


def get_optimized_query(
    category: str, primary_terms: str, fallback_level: int = 0
) -> str:
    """Get an optimized query for a specific category with fallback support.

    Args:
        category: The research category
        primary_terms: The primary search terms
        fallback_level: The fallback level (0 = primary, 1 = first fallback, etc.)

    Returns:
        Optimized query string
    """
    if category not in ENHANCED_CATEGORY_TEMPLATES:
        return primary_terms

    templates = ENHANCED_CATEGORY_TEMPLATES[category]

    if fallback_level == 0 or "primary" not in templates:
        template_key = "primary"
    else:
        template_key = f"fallback_{fallback_level}"
        if template_key not in templates:
            # If requested fallback level doesn't exist, use the highest available
            fallback_keys = sorted([
                k for k in templates.keys() if k.startswith("fallback_")
            ])
            if fallback_keys:
                template_key = fallback_keys[-1]
            else:
                template_key = "primary"

    template = templates[template_key]
    return template.format(primary_terms=primary_terms)


def log_search_event(
    category: str, query: str, result_count: int, duration_ms: float, success: bool
) -> None:
    """Log structured information about a search operation.

    Args:
        category: The research category
        query: The search query used
        result_count: Number of results returned
        duration_ms: Duration in milliseconds
        success: Whether the search was successful
    """
    event = {
        "event_type": "search_operation",
        "timestamp": datetime.now(UTC).isoformat(),
        "category": category,
        "query": query,
        "result_count": result_count,
        "duration_ms": duration_ms,
        "success": success,
    }
    logger.info(f"Search event: {json.dumps(event)}")
