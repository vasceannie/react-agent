"""Enhanced search functionality for research.

This module provides enhanced search functionality for the research process,
including progressive processing, category prioritization, and search result
standardization.
"""

import asyncio
import json
import time
import threading
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from react_agent.types import EnhancedSearchResult, SearchResult
from react_agent.utils.cache import create_checkpoint, load_checkpoint
from react_agent.utils.content import (
    detect_content_type,
    should_skip_content,
    validate_content,
)
from react_agent.utils.extraction import extract_category_information
from react_agent.utils.logging import (
    error_highlight,
    get_logger,
    info_highlight,
    warning_highlight,
)
from react_agent.utils.search import (
    CategorySuccessTracker,
    enhance_search_results,
    get_optimized_query,
    log_search_event,
    standardize_search_result,
)

# Initialize logger
logger = get_logger(__name__)

# Track active search requests
active_requests = 0
active_requests_lock = threading.Lock()

# Semaphore to limit concurrent API requests to prevent rate limiting errors
# Default is 3 concurrent requests, but can be configured via max_concurrent_searches
API_REQUEST_SEMAPHORE = asyncio.Semaphore(3)


async def execute_progressive_search(
    state: Dict[str, Any], config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute search with progressive processing for better efficiency.

    This function implements a more efficient search process by:
    1. Prioritizing categories based on historical success rates
    2. Processing search results as they arrive rather than waiting for all searches
    3. Using a queue-based approach for better resource utilization

    Args:
        state: The current research state
        config: Optional configuration

    Returns:
        Updated state dictionary
    """
    info_highlight("Executing progressive search")

    categories = state["categories"]

    # Get or initialize category success tracker
    success_tracker = getattr(
        state, "category_success_tracker", CategorySuccessTracker()
    )

    # Create a queue for search results
    results_queue = asyncio.Queue()

    # Track completed categories
    completed_categories = set()

    # Check cache for each category
    for category, category_state in categories.items():
        if category_state["complete"]:
            info_highlight(f"Category {category} already complete, skipping")
            completed_categories.add(category)
            continue

        # Check checkpoint
        cache_key = f"category_{state['original_query']}_{category}"
        if cached_state := load_checkpoint(cache_key):
            info_highlight(f"Using cached results for category: {category}")
            category_state["search_results"] = cached_state.get("search_results", [])
            category_state["extracted_facts"] = cached_state.get("extracted_facts", [])
            category_state["sources"] = cached_state.get("sources", [])
            category_state["complete"] = True
            completed_categories.add(category)
            continue

    # Prioritize remaining categories
    prioritized_categories = []
    for category, category_state in categories.items():
        if category not in completed_categories:
            # Skip categories with very low success rates
            if success_tracker.should_skip_category(category, threshold=0.15):
                warning_highlight(
                    f"Skipping category {category} due to low success rate"
                )
                category_state["status"] = "skipped_low_success"
                category_state["complete"] = True
                continue

            priority = success_tracker.get_category_priority(category)
            prioritized_categories.append((category, priority))

    # Sort by priority (higher priority first)
    prioritized_categories.sort(key=lambda x: x[1], reverse=True)

    if prioritized_categories:
        priority_info = ", ".join([
            f"{cat}({pri})" for cat, pri in prioritized_categories
        ])
        info_highlight(f"Prioritized categories: {priority_info}")

    # Get max concurrent requests from config or use default
    max_concurrent = 3
    if config and "max_concurrent_searches" in config:
        max_concurrent = config["max_concurrent_searches"]

    # Update semaphore with configured value
    global API_REQUEST_SEMAPHORE
    API_REQUEST_SEMAPHORE = asyncio.Semaphore(max_concurrent)
    info_highlight(
        f"Starting search for {len(prioritized_categories)} categories with max {max_concurrent} concurrent requests"
    )

    # Start search tasks for each category in priority order
    search_tasks = []
    for category, _ in prioritized_categories:
        task = asyncio.create_task(
            _search_and_process(state, category, results_queue, config)
        )
        search_tasks.append(task)

    # Start a task to process results from the queue
    processor_task = asyncio.create_task(
        _process_results_queue(state, results_queue, config)
    )

    # Wait for all search tasks to complete
    if search_tasks:
        await asyncio.gather(*search_tasks)

    # Signal the processor task to finish
    await results_queue.put(None)
    await processor_task

    # Check if all categories are complete
    all_complete = all(
        category_state["complete"] for category_state in categories.values()
    )

    if all_complete:
        info_highlight("All categories research complete")
        return {"status": "researched", "categories": categories}
    else:
        # Some categories still incomplete
        incomplete = [
            category
            for category, category_state in categories.items()
            if not category_state["complete"]
        ]
        info_highlight(f"Categories still incomplete: {', '.join(incomplete)}")
        return {"status": "research_incomplete", "categories": categories}


async def _search_and_process(
    state: Dict[str, Any],
    category: str,
    results_queue: asyncio.Queue,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Search for a category and process results as they arrive.

    This helper function executes the search for a specific category and
    immediately processes the results, allowing for better parallelization
    and resource utilization.

    Args:
        state: The current research state
        category: The category to search for
        results_queue: Queue for processing results
        config: Optional configuration
    """
    try:
        # Track active requests
        global active_requests
        info_highlight(f"Starting search for category {category}")

        # Get category state
        category_state = state["categories"][category]

        # Update status
        category_state["status"] = "searching"
        category_state["retry_count"] += 1

        # Get category-specific search parameters
        from react_agent.prompts.research import SEARCH_QUALITY_THRESHOLDS

        thresholds = SEARCH_QUALITY_THRESHOLDS.get(category, {})

        # Extract primary terms from query - use fewer terms for more focused queries
        query = category_state["query"]

        # Check for domain-specific terms that should be included
        domain_terms = []
        if (
            "education" in query.lower()
            or "university" in query.lower()
            or "college" in query.lower()
        ):
            domain_terms.append("higher education")

        if (
            "southwest" in query.lower()
            or "arizona" in query.lower()
            or "new mexico" in query.lower()
            or "texas" in query.lower()
        ):
            domain_terms.append("southwest US")

        # Extract core terms (first 3-4 words) for more focused queries
        primary_terms = " ".join(query.split()[:4])

        # Add domain-specific terms if they're not already in the primary terms
        for term in domain_terms:
            if term.lower() not in primary_terms.lower():
                primary_terms = f"{primary_terms} {term}"

        # Get success tracker
        success_tracker = getattr(
            state, "category_success_tracker", CategorySuccessTracker()
        )

        # Determine fallback level based on retry count
        fallback_level = min(2, category_state["retry_count"] - 1)
        if fallback_level < 0:
            fallback_level = 0

        # Always use optimized query for better results
        # Use enhanced query optimization
        optimized_query = get_optimized_query(category, primary_terms, fallback_level)
        info_highlight(
            f"Using optimized query for {category} (fallback level {fallback_level}): {optimized_query}"
        )
        query = optimized_query

        # Create a cache key for this search
        import hashlib

        cache_key = f"search_{category}_{hashlib.md5(query.encode()).hexdigest()}"

        # Check if we have cached results
        cached_results = load_checkpoint(cache_key)
        if cached_results:
            info_highlight(f"Using cached search results for {category}")
            search_results = cached_results
        else:
            # Record start time for performance tracking
            start_time = time.time()

            # Execute the search
            from react_agent.tools.jina import search

            # Track active requests and use semaphore for rate limiting
            global API_REQUEST_SEMAPHORE

            # Update active requests counter
            with active_requests_lock:
                active_requests += 1
                current_active = active_requests
                info_highlight(
                    f"Category {category} waiting for API slot (Active requests: {current_active})"
                )

            # Use semaphore to limit concurrent API requests
            async with API_REQUEST_SEMAPHORE:
                info_highlight(f"Executing search for {category}")

                try:
                    # Use the search function directly as it's already async
                    search_results = await search(query)
                    info_highlight(
                        f"Search completed for {category} with {len(search_results) if isinstance(search_results, list) else 'non-list'} results"
                    )

                    # Log the actual response structure for debugging
                    logger.debug(f"Raw search response type: {type(search_results)}")
                    if isinstance(search_results, dict):
                        logger.debug(
                            f"Raw search response keys: {list(search_results.keys())}"
                        )

                    # Cache successful results
                    if search_results and (
                        isinstance(search_results, list)
                        and len(search_results) > 0
                        or isinstance(search_results, dict)
                        and search_results.get("results")
                    ):
                        create_checkpoint(cache_key, search_results)

                except Exception as e:
                    import traceback

                    error_highlight(
                        f"Search failed for {category}: {str(e)}\nException type: {type(e)}\nTraceback: {traceback.format_exc()}"
                    )

                    # Try alternative approach with different query formulation
                    try:
                        fallback_query = (
                            f"{state['original_query']} {category.replace('_', ' ')}"
                        )
                        search_results = await search(fallback_query)
                        info_highlight(
                            f"Fallback search completed for {category} with {len(search_results) if isinstance(search_results, list) else 'non-list'} results"
                        )
                    except Exception as e2:
                        error_highlight(
                            f"Fallback search also failed for {category}: {str(e2)}"
                        )
                        category_state["status"] = "search_failed"
                        success_tracker.track_attempt(category, False)
                        return

        # Handle empty results with more flexible validation
        if not search_results or (
            isinstance(search_results, (list, dict))
            and not any([
                isinstance(search_results, list) and len(search_results) > 0,
                isinstance(search_results, dict) and search_results.get("results", []),
            ])
        ):
            warning_highlight(f"No valid results for category: {category}")

            # Try one more alternative approach with a simplified query
            try:
                # Create a more focused query by using just the key terms
                if " " in primary_terms:
                    # Extract just the first two words for a more focused query
                    simplified_terms = " ".join(primary_terms.split()[:2])
                    # Add the category name for context
                    simplified_query = (
                        f"{simplified_terms} {category.replace('_', ' ')}"
                    )
                else:
                    # If primary_terms is just one word, use it with the category
                    simplified_query = f"{primary_terms} {category.replace('_', ' ')}"

                info_highlight(
                    f"Trying simplified query for {category}: {simplified_query}"
                )

                # Use the search function directly
                search_results = await search(simplified_query)
                info_highlight(
                    f"Simplified search completed for {category} with {len(search_results) if isinstance(search_results, list) else 'non-list'} results"
                )

                if not search_results or (
                    isinstance(search_results, (list, dict))
                    and not any([
                        isinstance(search_results, list) and len(search_results) > 0,
                        isinstance(search_results, dict)
                        and search_results.get("results", []),
                    ])
                ):
                    category_state["status"] = "search_failed"
                    success_tracker.track_attempt(category, False)
                    return
            except Exception as e:
                error_highlight(f"Simplified search failed for {category}: {str(e)}")
                category_state["status"] = "search_failed"
                success_tracker.track_attempt(category, False)
                return

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Standardize and enhance results
        formatted_results = standardize_search_result(
            search_results, category=category, query=query, start_time=start_time
        )

        enhanced_results = enhance_search_results(
            formatted_results,
            category=category,
            query=query,
            processing_time_ms=processing_time_ms,
        )

        # Update the category state
        category_state["search_results"] = enhanced_results
        category_state["status"] = "searched"

        # Log search event
        log_search_event(
            category=category,
            query=query,
            result_count=len(enhanced_results),
            duration_ms=processing_time_ms,
            success=True,
        )

        # Track successful attempt
        success_tracker.track_attempt(category, True)

        # Add results to the processing queue
        for result in enhanced_results:
            await results_queue.put((category, result, state["original_query"]))

    except Exception as e:
        error_highlight(f"Error in _search_and_process for {category}: {str(e)}")

        # Update category state
        if category in state["categories"]:
            state["categories"][category]["status"] = "search_failed"

        # Track failed attempt
        success_tracker = getattr(
            state, "category_success_tracker", CategorySuccessTracker()
        )
        success_tracker.track_attempt(category, False)

    finally:
        # Decrement active requests counter
        with active_requests_lock:
            active_requests -= 1
            info_highlight(
                f"Completed search for category {category} (Active requests: {active_requests})"
            )


async def _process_results_queue(
    state: Dict[str, Any],
    results_queue: asyncio.Queue,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Process search results from the queue as they arrive.

    Args:
        state: The current research state
        results_queue: Queue containing search results to process
        config: Optional configuration
    """
    while True:
        try:
            # Get the next item from the queue
            item = await results_queue.get()

            if item is None:  # End marker
                break

            category, result, original_query = item

            # Process the result
            facts, sources, statistics = await _process_search_result(
                result=result,
                category=category,
                original_query=original_query,
                config=config,
            )

            # Update the category state
            if category in state["categories"]:
                category_state = state["categories"][category]

                # Ensure category_state fields are lists before extending
                if not isinstance(category_state.get("extracted_facts"), list):
                    category_state["extracted_facts"] = []
                if not isinstance(category_state.get("sources"), list):
                    category_state["sources"] = []
                if not isinstance(category_state.get("statistics"), list):
                    category_state["statistics"] = []

                # Add facts, sources, and statistics with type checking
                if isinstance(facts, list):
                    category_state["extracted_facts"].extend(facts)
                else:
                    warning_highlight(f"Category {category}: Expected list for facts, got {type(facts)}. Skipping facts.")

                if isinstance(sources, list):
                    category_state["sources"].extend(sources)
                else:
                    warning_highlight(f"Category {category}: Expected list for sources, got {type(sources)}. Skipping sources.")

                if isinstance(statistics, list):
                    category_state["statistics"].extend(statistics)
                else:
                    warning_highlight(f"Category {category}: Expected list for statistics, got {type(statistics)}. Skipping statistics.")

                # Mark as complete if we have enough facts
                if isinstance(category_state["extracted_facts"], list) and len(category_state["extracted_facts"]) >= 3:
                    category_state["complete"] = True
                    category_state["status"] = "facts_extracted"

                    # Cache the results
                    try:
                        cache_key = f"category_{original_query}_{category}"
                        cache_data = {
                            "search_results": category_state.get("search_results", []),
                            "extracted_facts": category_state.get(
                                "extracted_facts", []
                            ),
                            "sources": category_state.get("sources", []),
                        }
                        create_checkpoint(
                            cache_key, cache_data, ttl=86400
                        )  # Cache for 24 hours
                        info_highlight(f"Cached results for category: {category}")
                    except Exception as e:
                        error_highlight(
                            f"Error caching results for {category}: {str(e)}"
                        )

            # Mark the task as done
            results_queue.task_done()

        except Exception as e:
            error_highlight(f"Error processing result from queue: {str(e)}")
            try:
                results_queue.task_done()
            except:
                pass


async def _process_search_result(
    result: Dict[str, Any],
    category: str,
    original_query: str,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process a single search result and extract information.

    Args:
        result: The search result to process
        category: The research category
        original_query: The original search query
        config: Optional configuration

    Returns:
        Tuple of (facts, sources, statistics)
    """
    url = result.get("url", "")
    if not url or should_skip_content(url):
        return [], [], []

    # Check checkpoint first
    cache_key = f"search_result_{url}_{category}"
    if cached_state := load_checkpoint(cache_key):
        info_highlight(f"Using cached result for {url} in {category}")
        return (
            cached_state.get("facts", []),
            cached_state.get("sources", []),
            cached_state.get("statistics", []),
        )

    content = result.get("snippet", "")
    if not validate_content(content):
        return [], [], []

    content_type = detect_content_type(url, content)

    # Handle document file types with Docling if available
    from react_agent.utils.content import (
        DOCLING_AVAILABLE,
        process_document_with_docling,
    )

    if DOCLING_AVAILABLE and content_type in (
        "pdf",
        "doc",
        "excel",
        "presentation",
        "document",
    ):
        try:
            info_highlight(
                f"Processing document with Docling: {url}", category="extraction"
            )
            extracted_text, detected_type = process_document_with_docling(url)
            if validate_content(extracted_text):
                content = extracted_text
                content_type = detected_type
                info_highlight(
                    f"Successfully extracted text from document: {url}",
                    category="extraction",
                )
            else:
                warning_highlight(
                    f"Extracted text from document didn't meet validation criteria: {url}",
                    category="extraction",
                )
        except Exception as e:
            warning_highlight(
                f"Error processing document with Docling: {str(e)}",
                category="extraction",
            )

    # Get extraction prompt
    from react_agent.prompts.research import get_extraction_prompt

    prompt_template = get_extraction_prompt(
        category=category, query=original_query, url=url, content=content
    )

    try:
        # Extract category information
        from react_agent.utils.llm import call_model_json, get_extraction_model

        extraction_model = get_extraction_model()

        # Try using the extraction tool first
        try:
            from react_agent.tools.derivation import ExtractionTool

            extraction_tool = ExtractionTool()

            extraction_result = await extraction_tool.ainvoke(
                {
                    "operation": "category",
                    "text": content,
                    "url": url,
                    "source_title": result.get("title", ""),
                    "category": category,
                    "original_query": original_query,
                    "extraction_model": extraction_model,
                },
                config,
            )

            facts = extraction_result.get("facts", [])
            relevance_score = extraction_result.get("relevance", 0.5)
        except Exception as e:
            # Fall back to the original method
            from react_agent.utils.extraction import extract_category_information

            facts, relevance_score = await extract_category_information(
                content=content,
                url=url,
                title=result.get("title", ""),
                category=category,
                original_query=original_query,
                prompt_template=prompt_template,
                extraction_model=call_model_json,
                config=config,
            )

        # Extract statistics
        statistics = []
        try:
            # Try to use the statistics analysis tool
            from react_agent.tools.derivation import StatisticsAnalysisTool

            statistics_tool = StatisticsAnalysisTool()

            # Create a synthesis object with the content
            synthesis_data = {
                "synthesis": {
                    "main": {
                        "content": content,
                        "citations": [{"url": url, "title": result.get("title", "")}],
                        "statistics": []
                    }
                }
            }

            statistics = await statistics_tool.ainvoke(
                {
                    "operation": "synthesis",
                    "synthesis": synthesis_data
                },
                config,
            )

            if statistics:
                info_highlight(f"Successfully extracted {len(statistics)} statistics")
        except Exception as stats_error:
            warning_highlight(
                f"Error using statistics analysis tool: {str(stats_error)}"
            )

        # Add source information to facts
        for fact in facts:
            fact["source_url"] = url
            fact["source_title"] = result.get("title", "")

        source = {
            "url": url,
            "title": result.get("title", ""),
            "published_date": result.get("published_date"),
            "fact_count": len(facts),
            "relevance_score": relevance_score,
            "quality_score": result.get("quality_score", 0.5),
            "content_type": content_type,
        }

        result_tuple = (facts, [source], statistics)

        # Save to checkpoint with TTL
        create_checkpoint(
            cache_key,
            {
                "facts": facts,
                "sources": [source],
                "statistics": statistics,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            ttl=3600,  # 1 hour TTL
        )

        return result_tuple

    except Exception as e:
        warning_highlight(f"Error extracting from {url}: {str(e)}")
        return [], [], []
