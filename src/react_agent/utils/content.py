"""Content processing utilities for the research agent.

This module provides utilities for processing and validating content,
including chunking, preprocessing, content type detection, document processing,
and merging extraction results.

Examples:
    >>> text = "This is a sample text that will be split into chunks."
    >>> chunks = chunk_text(text, chunk_size=20, overlap=5)
    >>> print(chunks)
    ['This is a sample', 'sample text that', 'that will be split', 'split into chunks.']
    
    >>> content_type = detect_content_type("page.html", "<html><body>Content</body></html>")
    >>> print(content_type)
    html
    
    >>> valid = validate_content("This is valid content")
    >>> print(valid)
    True

    >>> document_text, doc_type = process_document("document.pdf", document_content)
    >>> print(doc_type)
    pdf
"""

import contextlib
import hashlib
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, TypedDict
from urllib.parse import unquote, urlparse

import nltk.data  # type: ignore[import]
import requests  # type: ignore[import]
from docling.document_converter import DocumentConverter  # type: ignore[import]

from react_agent.utils.cache import ProcessorCache
from react_agent.utils.defaults import (
    DATA_PATH_PATTERNS,
    DOCUMENT_PATTERNS,
    DOCUMENT_TYPE_MAPPING,
    HTML_PATH_PATTERNS,
    MARKETPLACE_PATTERNS,
    MAX_CONTENT_LENGTH,
    PROBLEMATIC_PATTERNS,
    PROBLEMATIC_SITES,
    PROCUREMENT_HTML_PATTERNS,
    REPORT_PATTERNS,
    TOKEN_CHAR_RATIO,
    ChunkConfig,
    get_category_merge_mapping,
    get_default_extraction_result,
)
from react_agent.utils.logging import (
    error_highlight,
    get_logger,
    info_highlight,
    log_performance_metrics,
    log_progress,
    warning_highlight,
)

# Initialize logger
logger = get_logger(__name__)

# Initialize cache
content_cache = ProcessorCache(thread_id="content")


class ContentState(TypedDict):
    """Typed dictionary for content state used in the graph.

    Attributes:
        content (str): The actual text content.
        url (str): Source URL of the content.
        content_type (str): Type of the content (e.g., 'html', 'json', 'text').
        chunks (List[str]): List of text chunks.
        metadata (Dict[str, Any]): Additional metadata about the content.
        timestamp (str): ISO formatted timestamp when the content was processed.
    """

    content: str
    url: str
    content_type: str
    chunks: List[str]
    metadata: Dict[str, Any]
    timestamp: str


class DocumentProcessingResult(TypedDict):
    """Typed dictionary for document processing results.

    Attributes:
        text (str): The extracted text content.
        content_type (str): The document type (e.g., 'pdf', 'doc', 'excel').
        metadata (Dict[str, Any]): Additional metadata about the document.
    """

    text: str
    content_type: str
    metadata: Dict[str, Any]


# Load the Punkt sentence tokenizer (moved outside the function)
try:
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
except LookupError:
    error_highlight(
        "NLTK Punkt tokenizer not found. Please download it.", category="nltk"
    )
    tokenizer = None


@content_cache.cache_result(ttl=3600)
def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    use_large_chunks: bool = False,
    min_chunk_size: int = 100,
) -> List[str]:
    """Split text into overlapping chunks, returning a list of chunks.

    Args:
        text: The text to be split.
        chunk_size: Desired size for each chunk. If None, defaults are taken from ChunkConfig.
        overlap: Overlap length between chunks. If None, defaults are taken from ChunkConfig.
        use_large_chunks: Flag to indicate if larger chunk sizes should be used.
        min_chunk_size: Minimum acceptable chunk size to avoid very small chunks.

    Returns:
        A list of text chunks.

    Examples:
        >>> text = "This is a sample text that needs to be chunked into smaller pieces."
        >>> chunk_text(text, chunk_size=20, overlap=5)
        ['This is a sample', 'sample text that', 'that needs to be', 'be chunked into', 'into smaller pieces.']
    """
    if not text or text.isspace():
        warning_highlight("Empty or whitespace-only text provided")
        return []

    # Set chunk parameters based on defaults if not provided.
    chunk_size = max(
        min_chunk_size,
        chunk_size
        or (
            ChunkConfig.LARGE_CHUNK_SIZE
            if use_large_chunks
            else ChunkConfig.DEFAULT_CHUNK_SIZE
        ),
    )
    overlap = min(
        chunk_size - 1,
        max(
            0,
            overlap
            or (
                ChunkConfig.LARGE_OVERLAP
                if use_large_chunks
                else ChunkConfig.DEFAULT_OVERLAP
            ),
        ),
    )

    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk size.")

    chunks: List[str] = []
    start = 0
    text_length = len(text)
    expected_chunks = math.ceil(text_length / (chunk_size - overlap))

    # Track chunking performance
    start_time = time.time()

    def _create_chunks(
            text: str,
            start: int,
            chunk_size: int,
            overlap: int,
            min_chunk_size: int,
            chunks: List[str],
            tokenizer: Any,
            expected_chunks: int,
        ) -> Tuple[int, List[str]]:
        text_length = len(text)
        end = min(start + chunk_size, text_length)
        if tokenizer and end < text_length:
            sentences = tokenizer.tokenize(text[start:end])
            end = start + len(sentences[-2]) if len(sentences) > 1 else end
        if chunk := text[start:end].strip():
            if chunks and len(chunk) < min_chunk_size:
                chunks[-1] = f"{chunks[-1]} {chunk}"
            else:
                chunks.append(chunk)
                total_chunks = math.ceil(text_length / chunk_size)
                log_frequency = max(1, total_chunks // 20)
                if len(chunks) % log_frequency == 0:
                    log_progress(
                        len(chunks), expected_chunks, "chunking", "Creating chunks"
                    )
        start = end - overlap
        return start, chunks

    create_chunks_func = _create_chunks if tokenizer else _create_chunks

    while start < text_length:
        start, chunks = create_chunks_func(
            text,
            start,
            chunk_size,
            overlap,
            min_chunk_size,
            chunks,
            tokenizer or None,
            expected_chunks,
        )

    end_time = time.time()
    log_performance_metrics(
        "Text chunking",
        start_time,
        end_time,
        "chunking",
        {
            "text_length": text_length,
            "chunks_created": len(chunks),
            "avg_chunk_size": text_length / max(1, len(chunks)),
        },
    )

    info_highlight(f"Created {len(chunks)} chunks", category="chunking")
    return chunks


# Content type detection functions
def detect_html(content: str) -> str | None:
    """Determine whether the provided content is HTML."""
    if not content:
        return None
    if re.search(r"<!DOCTYPE html>|<html|<body|<div", content, re.IGNORECASE):
        return "html"
    return None


def detect_json(content: str) -> dict | None:
    """Determine whether the provided content is valid JSON."""
    if not content:
        return None
    content = content.strip()
    if not content:
        return None
    if (content.startswith("{") and content.endswith("}")) or (
        content.startswith("[") and content.endswith("]")
    ):
        with contextlib.suppress(json.JSONDecodeError):
            json.loads(content)
            return {"type": "json"}
    return None


def detect_from_url_extension(url: str) -> str | None:
    """Infer content type based on the file extension in the URL."""
    if not url:
        return None

    try:
        ext = f".{url.lower().split('.')[-1]}"
        return DOCUMENT_TYPE_MAPPING.get(ext)
    except IndexError:
        return None


def detect_from_url_path(url: str) -> str | None:
    """Infer content type from common URL path patterns."""
    if not url:
        return None

    parsed_url = urlparse(url)
    url_path = parsed_url.path.lower()

    # Check patterns in priority order
    # 1. Data feeds have highest priority
    if any(pattern in url_path for pattern in DATA_PATH_PATTERNS):
        return "data"

    # 2. Procurement HTML content
    if any(pattern in url_path for pattern in PROCUREMENT_HTML_PATTERNS):
        return "html"

    # 3. Marketplace/catalogue pages (overlap with procurement)
    if any(pattern in url_path for pattern in MARKETPLACE_PATTERNS):
        return "html"

    # 4. Documents (file downloads, reports, etc.)
    if any(pattern in url_path for pattern in DOCUMENT_PATTERNS):
        return "document"

    # 5. Reports or research-oriented content
    if any(pattern in url_path for pattern in REPORT_PATTERNS):
        return "report"

    # 6. General HTML content (lowest priority)
    if any(pattern in url_path for pattern in HTML_PATH_PATTERNS):
        return "html"

    return None


def detect_from_content_heuristics(content: str) -> str | None:
    r"""Infer content type based on heuristics applied directly to the content."""
    if not content or len(content) < 50:
        return None

    content = content.strip()
    if content.startswith("<?xml") or (content.startswith("<") and ">" in content):
        return "xml"
    if "{" in content and "}" in content and '"' in content and ":" in content:
        return "data"
    return "text" if "\n\n" in content and len(content) > 200 else None


def detect_from_url_domain(url: str) -> str | None:
    """Infer content type based on URL domain characteristics."""
    if not url:
        return None

    common_web_domains = (".gov", ".org", ".edu", ".com", ".net", ".io")
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    if any(domain.endswith(d) for d in common_web_domains) and (
        parsed_url.path and "." not in parsed_url.path.split("/")[-1]
    ):
        return "html"
    return None


def fallback_detection(url: str, content: str) -> str:
    """Fallback detection logic for content type."""
    if content and content.strip():
        return "text"
    return "html" if url and url.startswith(("http://", "https://")) else "unknown"


# Document processing functions
def detect_document_type(url: str) -> str | None:
    """Detect if the URL points to a document file that can be processed by Docling.

    Args:
        url: The URL to check

    Returns:
        The document type if detectable, None otherwise
    """
    if not url:
        return None

    url_lower = url.lower()
    file_ext = Path(urlparse(url_lower).path).suffix.lower()

    return DOCUMENT_TYPE_MAPPING.get(file_ext)


def is_document_url(url: str) -> bool:
    """Check if the URL points to a document that can be processed.

    Args:
        url: The URL to check

    Returns:
        True if the URL points to a processable document, False otherwise
    """
    return detect_document_type(url) is not None


def download_document(url: str, timeout: int = 30) -> bytes | None:
    """Download document content from the provided URL.

    Args:
        url: The URL to download from
        timeout: Request timeout in seconds

    Returns:
        Document content as bytes if successful, None otherwise

    Raises:
        requests.RequestException: If the download fails
    """
    try:
        info_highlight(
            f"Downloading document from {url}", category="document_processing"
        )
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        error_highlight(
            f"Error downloading document: {str(e)}", category="document_processing"
        )
        raise


def create_temp_document_file(content: bytes, file_extension: str) -> Tuple[str, str]:
    """Create a temporary file for document processing.

    Args:
        content: The document content as bytes
        file_extension: The file extension (including dot)

    Returns:
        Tuple containing the temporary file path and name
    """
    with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
        temp_file.write(content)
        return temp_file.name, os.path.basename(temp_file.name)


def extract_document_text(file_path: str, document_type: str) -> str:
    """Extract text from a document file using Docling.

    Args:
        file_path: Path to the document file
        document_type: Type of document (pdf, doc, etc.)

    Returns:
        Extracted text content
    """
    try:
        converter = DocumentConverter()
        result = converter.convert(file_path)
        # Handle the ConversionResult type by extracting the text content
        return str(result)
    except Exception as e:
        error_highlight(
            f"Error extracting text from {document_type} document: {str(e)}",
            category="document_processing",
        )
        return f"Failed to extract text from {document_type} document: {str(e)}"


def process_document(url: str, content: bytes | None = None) -> Tuple[str, str]:
    """Process a document file and extract its text content.

    Args:
        url: The URL of the document
        content: The document content as bytes, if already available

    Returns:
        Tuple containing (extracted text, content type)

    Raises:
        ValueError: If document processing fails
    """
    start_time = time.time()
    document_type = detect_document_type(url) or "document"

    try:
        # Check if result is already cached

        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
        cache_key = f"document_processing_{url_hash}"
        cached_result = content_cache.get(cache_key)

        if cached_result and isinstance(cached_result, dict):
            info_highlight(
                f"Using cached document extraction for {url}",
                category="document_processing",
            )
            log_performance_metrics(
                "Document processing (cached)",
                start_time,
                time.time(),
                "document_processing",
                {"cache_hit": True},
            )
            return cached_result.get("text", ""), cached_result.get(
                "content_type", document_type
            )

        # Download content if not provided
        if content is None:
            content = download_document(url)
            if not content:
                raise ValueError(f"Failed to download document from {url}")

        # Create temporary file
        file_ext = os.path.splitext(Path(urlparse(url).path).name)[1].lower()
        temp_file_path, temp_file_name = create_temp_document_file(content, file_ext)

        try:
            # Extract text using Docling
            extracted_text = extract_document_text(temp_file_path, document_type)

            # Cache the result
            result: DocumentProcessingResult = {
                "text": extracted_text,
                "content_type": document_type,
                "metadata": {
                    "url": url,
                    "file_type": file_ext,
                    "processing_time": time.time() - start_time,
                },
            }

            content_cache.put(
                cache_key,
                result,
                ttl=3600,  # 1 hour TTL
            )

            log_performance_metrics(
                "Document processing",
                start_time,
                time.time(),
                "document_processing",
                {"content_type": document_type, "text_length": len(extracted_text)},
            )

            return extracted_text, document_type

        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                warning_highlight(
                    f"Error removing temporary file: {str(e)}",
                    category="document_processing",
                )

    except Exception as e:
        error_highlight(
            f"Error processing document: {str(e)}", category="document_processing"
        )
        raise ValueError(f"Failed to process document: {str(e)}") from e


def process_document_with_docling(
    url: str, content: bytes | None = None
) -> Tuple[str, str]:
    """Legacy wrapper for process_document function.

    This function maintains backward compatibility with existing code.

    Args:
        url: The URL of the document to process
        content: Binary content if already available

    Returns:
        Tuple containing (extracted text, content type)
    """
    return process_document(url, content)


def should_skip_content(url: str) -> bool:
    """Determine if the content from a given URL should be skipped based on certain rules."""
    try:
        decoded_url = unquote(url).lower()
    except Exception as e:
        error_highlight(f"Error decoding URL: {str(e)}", category="validation")
        decoded_url = url.lower()

    # Check if it's a document format that Docling can handle
    if is_document_url(url):
        info_highlight(
            f"Document format detected, will process with Docling: {url}",
            category="validation",
        )
        return False
    else:
        # Enhanced PDF detection when Docling is not available
        if any(p in decoded_url for p in (".pdf", "%2Fpdf", "%3Fpdf")):
            info_highlight(
                f"Skipping PDF content (Docling not available): {url}",
                category="validation",
            )
            return True

        # MIME-type pattern detection
        mime_patterns = [
            r"application/pdf",
            r"application/\w+?pdf",
            r"content-type:.*pdf",
        ]
        if any(re.match(p, decoded_url) for p in mime_patterns):
            info_highlight(
                f"Skipping PDF MIME-type pattern (Docling not available): {url}",
                category="validation",
            )
            return True

    url_lower = url.lower()

    # Check for problematic file types
    for pattern in PROBLEMATIC_PATTERNS:
        if re.search(pattern, url_lower):
            info_highlight(
                f"Skipping content with pattern {pattern}: {url}", category="validation"
            )
            return True

    # Check for problematic sites
    domain = urlparse(url).netloc.lower()
    for site in PROBLEMATIC_SITES:
        if site in domain:
            info_highlight(
                f"Skipping content from problematic site {site}: {url}",
                category="validation",
            )
            return True

    return False


CONTENT_DETECTION_STRATEGIES = [
    {"strategy": detect_html, "weight": 0.8, "content_based": True},
    {"strategy": detect_json, "weight": 0.7, "content_based": True},
    {"strategy": detect_from_url_extension, "weight": 0.9, "content_based": False},
    {"strategy": detect_from_url_path, "weight": 0.6, "content_based": False},
    {"strategy": detect_from_content_heuristics, "weight": 0.5, "content_based": True},
    {"strategy": detect_from_url_domain, "weight": 0.4, "content_based": False},
]


def detect_content_type(url: str, content: str) -> str:
    """Determine the content type using multiple detection strategies.

    This function attempts to identify the content type of a given URL and content
    using a series of detection strategies, falling back to a default type if
    none of the strategies are successful.

    Args:
        url: The URL of the content.
        content: The actual content.

    Returns:
        The detected content type as a string (e.g., 'html', 'json', 'text').

    Examples:
        >>> url = "https://example.com/page.html"
        >>> content = "<html><body>Content</body></html>"
        >>> detect_content_type(url, content)
        'html'

        >>> url = "https://api.example.com/data.json"
        >>> content = '{"key": "value"}'
        >>> detect_content_type(url, content)
        'json'
    """
    info_highlight(f"Detecting content type for URL: {url}", category="content_type")

    if doc_type := detect_document_type(url):
        info_highlight(f"Detected document type: {doc_type}", category="content_type")
        return doc_type

    for strategy_data in CONTENT_DETECTION_STRATEGIES:
        strategy = strategy_data["strategy"]
        content_based = strategy_data["content_based"]

        try:
            if result := strategy(content if content_based else url):
                info_highlight(f"Detected content type: {result}", category="content_type")
                return result
        except TypeError:
            # Skip if strategy is not callable
            continue

    return fallback_detection(url, content)


MIN_CONTENT_LENGTH = 10  # Or retrieve from a config file/env variable


def validate_content(content: str) -> bool:
    """Validate that the provided content meets minimum requirements."""
    if not content or not isinstance(content, str):
        warning_highlight(
            "Invalid content type or empty content", category="validation"
        )
        return False

    if len(content) < MIN_CONTENT_LENGTH:  # Minimum content length requirement
        warning_highlight(
            f"Content too short: {len(content)} characters", category="validation"
        )
        return False

    return True


def preprocess_content(content: str, url: str) -> str:
    """Clean and preprocess content prior to further processing or model ingestion."""
    if not content:
        return ""

    # Remove excessive whitespace
    content = re.sub(r"\s+", " ", content)

    # Trim content to maximum length if needed
    if len(content) > MAX_CONTENT_LENGTH:
        info_highlight(
            f"Trimming content from {len(content)} to {MAX_CONTENT_LENGTH} characters",
            category="preprocessing",
        )
        content = content[:MAX_CONTENT_LENGTH]

    # Basic cleaning of HTML remnants if they exist
    content = re.sub(r"<[^>]*>", " ", content)

    # Normalize whitespace after cleaning
    content = re.sub(r"\s+", " ", content).strip()

    return content


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text based on a fixed character-to-token ratio."""
    return int(len(text) / TOKEN_CHAR_RATIO) if text else 0


# Field merging utilities for handling chunk results
def _handle_list_extend(
    merged_field: List[Any], value: List[Any], seen_items: Set[str]
) -> None:
    """Handle the 'extend' operation for list values."""
    # Convert the entire value to string for deduplication
    value_str = json.dumps(value, sort_keys=True)
    
    # Add the entire list as an item if not already seen
    if value_str not in seen_items:
        seen_items.add(value_str)
        merged_field.append(value)


def _handle_dict_update(merged_field: Dict[str, Any], value: Dict[str, Any]) -> None:
    """Handle the 'update' operation for dictionary values."""
    # Update dictionary fields, but don't overwrite with empty values
    for k, v in value.items():
        if (v or v == 0) and (k not in merged_field or not merged_field[k]):
            merged_field[k] = v


def _handle_numeric_operation(
    merged_field: float | None, value: float, operation: str
) -> float | None:
    """Handle numeric operations (max, min)."""
    if (
        operation == "max"
        and (merged_field is None or value > merged_field)
        or operation != "max"
        and operation == "min"
        and (merged_field is None or value < merged_field)
    ):
        return value
    return merged_field


def _handle_average_collection(
    merged: Dict[str, Any], field: str, value: float
) -> None:
    """Collect values for average operation."""
    if "values" not in merged:
        merged["values"] = {}
    if field not in merged["values"]:
        merged["values"][field] = []
    merged["values"][field].append(value)


def _initialize_field(merged: Dict[str, Any], field: str, value: Any) -> None:
    """Initialize a field in the merged dictionary if it doesn't exist."""
    if field not in merged:
        if isinstance(value, list):
            merged[field] = []
        elif isinstance(value, dict):
            merged[field] = {}
        else:
            merged[field] = None


def _merge_field(
    merged: Dict[str, Any],
    results: List[Dict[str, Any]],
    field: str,
    operation: str,
    seen_items: Set[str],
) -> None:
    """Merge a specific field from result dictionaries into the merged dictionary."""
    for result in results:
        if field not in result:
            continue

        value = result[field]
        if not value and value != 0:  # Skip empty values but keep zeros
            continue

        # Initialize field if it doesn't exist in merged dict
        _initialize_field(merged, field, value)

        # Handle different merge operations
        if operation == "extend" and isinstance(value, list):
            _handle_list_extend(merged[field], value, seen_items)

        elif operation == "update" and isinstance(value, dict):
            if not isinstance(merged[field], dict):
                merged[field] = {}
            _handle_dict_update(merged[field], value)

        elif operation in {"max", "min"} and isinstance(value, (int, float)):
            merged[field] = _handle_numeric_operation(merged[field], value, operation)

        elif operation in {"average", "avg"} and isinstance(value, (int, float)):
            _handle_average_collection(merged, field, value)

        elif operation == "first" and merged[field] is None:
            # Take first non-empty value
            merged[field] = value


def _get_total_fields(
    merge_strategy: Dict[str, str], results: List[Dict[str, Any]]
) -> int:
    """Determine the total number of fields for progress tracking."""
    total_fields = len(merge_strategy) if merge_strategy else 0
    if total_fields == 0 and results:
        # If no merge strategy, count fields in first result
        total_fields = len(results[0].keys())
    return total_fields


def _process_merge_strategy_fields(
    merged: Dict[str, Any],
    results: List[Dict[str, Any]],
    merge_strategy: Dict[str, str],
    seen_items: Set[str],
    category: str,
) -> int:
    """Process fields according to the merge strategy."""
    total_fields = _get_total_fields(merge_strategy, results)
    current_field = 0

    for field, operation in merge_strategy.items():
        _merge_field(merged, results, field, operation, seen_items)
        current_field += 1
        if total_fields > 0:
            log_progress(
                current_field, total_fields, "merging", f"Merging {category} results"
            )

    return current_field


def _process_remaining_fields(
    merged: Dict[str, Any],
    results: List[Dict[str, Any]],
    merge_strategy: Dict[str, str],
    seen_items: Set[str],
) -> None:
    """Process fields in results that aren't in the merge strategy."""
    for result in results:
        for field in result:
            if field not in merge_strategy and field not in merged:
                # Select appropriate merge operation based on field type
                if isinstance(result[field], list):
                    _merge_field(merged, results, field, "extend", seen_items)
                elif isinstance(result[field], dict):
                    _merge_field(merged, results, field, "update", seen_items)
                elif isinstance(result[field], (int, float)):
                    _merge_field(merged, results, field, "max", seen_items)
                elif result[field] is not None:
                    # For other types, just take the first non-None value
                    merged[field] = result[field]


def _finalize_averages(merged: Dict[str, Any], merge_strategy: Dict[str, str]) -> None:
    """Calculate final averages for 'avg' operations."""
    if "values" not in merged:
        return

    for field, values in merged.get("values", {}).items():
        if values:
            # Calculate the average of collected values
            merged[field] = sum(values) / len(values)

    # Remove the temporary values storage
    if "values" in merged:
        del merged["values"]


def merge_chunk_results(
    results: List[Dict[str, Any]],
    category: str,
    merge_strategy: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Merge multiple chunk extraction results into a single consolidated result."""
    if not results:
        warning_highlight(f"No results to merge for category: {category}")
        return get_default_extraction_result(category)

    start_time = time.time()

    info_highlight(f"Merging {len(results)} chunk results for category: {category}")

    # Get the default merge strategy if none provided
    if not merge_strategy:
        merge_strategy = get_category_merge_mapping(category)

    merged: Dict[str, Any] = {}
    seen_items: Set[str] = set()

    # Process fields according to merge strategy
    _process_merge_strategy_fields(
        merged, results, merge_strategy, seen_items, category
    )

    # Process any remaining fields not in the merge strategy
    _process_remaining_fields(merged, results, merge_strategy, seen_items)

    # Calculate final averages
    _finalize_averages(merged, merge_strategy)

    end_time = time.time()
    log_performance_metrics(
        f"Merging {category} results",
        start_time,
        end_time,
        "merging",
        {"num_results": len(results), "num_fields": len(merged)},
    )

    return merged


# Export public functions
__all__ = [
    "chunk_text",
    "preprocess_content",
    "estimate_tokens",
    "should_skip_content",
    "merge_chunk_results",
    "validate_content",
    "detect_content_type",
    "process_document",
    "process_document_with_docling",
    "is_document_url",
]
