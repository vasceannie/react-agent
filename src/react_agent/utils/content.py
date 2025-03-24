"""Content processing utilities for the research agent.

This module provides utilities for processing and validating content,
including chunking, preprocessing, and content type detection.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, TypedDict, Set, Literal
import re
from urllib.parse import urlparse, unquote
import logging
from datetime import datetime, timezone
import json
import contextlib
import urllib.parse
from react_agent.utils.logging import (
    get_logger,
    info_highlight, 
    warning_highlight,
    error_highlight,
    log_progress,
    log_performance_metrics
)
from react_agent.utils.extraction import safe_json_parse
from react_agent.utils.defaults import ChunkConfig, get_default_extraction_result, get_category_merge_mapping
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from react_agent.utils.cache import create_checkpoint, load_checkpoint, cache_result

# Initialize logger
logger = get_logger(__name__)

class ContentState(TypedDict):
    """Type definition for content state in the graph."""
    content: str
    url: str
    content_type: str
    chunks: List[str]
    metadata: Dict[str, Any]
    timestamp: str

def create_checkpoint(key: str, data: Dict[str, Any], ttl: int = 3600) -> None:
    """Create a checkpoint using the caching system.
    
    Args:
        key: Unique identifier for the checkpoint
        data: Data to store in the checkpoint
        ttl: Time to live in seconds (default 1 hour)
        
    Examples:
        >>> data = {
        ...     "content": "Sample content",
        ...     "url": "example.com",
        ...     "content_type": "text"
        ... }
        >>> create_checkpoint("test_key", data)
        
        >>> create_checkpoint("temp_key", data, ttl=1800)  # 30 minute TTL
    """
    # Use the imported create_checkpoint function
    create_checkpoint(key, data, ttl)

def load_checkpoint(key: str) -> Optional[Dict[str, Any]]:
    """Load a checkpoint using the caching system.
    
    Args:
        key: Unique identifier for the checkpoint
        
    Returns:
        Channel values if found, None otherwise
        
    Examples:
        >>> # Assuming checkpoint exists
        >>> load_checkpoint("test_key")
        {'content': 'Sample content', 'url': 'example.com', 'content_type': 'text'}
        
        >>> # Non-existent checkpoint
        >>> load_checkpoint("missing_key")
        None
    """
    # Use the imported load_checkpoint function
    return load_checkpoint(key)

# Constants
DEFAULT_CHUNK_SIZE: int = 40000
DEFAULT_OVERLAP: int = 5000
MAX_CONTENT_LENGTH: int = 100000
TOKEN_CHAR_RATIO: float = 4.0

# Problematic content patterns
PROBLEMATIC_PATTERNS: List[str] = [
    r'\.pdf(\?|$)',  # Modified to catch PDF URLs with query params
    r'\.docx?$',
    r'\.xlsx?$',
    r'\.ppt$',
    r'\.zip$',
    r'\.rar$',
    r'\.exe$',
    r'\.dmg$',
    r'\.iso$',
    r'\.tar$',
    r'\.gz$'
]

# Known problematic sites
PROBLEMATIC_SITES: List[str] = [
    'iaeme.com',
    'scribd.com',
    'slideshare.net',
    'academia.edu'
]

@cache_result(ttl=3600)
def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    use_large_chunks: bool = False,
    min_chunk_size: int = 100
) -> List[str]:
    """Split text into overlapping chunks with enhanced robustness and caching.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk (defaults to ChunkConfig values)
        overlap: Overlap between chunks (defaults to ChunkConfig values)
        use_large_chunks: Whether to use large chunk sizes
        min_chunk_size: Minimum size for a chunk to avoid too small chunks
        
    Returns:
        List of text chunks
        
    Examples:
        >>> text = "This is a sample text that needs to be chunked into smaller pieces."
        >>> chunk_text(text, chunk_size=20, overlap=5)
        ['This is a sample', 'sample text that', 'that needs to be', 'be chunked into', 'into smaller pieces.']
        
        >>> # Using large chunks
        >>> chunk_text(text, use_large_chunks=True)
        ['This is a sample text that needs to be chunked into smaller pieces.']
        
        >>> # Empty input
        >>> chunk_text("")
        []
    """
    if not text or text.isspace():
        warning_highlight("Empty or whitespace-only text provided")
        return []

    # Set chunk parameters
    chunk_size = max(min_chunk_size, chunk_size or (
        ChunkConfig.LARGE_CHUNK_SIZE if use_large_chunks 
        else ChunkConfig.DEFAULT_CHUNK_SIZE
    ))
    overlap = min(chunk_size - 1, max(0, overlap or (
        ChunkConfig.LARGE_OVERLAP if use_large_chunks 
        else ChunkConfig.DEFAULT_OVERLAP
    )))

    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate chunk boundaries
        end = min(start + chunk_size, text_length)

        # Find natural break point
        if end < text_length:
            end = text.rfind(' ', start + min_chunk_size, end) or end

        if chunk := text[start:end].strip():
            if chunks and len(chunk) < min_chunk_size:
                chunks[-1] = f"{chunks[-1]} {chunk}"
            else:
                chunks.append(chunk)

        start = end - overlap

    info_highlight(f"Created {len(chunks)} chunks", category="chunking")
    return chunks

def detect_html(content: str) -> Optional[str]:
    """Detect if content is HTML.
    
    Args:
        content: String content to analyze
        
    Returns:
        'html' if HTML is detected, None otherwise
        
    Examples:
        >>> detect_html("<html><body>Hello</body></html>")
        'html'
        
        >>> detect_html("<!doctype html><div>Content</div>")
        'html'
        
        >>> detect_html("Plain text content")
        None
        
        >>> detect_html("")
        None
    """
    if not content:
        return None
    content_lower = content.strip().lower()
    if content_lower.startswith('<!doctype html') or content_lower.startswith('<html'):
        return 'html'
    if '<body' in content_lower and '</body>' in content_lower:
        return 'html'
    if '<div' in content_lower and '</div>' in content_lower:
        return 'html'
    return None

def detect_json(content: str) -> Optional[str]:
    """Detect if content is JSON.
    
    Args:
        content: String content to analyze
        
    Returns:
        'json' if valid JSON is detected, None otherwise
        
    Examples:
        >>> detect_json('{"key": "value"}')
        'json'
        
        >>> detect_json('[1, 2, 3]')
        'json'
        
        >>> detect_json('Invalid content')
        None
        
        >>> detect_json('')
        None
    """
    if not content:
        return None
    content = content.strip()
    if (content.startswith('{') and content.endswith('}')) or (content.startswith('[') and content.endswith(']')):
        with contextlib.suppress(json.JSONDecodeError):
            json.loads(content)
            return 'json'
    return None

def detect_from_url_extension(url: str) -> Optional[str]:
    """Detect content type from URL file extension.
    
    Args:
        url: URL to analyze
        
    Returns:
        Content type based on file extension, or None if not detected
        
    Examples:
        >>> detect_from_url_extension('document.pdf')
        'pdf'
        
        >>> detect_from_url_extension('page.html')
        'html'
        
        >>> detect_from_url_extension('data.json')
        'json'
        
        >>> detect_from_url_extension('noextension')
        None
    """
    if not url:
        return None
        
    extensions = {
        '.pdf': 'pdf', '.doc': 'doc', '.docx': 'doc', '.xls': 'excel', '.xlsx': 'excel',
        '.ppt': 'presentation', '.pptx': 'presentation', '.txt': 'text', '.md': 'text',
        '.rst': 'text', '.html': 'html', '.htm': 'html', '.json': 'json', '.xml': 'xml',
        '.csv': 'data'
    }
    
    try:
        ext = f".{url.lower().split('.')[-1]}"
        return extensions.get(ext)
    except IndexError:
        return None

def detect_from_url_path(url: str) -> Optional[str]:
    """Detect content type from URL path patterns."""
    if not url:
        return None
        
    html_path_patterns = (
        '/wiki/', '/articles/', '/blog/', '/news/',
        '/docs/', '/help/', '/support/', '/pages/',
        '/product/', '/service/', '/consumers/',
        '/detail/', '/view/', '/content/'
    )
    return 'html' if any(pattern in url.lower() for pattern in html_path_patterns) else None

def detect_from_content_heuristics(content: str) -> Optional[str]:
    """Detect content type from content patterns."""
    if not content or len(content) < 50:
        return None

    content = content.strip()
    if content.startswith('<?xml') or (content.startswith('<') and '>' in content):
        return 'xml'
    if '{' in content and '}' in content and '"' in content and ':' in content:
        return 'data'
    return 'text' if '\n\n' in content and len(content) > 200 else None

def detect_from_url_domain(url: str) -> Optional[str]:
    """Detect content type from URL domain."""
    if not url:
        return None
        
    common_web_domains = ('.gov', '.org', '.edu', '.com', '.net', '.io')
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    if any(domain.endswith(d) for d in common_web_domains) and (parsed_url.path and '.' not in parsed_url.path.split('/')[-1]):
        return 'html'
    return None

def detect_content_type(url: str, content: str) -> str:
    """Detect content type from URL and content using a modular approach.
    
    Args:
        url: URL of the content
        content: Content to analyze
        
    Returns:
        Detected content type string
        
    Examples:
        >>> detect_content_type('page.html', '<html><body>Content</body></html>')
        'html'
        
        >>> detect_content_type('data.json', '{"key": "value"}')
        'json'
        
        >>> detect_content_type('article', 'Plain text content\n\nMore content')
        'text'
        
        >>> detect_content_type('', '')
        'unknown'
    """
    info_highlight(f"Detecting content type for URL: {url}", category="content_type")
    
    detector_functions = [
        (detect_html, content),
        (detect_json, content),
        (detect_from_url_extension, url),
        (detect_from_url_path, url),
        (detect_from_content_heuristics, content),
        (detect_from_url_domain, url)
    ]

    for detector, arg in detector_functions:
        if result := detector(arg):
            info_highlight(f"Detected content type: {result}", category="content_type")
            return result

    # Fallback detection
    result = fallback_detection(url, content)
    info_highlight(f"Using fallback detection, type: {result}", category="content_type")
    return result

def fallback_detection(url: str, content: str) -> str:
    """Fallback detection logic."""
    if content and content.strip():
        return 'text'
    return 'html' if url and url.startswith(('http://', 'https://')) else 'unknown'

def preprocess_content(content: str, url: str) -> str:
    """Clean and preprocess content before sending to model with improved performance.
    
    Args:
        content: Content to preprocess
        url: URL of the content
        
    Returns:
        Preprocessed content string
        
    Examples:
        >>> content = "Copyright © 2024 Example Corp. All rights reserved.\nActual content here"
        >>> preprocess_content(content, "example.com")
        'Actual content here'
        
        >>> content = "Please enable JavaScript to continue.\nImportant content"
        >>> preprocess_content(content, "example.com")
        'Important content'
        
        >>> preprocess_content("", "example.com")
        ''
    """
    if not content:
        warning_highlight("Empty content provided", category="preprocessing")
        return ""

    info_highlight(f"Preprocessing content from {url}", category="preprocessing")
    info_highlight(f"Initial content length: {len(content)}", category="preprocessing")

    # Generate cache key
    cache_key = f"preprocess_content_{hash(f'{content}_{url}')}"
    
    # Check cache with TTL
    if cached_state := load_checkpoint(cache_key):
        cached_result = cached_state.get("result")
        if isinstance(cached_result, dict) and cached_result:
            timestamp = datetime.fromisoformat(cached_state.get("timestamp", ""))
            if (datetime.now() - timestamp).total_seconds() < 3600:  # 1 hour TTL
                return cached_result.get("content", "")

    # Compile regex patterns once
    boilerplate_patterns = [
        (re.compile(r'Copyright © \d{4}.*?reserved\.', re.IGNORECASE | re.DOTALL), ''),
        (re.compile(r'Terms of Service.*?Privacy Policy', re.IGNORECASE | re.DOTALL), ''),
        (re.compile(r'Please enable JavaScript.*?continue', re.IGNORECASE | re.DOTALL), '')
    ]
    
    # Apply boilerplate removal patterns
    for pattern, replacement in boilerplate_patterns:
        content = pattern.sub(replacement, content)

    # Remove redundant whitespace efficiently
    content = ' '.join(content.split())

    # Site-specific cleaning with compiled pattern
    domain = urlparse(url).netloc.lower()
    if 'iaeme.com' in domain:
        iaeme_pattern = re.compile(r'International Journal.*?Indexing', re.IGNORECASE | re.DOTALL)
        content = iaeme_pattern.sub('', content)

    # Truncate if too long
    if len(content) > MAX_CONTENT_LENGTH:
        warning_highlight(
            f"Content exceeds {MAX_CONTENT_LENGTH} characters, truncating",
            category="preprocessing"
        )
        content = f"{content[:MAX_CONTENT_LENGTH]}..."

    # Save to cache with TTL
    create_checkpoint(
        cache_key,
        {
            "result": {"content": content},
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        ttl=3600  # 1 hour TTL
    )
    
    info_highlight(f"Final content length: {len(content)}", category="preprocessing")
    return content

def estimate_tokens(text: str) -> int:
    """Estimate number of tokens in text using character ratio.
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated number of tokens
        
    Examples:
        >>> # Assuming TOKEN_CHAR_RATIO = 4.0
        >>> estimate_tokens("This is a test string")
        5  # ~20 characters / 4.0 = 5 tokens
        
        >>> estimate_tokens("")
        0
        
        >>> estimate_tokens("Short")
        1  # ~5 characters / 4.0 = 1 token
    """
    return int(len(text) / TOKEN_CHAR_RATIO) if text else 0

def should_skip_content(url: str) -> bool:
    """Check if content should be skipped based on URL patterns.
    
    Args:
        url: URL to check
        
    Returns:
        True if content should be skipped, False otherwise
        
    Examples:
        >>> should_skip_content("http://example.com/document.pdf")
        True
        
        >>> should_skip_content("http://example.com/page.html")
        False
        
        >>> should_skip_content("http://scribd.com/document")
        True  # Problematic site
        
        >>> should_skip_content("")
        True
    """
    try:
        decoded_url = urllib.parse.unquote(url).lower()
    except Exception as e:
        error_highlight(f"Error decoding URL: {str(e)}", category="validation")
        decoded_url = url.lower()

    # Enhanced PDF detection
    if any(p in decoded_url for p in ('.pdf', '%2Fpdf', '%3Fpdf')):
        info_highlight(f"Skipping PDF content: {url}", category="validation")
        return True

    # Add MIME-type pattern detection
    mime_patterns = [
        r'application/pdf',
        r'application/\w+?pdf',
        r'content-type:.*pdf'
    ]
    if any(re.match(p, decoded_url) for p in mime_patterns):
        info_highlight(f"Skipping PDF MIME-type pattern: {url}", category="validation")
        return True

    if not url:
        return True

    url_lower = url.lower()

    # Check for problematic file types
    for pattern in PROBLEMATIC_PATTERNS:
        if re.search(pattern, url_lower):
            info_highlight(f"Skipping content with pattern {pattern}: {url}", category="validation")
            return True

    # Check for problematic sites
    domain = urlparse(url).netloc.lower()
    for site in PROBLEMATIC_SITES:
        if site in domain:
            info_highlight(
                f"Skipping content from problematic site {site}: {url}",
                category="validation"
            )
            return True

    return False

def _merge_field(merged: Dict[str, Any], result: Dict[str, Any], field: str, operation: str, seen_items: set) -> None:
    """Helper function to merge a single field based on operation type.
    
    Args:
        merged: Dictionary containing the merged results
        result: Dictionary containing the current result to merge
        field: Field name to merge
        operation: Type of merge operation ('extend', 'update', 'max', 'min', 'avg')
        seen_items: Set of already seen items to prevent duplicates
        
    Examples:
        # Extend operation
        >>> merged = {'items': [1, 2]}
        >>> result = {'items': [3, 4]}
        >>> seen_items = set()
        >>> _merge_field(merged, result, 'items', 'extend', seen_items)
        >>> merged
        {'items': [1, 2, 3, 4]}

        # Update operation
        >>> merged = {'counts': {'a': 1}}
        >>> result = {'counts': {'b': 2}}
        >>> _merge_field(merged, result, 'counts', 'update', seen_items)
        >>> merged
        {'counts': {'a': 1, 'b': 2}}

        # Max operation
        >>> merged = {'score': 5}
        >>> result = {'score': 8}
        >>> _merge_field(merged, result, 'score', 'max', seen_items)
        >>> merged
        {'score': 8}
    """
    if field not in result:
        return

    value = result[field]
    if operation == "extend":
        merged[field] = merged.get(field, [])
        item_key = json.dumps(value, sort_keys=True)
        if item_key not in seen_items:
            merged[field].append(value)
            seen_items.add(item_key)
    elif operation == "update":
        merged[field] = merged.get(field, {})
        merged[field].update(value)
    elif operation in {"max", "min"}:
        if field not in merged or (operation == "max" and value > merged[field]) or (operation == "min" and value < merged[field]):
            merged[field] = value
    elif operation == "avg":
        merged[field] = merged.get(field, [])
        merged[field].append(value)

def merge_chunk_results(results: List[Dict[str, Any]], category: str, merge_strategy: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Merge results from multiple chunks into a single result based on merge strategy.
    
    Args:
        results: List of dictionaries containing results from each chunk
        category: Category of the extraction (e.g., 'research', 'summary')
        merge_strategy: Optional dictionary mapping fields to merge operations
                       ('extend', 'update', 'max', 'min', 'avg')
        
    Returns:
        Dictionary containing the merged results
        
    Examples:
        >>> results = [
        ...     {'findings': ['finding1'], 'score': 5},
        ...     {'findings': ['finding2'], 'score': 8}
        ... ]
        >>> merge_strategy = {'findings': 'extend', 'score': 'max'}
        >>> merge_chunk_results(results, 'research', merge_strategy)
        {
            'findings': ['finding1', 'finding2'],
            'score': 8
        }

        # With average operation
        >>> results = [
        ...     {'rating': 4.0},
        ...     {'rating': 6.0}
        ... ]
        >>> merge_strategy = {'rating': 'avg'}
        >>> merge_chunk_results(results, 'review', merge_strategy)
        {
            'rating': 5.0
        }
    """
    if not results:
        warning_highlight("No results to merge", category=category)
        return get_default_extraction_result(category)

    info_highlight(f"Merging {len(results)} chunk results", category=category)
    merged = get_default_extraction_result(category)
    merge_mappings = merge_strategy or get_category_merge_mapping(category)
    seen_items = set()

    for result in results:
        if "content" in result:
            try:
                result = safe_json_parse(result["content"], category)
            except Exception as e:
                error_highlight(f"Error parsing content: {str(e)}", category=category)
                continue
                
        for field, operation in merge_mappings.items():
            _merge_field(merged, result, field, operation, seen_items)
    
    # Calculate averages
    for field, operation in merge_mappings.items():
        if operation == "avg" and field in merged:
            merged[field] = sum(merged[field]) / len(results)

    info_highlight(f"Merged {len(results)} results successfully", category=category)
    return merged

def validate_content(content: str) -> bool:
    """Validate content before processing.
    
    Args:
        content: Content string to validate
        
    Returns:
        True if content is valid (non-empty string with minimum length),
        False otherwise
        
    Examples:
        >>> validate_content("This is valid content")
        True
        
        >>> validate_content("")  # Empty string
        False
        
        >>> validate_content("Hi")  # Too short
        False
        
        >>> validate_content(None)  # Invalid type
        False
    """
    if not content or not isinstance(content, str):
        warning_highlight("Invalid content type or empty content", category="validation")
        return False
        
    if len(content) < 10:  # Minimum content length
        warning_highlight(
            f"Content too short: {len(content)} characters",
            category="validation"
        )
        return False
        
    return True

__all__ = [
    "chunk_text",
    "preprocess_content",
    "estimate_tokens",
    "should_skip_content",
    "merge_chunk_results",
    "validate_content",
    "detect_content_type"
] 