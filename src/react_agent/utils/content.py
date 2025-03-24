"""Content processing utilities for the research agent.

This module provides utilities for processing and validating content,
including chunking, preprocessing, and content type detection.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import re
from urllib.parse import urlparse
import logging
from datetime import datetime, timezone
import json

from react_agent.utils.logging import get_logger, info_highlight, warning_highlight, error_highlight
from react_agent.utils.extraction import safe_json_parse
from react_agent.utils.defaults import ChunkConfig, get_default_extraction_result, get_category_merge_mapping
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

# Initialize logger
logger = get_logger(__name__)

# Initialize memory saver for caching
memory_saver = MemorySaver()

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

def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    use_large_chunks: bool = False,
    min_chunk_size: int = 100  # Minimum chunk size to avoid too small chunks
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
    """
    if not text or text.isspace():
        return []
    
    # Generate cache key
    cache_key = f"chunk_text_{hash(f'{text}_{chunk_size}_{overlap}_{use_large_chunks}_{min_chunk_size}')}"
    
    # Check cache with TTL
    if cached_state := memory_saver.get(RunnableConfig(configurable={"checkpoint_id": cache_key})):
        cached_result = cached_state.get("result")
        if isinstance(cached_result, dict) and cached_result:
            timestamp = datetime.fromisoformat(cached_state.get("timestamp", ""))
            if (datetime.now() - timestamp).total_seconds() < 3600:  # 1 hour TTL
                return cached_result.get("chunks", [])

    # Use appropriate chunk size and overlap based on configuration
    if use_large_chunks:
        chunk_size = chunk_size or ChunkConfig.LARGE_CHUNK_SIZE
        overlap = overlap or ChunkConfig.LARGE_OVERLAP
    else:
        chunk_size = chunk_size or ChunkConfig.DEFAULT_CHUNK_SIZE
        overlap = overlap or ChunkConfig.DEFAULT_OVERLAP
        
    # Ensure chunk size is positive and overlap is less than chunk size
    chunk_size = max(min_chunk_size, chunk_size)
    overlap = min(chunk_size - 1, max(0, overlap))
    
    # Pre-calculate text length and create list with estimated size
    text_length = len(text)
    estimated_chunks = (text_length // (chunk_size - overlap)) + 1
    chunks: List[str] = []
    
    start = 0
    while start < text_length:
        end = start + chunk_size
        
        if end >= text_length:
            # If the remaining text is too small, append it to the last chunk
            if chunks and len(text[start:]) < min_chunk_size:
                chunks[-1] = chunks[-1] + text[start:]
            else:
                chunks.append(text[start:])
            break
            
        # Find the last space before the chunk end
        last_space = text.rfind(' ', start, end)
        if last_space > start:
            end = last_space
            
        # Ensure we don't create chunks smaller than min_chunk_size
        if end - start < min_chunk_size and chunks:
            # Append to previous chunk instead of creating a new one
            chunks[-1] = chunks[-1] + text[start:end]
        else:
            chunks.append(text[start:end])
            
        start = end - overlap
    
    # Save to cache with TTL
    memory_saver.save(
        RunnableConfig(configurable={"checkpoint_id": cache_key}),
        {
            "result": {"chunks": chunks},
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        ttl=3600  # 1 hour TTL
    )
    
    return chunks

def detect_html(content: str) -> Optional[str]:
    """Detect if content is HTML."""
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
    """Detect if content is JSON."""
    if not content:
        return None
    content = content.strip()
    if (content.startswith('{') and content.endswith('}')) or (content.startswith('[') and content.endswith(']')):
        with contextlib.suppress(json.JSONDecodeError):
            json.loads(content)
            return 'json'
    return None

def detect_from_url_extension(url: str) -> Optional[str]:
    """Detect content type from URL file extension."""
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
    """Detect content type from URL and content using a modular approach."""
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
            return result

    return fallback_detection(url, content)

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
    """
    if not content:
        return ""

    logger.info(f"Preprocessing content from {url}")
    logger.debug(f"Initial content length: {len(content)}")

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
        warning_highlight(f"Content exceeds {MAX_CONTENT_LENGTH} characters, truncating")
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
    
    logger.debug(f"Final content length: {len(content)}")
    return content  # Correct final return

def estimate_tokens(text: str) -> int:
    """Estimate number of tokens in text.
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated number of tokens
    """
    return int(len(text) / TOKEN_CHAR_RATIO) if text else 0

def should_skip_content(url: str) -> bool:
    """Check if content should be skipped based on URL patterns."""
    try:
        decoded_url = urllib.parse.unquote(url).lower()
    except Exception:
        decoded_url = url.lower()

    # Enhanced PDF detection
    if any(p in decoded_url for p in ('.pdf', '%2Fpdf', '%3Fpdf')):
        info_highlight(f"Skipping PDF content: {url}")
        return True

    # Add MIME-type pattern detection
    mime_patterns = [
        r'application/pdf',
        r'application/\w+?pdf',
        r'content-type:.*pdf'
    ]
    if any(re.match(p, decoded_url) for p in mime_patterns):
        info_highlight(f"Skipping PDF MIME-type pattern: {url}")
        return True

    if not url:
        return True

    url_lower = url.lower()

    # Check for problematic file types
    for pattern in PROBLEMATIC_PATTERNS:
        if re.search(pattern, url_lower):
            info_highlight(f"Skipping content with pattern {pattern}: {url}")
            return True

    # Check for problematic sites
    domain = urlparse(url).netloc.lower()
    for site in PROBLEMATIC_SITES:
        if site in domain:
            info_highlight(f"Skipping content from problematic site {site}: {url}")
            return True

    return False

def merge_chunk_results(
    results: List[Dict[str, Any]], 
    category: str,
    merge_strategy: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Merge results from multiple chunks into a single result with enhanced merging strategies.
    
    Args:
        results: List of chunk results to merge
        category: Research category being processed
        merge_strategy: Optional custom merge strategy for specific fields
        
    Returns:
        Merged result dictionary
    """
    if not results:
        return get_default_extraction_result(category)

    logger.info(f"Merging {len(results)} chunk results for category {category}")

    # Initialize merged result
    merged: Dict[str, Any] = get_default_extraction_result(category)

    # Get merge mappings for the category or use provided strategy
    merge_mappings = merge_strategy or get_category_merge_mapping(category)

    # Track unique items to avoid duplicates
    seen_items = set()

def merge_chunk_results(results: List[Dict[str, Any]], category: str) -> Dict[str, Any]:
    """Merge results from multiple chunks into a single result.
    
    Args:
        results: List of chunk results to merge
        category: Research category being processed
        
    Returns:
        Merged result dictionary
    """
    if not results:
        return get_default_extraction_result(category)

    logger.info(f"Merging {len(results)} chunk results for category {category}")

    # Initialize merged result
    merged = defaultdict(list, get_default_extraction_result(category))

    # Get category-specific merge mappings
    merge_mappings = get_category_merge_mappings().get(category, {"extracted_facts": "extend"})

    # Process each result
    for result in results:
        # Handle the new response format with content field
        if "content" in result:
            try:
                parsed_content = safe_json_parse(result["content"], category)
                result = parsed_content
            except Exception as e:
                error_highlight(f"Error parsing content in merge_chunk_results: {str(e)}")
                continue

        for field, operation in merge_mappings.items():
            if field in result:
                if operation == "extend":
                    if field not in merged:
                        merged[field] = []
                    # Add only unique items
                    for item in result[field]:
                        item_key = json.dumps(item, sort_keys=True)
                        if item_key not in seen_items:
                            merged[field].append(item)
                            seen_items.add(item_key)
                elif operation == "update":
                    if field not in merged:
                        merged[field] = {}
                    merged[field].update(result[field])
                elif operation == "max":
                    # Take the maximum value for numeric fields
                    if field not in merged or result[field] > merged[field]:
                        merged[field] = result[field]
                elif operation == "min":
                    # Take the minimum value for numeric fields
                    if field not in merged or result[field] < merged[field]:
                        merged[field] = result[field]
                elif operation == "avg":
                    # Calculate average for numeric fields
                    if field not in merged:
                        merged[field] = []
                    merged[field].append(result[field])
                    if len(merged[field]) == len(results):
                        merged[field] = sum(merged[field]) / len(merged[field])

    return merged


def validate_content(content: str) -> bool:
    """Validate content before processing.
    
    Args:
        content: Content to validate
        
    Returns:
        True if content is valid, False otherwise
    """
    if not content or not isinstance(content, str):
        warning_highlight("Invalid content type or empty content")
        return False
        
    if len(content) < 10:  # Minimum content length
        warning_highlight(f"Content too short: {len(content)} characters")
        return False
        
    return True

def detect_content_type(url: str, content: str) -> str:
    """Detect content type from URL and content.
    
    Args:
        url: URL of the content
        content: Content to analyze
        
    Returns:
        Detected content type
    """
    if not url:
        return "unknown"
        
    url_lower = url.lower()
    
    if url_lower.endswith('.pdf'):
        return 'pdf'
    elif url_lower.endswith(('.doc', '.docx')):
        return 'doc'
    elif url_lower.endswith(('.xls', '.xlsx')):
        return 'excel'
    elif url_lower.endswith(('.ppt', '.pptx')):
        return 'presentation'
    elif url_lower.endswith(('.txt', '.md', '.rst')):
        return 'text'
    elif url_lower.endswith(('.html', '.htm')):
        return 'html'
    elif url_lower.endswith(('.json', '.xml')):
        return 'data'
    else:
        return 'unknown'

__all__ = [
    "chunk_text",
    "preprocess_content",
    "estimate_tokens",
    "should_skip_content",
    "merge_chunk_results",
    "validate_content",
    "detect_content_type"
] 