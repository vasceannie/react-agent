from collections import defaultdict
import contextlib
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
import json
from urllib.parse import urlparse
import re
import urllib
import logging
from .logging import warning_highlight, get_logger

"""Content processing utilities for handling large documents and context limits.

This module provides utilities for processing and managing content before sending it to LLMs,
including chunking, preprocessing, and content validation.
"""

from react_agent.utils.logging import (
    error_highlight,
    get_logger,
    info_highlight,
    warning_highlight,
)

# Initialize logger
logger = get_logger(__name__)

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

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> List[str]:
    """Split text into overlapping chunks of specified size.
    
    Args:
        text: Text to split into chunks
        chunk_size: Size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
        
    logger.info(f"Chunking text of length {len(text)}")
    chunks: List[str] = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
            
        # Find the last period or newline before the chunk end
        last_period = text.find('.', start, end)
        last_newline = text.find('\n', start, end)
        split_point = max(last_period, last_newline)
        
        if split_point > start:
            end = split_point + 1
            
        chunks.append(text[start:end])
        start = end - overlap
        
    logger.info(f"Created {len(chunks)} chunks")
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
    """Clean and preprocess content before sending to model."""
    content_type = detect_content_type(url, content)
    
    # Modified to attempt extraction for document-like URLs even if type detection fails
    if content_type != 'html':
        warning_highlight(f"Non-HTML content detected ({content_type}): {url} - attempting extraction")
        # Don't return empty - proceed with basic cleaning
        
    # Keep existing cleaning logic but remove early return
    content = re.sub(r'Copyright © \d{4}.*?reserved\.', '', content, flags=re.IGNORECASE | re.DOTALL)
    content = re.sub(r'Terms of Service.*?Privacy Policy', '', content, flags=re.IGNORECASE | re.DOTALL)
    
    # Add PDF content detection fallback
    if '%PDF' in content[:4]:
        warning_highlight(f"PDF content detected: {url}")
        return ""  # Actual PDF handling would require text extraction

    # Keep existing processing AFTER PDF check
    content = re.sub(r'Please enable JavaScript.*?continue', '', content, flags=re.IGNORECASE | re.DOTALL)
    content = re.sub(r'\s+', ' ', content)
    content = content.strip()
    
    # Site-specific cleaning
    domain = urlparse(url).netloc.lower()
    if 'iaeme.com' in domain:
        content = re.sub(r'International Journal.*?Indexing', '', content, flags=re.IGNORECASE | re.DOTALL)

    # Truncate if too long
    if len(content) > MAX_CONTENT_LENGTH:
        warning_highlight(f"Content exceeds {MAX_CONTENT_LENGTH} characters, truncating")
        content = f"{content[:MAX_CONTENT_LENGTH]}..."

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

def dict_to_hashable(d):
    """Convert a dictionary or list to a hashable representation for deduplication."""
    if isinstance(d, dict):
        return tuple(sorted((k, dict_to_hashable(v)) for k, v in d.items()))
    elif isinstance(d, list):
        return tuple(dict_to_hashable(x) for x in d)
    else:
        return d

def extend_with_deduplication(existing_list: List, new_items: List) -> None:
    """Extend a list with new items, avoiding duplicates."""
    existing_hashables = {dict_to_hashable(item) for item in existing_list}

    for item in new_items:
        item_hashable = dict_to_hashable(item)
        if item_hashable not in existing_hashables:
            existing_list.append(item)
            existing_hashables.add(item_hashable)

def update_dict_if_empty(target_dict: Dict, source_dict: Dict) -> None:
    """Update target dictionary with values from source dictionary if target values are empty."""
    for key, value in source_dict.items():
        if value and key in target_dict and not target_dict[key]:
            target_dict[key] = value

def get_category_merge_mappings() -> Dict[str, Dict[str, str]]:
    """Return the merge mappings for different categories."""
    return {
        "market_dynamics": {
            "extracted_facts": "extend",
            "market_metrics": "update"
        },
        "provider_landscape": {
            "extracted_vendors": "extend",
            "vendor_relationships": "extend"
        },
        "technical_requirements": {
            "extracted_requirements": "extend",
            "standards": "extend"
        },
        "regulatory_landscape": {
            "extracted_regulations": "extend",
            "compliance_requirements": "extend"
        },
        "cost_considerations": {
            "extracted_costs": "extend",
            "pricing_models": "extend"
        },
        "best_practices": {
            "extracted_practices": "extend",
            "methodologies": "extend"
        },
        "implementation_factors": {
            "extracted_factors": "extend",
            "challenges": "extend"
        }
    }

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
        if not isinstance(result, dict):
            warning_highlight(f"Invalid chunk result format: {type(result)}")
            continue

        # Apply merge operations based on field type
        for field, operation in merge_mappings.items():
            if field not in result:
                continue

            if operation == "extend":
                extend_with_deduplication(merged[field], result[field])
            elif operation == "update" and isinstance(result[field], dict):
                update_dict_if_empty(merged[field], result[field])

    if relevance_scores := [
        r.get("relevance_score", 0.0) for r in results if isinstance(r, dict)
    ]:
        merged["relevance_score"] = sum(relevance_scores) / len(relevance_scores)

    logger.info(f"Merged result contains {len(merged.get('extracted_facts', []))} facts")
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

def get_default_extraction_result(category: str) -> Dict[str, Any]:
    """Get a default empty extraction result when parsing fails."""
    # Add schema validation to ensure consistent structure
    defaults = {
        "market_dynamics": {
            "extracted_facts": [],
            "market_metrics": {
                "market_size": None,
                "growth_rate": None,
                "forecast_period": None
            },
            "relevance_score": 0.0
        },
        "provider_landscape": {
            "extracted_vendors": [],
            "vendor_relationships": [],
            "relevance_score": 0.0
        },
        "technical_requirements": {
            "extracted_requirements": [],
            "standards": [],
            "relevance_score": 0.0
        },
        "regulatory_landscape": {
            "extracted_regulations": [],
            "compliance_requirements": [],
            "relevance_score": 0.0
        },
        "cost_considerations": {
            "extracted_costs": [],
            "pricing_models": [],
            "relevance_score": 0.0
        },
        "best_practices": {
            "extracted_practices": [],
            "methodologies": [],
            "relevance_score": 0.0
        },
        "implementation_factors": {
            "extracted_factors": [],
            "challenges": [],
            "relevance_score": 0.0,
            "source_validation": {
                "is_pdf": False,
                "is_trusted_domain": False
            }
        }
    }
    
    return defaults.get(category, {"extracted_facts": [], "relevance_score": 0.0})

__all__ = [
    "chunk_text",
    "preprocess_content",
    "estimate_tokens",
    "should_skip_content",
    "merge_chunk_results",
    "validate_content",
    "detect_content_type",
    "get_default_extraction_result"
]
