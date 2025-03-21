"""Content processing utilities for handling large documents and context limits.

This module provides utilities for processing and managing content before sending it to LLMs,
including chunking, preprocessing, and content validation.
"""

from typing import List, Dict, Any, Optional, Union
import re
from urllib.parse import urlparse
import logging

from react_agent.utils.logging import get_logger, info_highlight, warning_highlight, error_highlight

# Initialize logger
logger = get_logger(__name__)

# Constants
DEFAULT_CHUNK_SIZE: int = 40000
DEFAULT_OVERLAP: int = 5000
MAX_CONTENT_LENGTH: int = 100000
TOKEN_CHAR_RATIO: float = 4.0

# Problematic content patterns
PROBLEMATIC_PATTERNS: List[str] = [
    r'\.pdf$',
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
        last_period = text.rfind('.', start, end)
        last_newline = text.rfind('\n', start, end)
        split_point = max(last_period, last_newline)
        
        if split_point > start:
            end = split_point + 1
            
        chunks.append(text[start:end])
        start = end - overlap
        
    logger.info(f"Created {len(chunks)} chunks")
    return chunks

def preprocess_content(content: str, url: str) -> str:
    """Clean and preprocess content before sending to model.
    
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

    # Remove common boilerplate
    content = re.sub(r'Copyright © \d{4}.*?reserved\.', '', content, flags=re.IGNORECASE | re.DOTALL)
    content = re.sub(r'Terms of Service.*?Privacy Policy', '', content, flags=re.IGNORECASE | re.DOTALL)
    content = re.sub(r'Please enable JavaScript.*?continue', '', content, flags=re.IGNORECASE | re.DOTALL)

    # Remove redundant whitespace
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
    return content

def estimate_tokens(text: str) -> int:
    """Estimate number of tokens in text.
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated number of tokens
    """
    return int(len(text) / TOKEN_CHAR_RATIO) if text else 0

def should_skip_content(url: str) -> bool:
    """Check if content should be skipped based on URL patterns.
    
    Args:
        url: URL to check
        
    Returns:
        True if content should be skipped, False otherwise
    """
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
    merged: Dict[str, Any] = get_default_extraction_result(category)

    # Define category-specific merge mappings
    merge_mappings = {
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

    # Merge results based on category mappings
    for result in results:
        if not isinstance(result, dict):
            warning_highlight(f"Invalid chunk result format: {type(result)}")
            continue

        mappings = merge_mappings.get(category, {"extracted_facts": "extend"})

        for field, operation in mappings.items():
            if field not in result:
                continue

            if operation == "extend":
                merged[field].extend(result[field])
            elif operation == "update" and isinstance(result[field], dict):
                for key, value in result[field].items():
                    if value and not merged[field][key]:
                        merged[field][key] = value

    if relevance_scores := [r.get("relevance_score", 0.0) for r in results]:
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

def get_default_extraction_result(category: str) -> Dict[str, Any]:
    """Get a default empty extraction result when parsing fails.
    
    Args:
        category: Research category
        
    Returns:
        Default empty result dictionary
    """
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
            "relevance_score": 0.0
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