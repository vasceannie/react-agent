"""Default values and configurations for the research agent.

This module consolidates all default values, configurations, and common structures
used across the research agent to maintain consistency and reduce duplication.
"""

from dataclasses import dataclass
from typing import Any, Dict, List


# Default chunking configurations
@dataclass
class ChunkConfig:
    """Configuration for text chunking operations."""
    DEFAULT_CHUNK_SIZE: int = 4000
    DEFAULT_OVERLAP: int = 500
    LARGE_CHUNK_SIZE: int = 40000
    LARGE_OVERLAP: int = 5000


# Constants for content processing
DEFAULT_CHUNK_SIZE: int = 40000
DEFAULT_OVERLAP: int = 5000
MAX_CONTENT_LENGTH: int = 100000
TOKEN_CHAR_RATIO: float = 4.0

# Problematic content patterns to skip certain file types
PROBLEMATIC_PATTERNS: List[str] = [
    r'\.zip$',
    r'\.rar$',
    r'\.exe$',
    r'\.dmg$',
    r'\.iso$',
    r'\.tar$',
    r'\.gz$'
]

# Known problematic sites to avoid
PROBLEMATIC_SITES: List[str] = [
    'iaeme.com',
    'scribd.com',
    'slideshare.net',
    'academia.edu'
]

# Document type mappings
DOCUMENT_TYPE_MAPPING: Dict[str, str] = {
    '.pdf': 'pdf',
    '.doc': 'doc',
    '.docx': 'doc',
    '.xls': 'excel',
    '.xlsx': 'excel',
    '.ppt': 'presentation',
    '.pptx': 'presentation',
    '.odt': 'document',
    '.rtf': 'document'
}


# Default extraction result structure
DEFAULT_EXTRACTION_RESULTS = {
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

# Category-specific merge mappings
CATEGORY_MERGE_MAPPINGS = {
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


def get_default_extraction_result(category: str) -> Dict[str, Any]:
    """Get a default empty extraction result when parsing fails.
    
    Args:
        category: Research category
        
    Returns:
        Default empty result dictionary
    """
    return DEFAULT_EXTRACTION_RESULTS.get(category, {"extracted_facts": [], "relevance_score": 0.0})


def get_category_merge_mapping(category: str) -> Dict[str, str]:
    """Get the merge mapping for a specific category.
    
    Args:
        category: Research category
        
    Returns:
        Dictionary mapping field names to merge operations
    """
    return CATEGORY_MERGE_MAPPINGS.get(category, {})


# URL path patterns for content type detection
HTML_PATH_PATTERNS = (
    "/wiki/",
    "/articles/",
    "/blog/",
    "/news/",
    "/docs/",
    "/help/",
    "/support/",
    "/pages/",
    "/product/",
    "/service/",
    "/consumers/",
    "/detail/",
    "/view/",
    "/content/",
)

# Data feed patterns (JSON/XML/CSV)
DATA_PATH_PATTERNS = (
    "/api/",
    "/data/",
    "/feeds/",            # plural version
    "/feed/",
    "/export/",
    "/export-data/",
    "/catalog.",
    "/product-feed",
    "/pricing-data",
    ".json",
    ".xml",
    ".csv"
)

# Procurement/specific HTML content patterns
PROCUREMENT_HTML_PATTERNS = (
    "/procurement/", 
    "/procurements/",     # plural
    "/sourcing/",
    "/tender/",
    "/tender-notice/",    # additional synonym
    "/rfp/",
    "/rfq/",
    "/rfx/",
    "/bid-request/",      # capturing bid-related pages
    "/bid-invitation/",
    "/bidding/",
    "/contracts/",
    "/contract-management/",  # extended contract keyword
    "/supplier/",
    "/suppliers/",        # plural
    "/vendor/",
    "/vendors/",          # plural
    "/purchase-order/",
    "/category-management/",
    "/strategic-sourcing/",
    "/purchasing/",
    "/reverse-auction/",
    "/scorecard/",
    "/supplier-portal/",
    "/vendor-management/",
    "/e-procurement/",    # electronic procurement systems
    "/quotation/",        # for RFQs and pricing inquiries
    "/quote/"
)

# Marketplace/catalogue indicators
MARKETPLACE_PATTERNS = (
    "/product/",
    "/sku/",
    "/listing/",
    "/catalog/",
    "/inventory/",
    "/stock/",
    "/marketplace/",
    "/b2b/",
    "/bulk-pricing/",
    "/moq/",              # minimum order quantity
    "/lead-time/",
    "/vendor-central/",
    "/shop/",             # capturing shopping portals
    "/store/",
    "/deals/",
    "/offers/"
)

# Document file extensions and patterns
DOCUMENT_PATTERNS = (
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".rtf",              # rich text format
    ".odt",              # OpenDocument text
    ".ods"               # OpenDocument spreadsheet
)

# Report-specific patterns (research, studies, whitepapers)
REPORT_PATTERNS = (
    "/report/",
    "/analysis/",
    "/study/",
    "/whitepaper/",
    "/benchmark/",
    "/survey/",
    "/insights/",        # additional analysis perspective
    "/case-study/",
    "/evaluation/",
    "/audit/",
    "/dossier/"          # detailed collection of documents
)

# Export all defaults
__all__ = [
    "ChunkConfig",
    "DEFAULT_EXTRACTION_RESULTS",
    "CATEGORY_MERGE_MAPPINGS",
    "get_default_extraction_result",
    "get_category_merge_mapping",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_OVERLAP",
    "MAX_CONTENT_LENGTH",
    "TOKEN_CHAR_RATIO",
    "PROBLEMATIC_PATTERNS",
    "PROBLEMATIC_SITES",
    "DOCUMENT_TYPE_MAPPING",
    "HTML_PATH_PATTERNS",
    "DATA_PATH_PATTERNS",
    "PROCUREMENT_HTML_PATTERNS",
    "MARKETPLACE_PATTERNS",
    "DOCUMENT_PATTERNS",
    "REPORT_PATTERNS"
]