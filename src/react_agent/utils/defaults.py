"""Default values and configurations for the research agent.

This module consolidates all default values, configurations, and common structures
used across the research agent to maintain consistency and reduce duplication.
"""

from dataclasses import dataclass
from typing import Any, Dict


# Default chunking configurations
@dataclass
class ChunkConfig:
    """Configuration for text chunking operations."""
    DEFAULT_CHUNK_SIZE: int = 4000
    DEFAULT_OVERLAP: int = 500
    LARGE_CHUNK_SIZE: int = 40000
    LARGE_OVERLAP: int = 5000


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


# Export all defaults
__all__ = [
    "ChunkConfig",
    "DEFAULT_EXTRACTION_RESULTS",
    "CATEGORY_MERGE_MAPPINGS",
    "get_default_extraction_result",
    "get_category_merge_mapping"
] 