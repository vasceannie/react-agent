"""Enhanced query templates for search optimization.

This module provides enhanced query templates for different research categories,
with fallback options to improve search results for problematic categories.
"""

from typing import Dict, Final

# Enhanced query templates for problematic categories
ENHANCED_CATEGORY_TEMPLATES: Final[Dict[str, Dict[str, str]]] = {
    "best_practices": {
        "primary": "{primary_terms} best practices specific examples",
        "fallback_1": "{primary_terms} proven guidelines successful approaches",
        "fallback_2": "{primary_terms} implementation strategies case studies"
    },
    "regulatory_landscape": {
        "primary": "{primary_terms} specific regulations compliance requirements",
        "fallback_1": "{primary_terms} recent laws regulatory changes",
        "fallback_2": "{primary_terms} compliance framework legal standards"
    },
    "implementation_factors": {
        "primary": "{primary_terms} implementation success factors examples",
        "fallback_1": "{primary_terms} practical implementation strategies",
        "fallback_2": "{primary_terms} implementation challenges solutions"
    },
    "provider_landscape": {
        "primary": "{primary_terms} top vendors market leaders",
        "fallback_1": "{primary_terms} provider comparison competitive analysis",
        "fallback_2": "{primary_terms} emerging providers market share"
    },
    "technical_requirements": {
        "primary": "{primary_terms} specific technical requirements specifications",
        "fallback_1": "{primary_terms} technical standards compatibility",
        "fallback_2": "{primary_terms} technical infrastructure requirements"
    },
    "market_dynamics": {
        "primary": "{primary_terms} current market trends data statistics",
        "fallback_1": "{primary_terms} market analysis growth projections",
        "fallback_2": "{primary_terms} industry developments market research"
    },
    "cost_considerations": {
        "primary": "{primary_terms} cost analysis pricing models",
        "fallback_1": "{primary_terms} budget considerations roi examples",
        "fallback_2": "{primary_terms} cost comparison financial impact"
    }
}