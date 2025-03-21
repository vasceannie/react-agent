"""Enhanced query optimization to improve search relevance.

This enhances the query optimization to include more procurement-specific terms
and domain-specific vocabularies to increase search precision.
"""

from typing import Dict, List, Set, Optional
import re

# Domain-specific keyword repositories
PROCUREMENT_TERMS = {
    "sourcing": ["strategic sourcing", "supplier selection", "supplier qualification", "vendor selection"],
    "contracts": ["contract terms", "contract management", "agreement", "obligations", "SLAs", "KPIs"],
    "pricing": ["volume discounts", "rebates", "bulk discounts", "pricing models", "cost-plus", "fixed price"],
    "rfp": ["request for proposal", "request for information", "request for quote", "bid", "tender"],
    "procurement": ["procurement strategy", "procurement process", "buying", "purchasing"],
    "payment": ["invoicing", "payment terms", "purchase orders", "net-30", "net-60"],
    "suppliers": ["vendors", "distributors", "manufacturers", "providers", "supply base"],
    "strategy": ["category strategy", "category management", "spend analysis", "cost reduction"],
    "risk": ["risk management", "risk mitigation", "compliance", "qualifications", "certifications"],
    "process": ["auction", "reverse auction", "e-procurement", "p2p", "procure to pay"]
}

# Industry vertical specializations - can be expanded as needed
INDUSTRY_VERTICALS = {
    "education": ["university", "college", "campus", "academic", "educational"],
    "healthcare": ["hospital", "clinic", "medical", "patient care", "healthcare"],
    "manufacturing": ["factory", "industrial", "production", "assembly", "plant"],
    "government": ["public sector", "government agency", "municipal", "federal", "state"],
    "retail": ["retail operations", "store", "outlet", "retail chain", "merchandising"],
    "utilities": ["energy", "water", "electricity", "gas", "utility provider"]
}

# Maintenance categories
MAINTENANCE_CATEGORIES = {
    "preventive": ["preventive maintenance", "scheduled maintenance", "routine service"],
    "corrective": ["corrective maintenance", "repair", "fix", "troubleshooting"],
    "predictive": ["predictive maintenance", "condition monitoring", "predictive analytics"],
    "supplies": ["consumables", "disposables", "tools", "equipment", "spare parts"]
}

def detect_vertical(query: str) -> str:
    """Detect the industry vertical from the query."""
    query_lower = query.lower()

    return next(
        (
            vertical
            for vertical, keywords in INDUSTRY_VERTICALS.items()
            if any(keyword in query_lower for keyword in keywords)
        ),
        "general",
    )

def expand_acronyms(query: str) -> str:
    """Expand common industry acronyms in the query."""
    acronyms = {
        "mro": "maintenance repair operations",
        "rfp": "request for proposal",
        "rfq": "request for quote",
        "rfi": "request for information",
        "eam": "enterprise asset management",
        "cmms": "computerized maintenance management system",
        "kpi": "key performance indicator",
        "sla": "service level agreement",
        "tcoo": "total cost of ownership",
        "p2p": "procure to pay"
    }
    
    words = query.split()
    for i, word in enumerate(words):
        word_lower = word.lower().strip(",.;:()[]{}\"'")
        if word_lower in acronyms:
            # Replace acronym with expansion while preserving original casing and punctuation
            prefix = ""
            suffix = ""
            if not word.isalnum():
                prefix = word[:len(word) - len(word.lstrip(",.;:()[]{}\"'"))]
                suffix = word[len(word.rstrip(",.;:()[]{}\"'")):]
            words[i] = prefix + acronyms[word_lower] + " (" + word.strip(",.;:()[]{}\"'") + ")" + suffix
    
    return " ".join(words)

def optimize_query(
    original_query: str, 
    category: str, 
    vertical: Optional[str] = None,
    include_all_keywords: bool = False
) -> str:
    """Create optimized queries for specific research categories with enhanced domain-specific terms.
    
    Args:
        original_query: The original search query
        category: Research category to optimize for
        vertical: Industry vertical context (auto-detected if None)
        include_all_keywords: Whether to include all keywords for comprehensive searches
        
    Returns:
        Optimized search query
    """
    # Clean the original query
    original_query = original_query.split("Additional context:")[0].strip()
    
    # Expand acronyms for better search results
    expanded_query = expand_acronyms(original_query)
    
    # Detect vertical if not provided
    if vertical is None:
        vertical = detect_vertical(original_query)
    
    # Get vertical-specific terms
    vertical_terms = INDUSTRY_VERTICALS.get(vertical, [""])
    
    # Define enhanced category-specific query templates with keyword banks
    query_templates = {
        "market_dynamics": {
            "template": "{query} market trends analysis {vertical} {procurement_terms}",
            "keyword_groups": ["contracts", "procurement", "strategy"]
        },
        "provider_landscape": {
            "template": "{query} {vertical} {procurement_terms}",
            "keyword_groups": ["suppliers", "rfp", "strategy"]
        },
        "technical_requirements": {
            "template": "{query} technical specifications {vertical} {procurement_terms}",
            "keyword_groups": ["rfp", "procurement", "risk"]
        },
        "regulatory_landscape": {
            "template": "{query} regulations compliance {vertical} {procurement_terms}",
            "keyword_groups": ["contracts", "risk", "process"]
        },
        "cost_considerations": {
            "template": "{query} pricing cost budget {vertical} {procurement_terms}",
            "keyword_groups": ["pricing", "payment", "strategy"]
        },
        "best_practices": {
            "template": "{query} best practices case studies {vertical} {procurement_terms}",
            "keyword_groups": ["strategy", "risk", "process"]
        },
        "implementation_factors": {
            "template": "{query} implementation factors {vertical} {procurement_terms}",
            "keyword_groups": ["procurement", "suppliers", "risk"]
        }
    }
    
    # Get template for this category
    if category not in query_templates:
        return expanded_query
    
    template = query_templates[category]["template"]
    
    # Add maintenance-specific terms if relevant
    maintenance_terms = []
    if "maintenance" in original_query.lower() or "repair" in original_query.lower():
        for terms in MAINTENANCE_CATEGORIES.values():
            maintenance_terms.extend(terms[:2])  # Add top 2 terms from each maintenance category
    
    # Get procurement terms for this category
    procurement_terms = []
    if include_all_keywords:
        # Include all keyword groups for comprehensive searches
        for term_group in PROCUREMENT_TERMS.values():
            procurement_terms.extend(term_group[:2])  # Top 2 terms from each group
    else:
        # Include only category-specific keyword groups
        keyword_groups = query_templates[category]["keyword_groups"]
        for group in keyword_groups:
            if group in PROCUREMENT_TERMS:
                procurement_terms.extend(PROCUREMENT_TERMS[group][:2])  # Top 2 terms from each group
    
    # Format the template
    optimized_query = template.format(
        query=expanded_query,
        vertical=" ".join(vertical_terms[:2]),  # Top 2 vertical terms
        procurement_terms=" ".join(procurement_terms + maintenance_terms)
    )
    
    # Ensure the query isn't too long for search engines (typically ~150-200 chars)
    if len(optimized_query) > 180:
        # Prioritize original query with key terms
        terms = " ".join(procurement_terms[:4])  # Limit to 4 procurement terms when query is long
        optimized_query = f"{expanded_query} {vertical_terms[0]} {terms}"
    
    return optimized_query