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
    """Create optimized queries for specific research categories with enhanced domain-specific terms."""
    # Clean the original query
    original_query = original_query.split("Additional context:")[0].strip()
    
    # Expanded acronyms for better search results
    expanded_query = expand_acronyms(original_query)
    
    # Detect vertical if not provided
    if vertical is None:
        vertical = detect_vertical(original_query)
    
    # Get vertical-specific terms
    vertical_terms = INDUSTRY_VERTICALS.get(vertical, [""])
    
    # Define enhanced category-specific query templates with keyword banks
    query_templates = {
        "market_dynamics": {
            "template": "{primary_terms} market trends",
            "keyword_groups": ["contracts", "procurement", "strategy"]
        },
        "provider_landscape": {
            "template": "{primary_terms} vendors suppliers",
            "keyword_groups": ["suppliers", "rfp", "strategy"]
        },
        "technical_requirements": {
            "template": "{primary_terms} technical specifications",
            "keyword_groups": ["rfp", "procurement", "risk"]
        },
        "regulatory_landscape": {
            "template": "{primary_terms} regulations compliance",
            "keyword_groups": ["contracts", "risk", "process"]
        },
        "cost_considerations": {
            "template": "{primary_terms} pricing cost budget",
            "keyword_groups": ["pricing", "payment", "strategy"]
        },
        "best_practices": {
            "template": "{primary_terms} best practices",
            "keyword_groups": ["strategy", "risk", "process"]
        },
        "implementation_factors": {
            "template": "{primary_terms} implementation factors",
            "keyword_groups": ["procurement", "suppliers", "risk"]
        }
    }
    
    # Extract primary terms from the query (filter out common words)
    words = expanded_query.split()
    stop_words = ["help", "me", "research", "find", "information", "about", "on", "for", 
                  "the", "and", "or", "in", "to", "with", "by", "is", "are"]
    
    primary_terms = []
    for word in words:
        if word.lower() not in stop_words and len(word) > 3:
            primary_terms.append(word)
            # Limit to first 3-4 meaningful terms
            if len(primary_terms) >= 4:
                break
    
    # If no primary terms found, use the whole query up to a limit
    if not primary_terms:
        primary_terms = words[:3]
    
    # Get template for this category
    if category not in query_templates:
        return " ".join(primary_terms)
    
    template = query_templates[category]["template"]
    
    # Get procurement terms for this category (but use fewer to keep query simple)
    procurement_terms = []
    if include_all_keywords:
        # Include more keywords for comprehensive searches but still limit
        for group, terms in PROCUREMENT_TERMS.items():
            if group in query_templates[category]["keyword_groups"]:
                procurement_terms.append(terms[0])  # Just top 1 term from each group
    else:
        # Include only minimal keyword groups
        keyword_groups = query_templates[category]["keyword_groups"]
        if keyword_groups:
            top_group = keyword_groups[0]
            if top_group in PROCUREMENT_TERMS:
                procurement_terms.append(PROCUREMENT_TERMS[top_group][0])  # Just top term
    
    # Format the template with just essential terms
    primary_terms_str = " ".join(primary_terms)
    
    optimized_query = template.format(
        primary_terms=primary_terms_str
    )
    
    # Add at most one procurement term if we need to for context
    if procurement_terms:
        optimized_query += " " + procurement_terms[0]
        
    # Add at most one vertical term if needed
    if vertical != "general" and vertical_terms:
        optimized_query += " " + vertical_terms[0]
    
    # Ensure the query isn't too long for search engines
    if len(optimized_query) > 100:
        optimized_query = optimized_query[:100].rsplit(' ', 1)[0]
    
    return optimized_query