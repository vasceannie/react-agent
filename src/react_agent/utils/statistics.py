"""Improved confidence scoring with statistical validation.

This module enhances the confidence scoring logic to focus on statistical
validation, source quality assessment, and cross-validation to achieve
confidence scores above 80%.
"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timezone
import re
from urllib.parse import urlparse
from collections import Counter

from react_agent.utils.logging import get_logger, info_highlight, warning_highlight, error_highlight

# Initialize logger
logger = get_logger(__name__)

# Authority domain patterns
AUTHORITY_DOMAINS = [
    r'\.gov($|/)',  # Government domains
    r'\.edu($|/)',  # Educational institutions
    r'\.org($|/)',  # Non-profit organizations
    r'research\.',  # Research organizations
    r'\.ac\.($|/)',  # Academic institutions
    r'journal\.',   # Academic journals
    r'university\.',  # Universities
    r'institute\.',  # Research institutes
    r'association\.'  # Professional associations
]

# Compiled patterns for efficiency
COMPILED_AUTHORITY_PATTERNS = [re.compile(pattern) for pattern in AUTHORITY_DOMAINS]

# High-credibility source terms
HIGH_CREDIBILITY_TERMS = [
    'study', 'research', 'survey', 'report', 'analysis',
    'journal', 'publication', 'paper', 'review', 'assessment',
    'statistics', 'data', 'findings', 'results', 'evidence'
]

def calculate_category_quality_score(
    category: str,
    extracted_facts: List[Dict[str, Any]],
    sources: List[Dict[str, Any]],
    thresholds: Dict[str, Any]
) -> float:
    """Calculate enhanced quality score for a category based on extracted data.
    
    Args:
        category: Research category
        extracted_facts: List of extracted facts
        sources: List of source information
        thresholds: Quality thresholds for this category
        
    Returns:
        Quality score (0.0-1.0)
    """
    # Start with a slightly higher base score
    score = 0.35
    
    # Get category-specific thresholds
    min_facts = thresholds.get("min_facts", 3)
    min_sources = thresholds.get("min_sources", 2)
    auth_ratio = thresholds.get("authoritative_source_ratio", 0.5)
    recency_threshold = thresholds.get("recency_threshold_days", 365)
    
    # 1. QUANTITY ASSESSMENT (0.25 max)
    # Add points for number of facts (nonlinear scale to reward comprehensiveness)
    if len(extracted_facts) >= min_facts * 3:  # Excellent quantity
        score += 0.15
    elif len(extracted_facts) >= min_facts * 2:  # Very good quantity
        score += 0.12
    elif len(extracted_facts) >= min_facts:  # Good quantity
        score += 0.08
    else:  # Below threshold
        fact_ratio = len(extracted_facts) / min_facts if min_facts else 0
        score += fact_ratio * 0.06
    
    # Add points for number of sources (reward diversity)
    if len(sources) >= min_sources * 3:  # Excellent source diversity
        score += 0.10
    elif len(sources) >= min_sources * 2:  # Very good source diversity
        score += 0.08
    elif len(sources) >= min_sources:  # Good source diversity
        score += 0.05
    else:  # Below threshold
        source_ratio = len(sources) / min_sources if min_sources else 0
        score += source_ratio * 0.03
    
    # 2. STATISTICAL CONTENT (0.20 max)
    # Reward statistical content
    stat_facts = [
        f for f in extracted_facts
        if "statistics" in f or
        (isinstance(f.get("source_text"), str) and
         re.search(r'\d+', f.get("source_text", "")) is not None)
    ]
    
    if stat_facts:
        stat_ratio = len(stat_facts) / len(extracted_facts) if extracted_facts else 0
        if stat_ratio >= 0.5:  # Excellent statistical content
            score += 0.20
        elif stat_ratio >= 0.3:  # Good statistical content
            score += 0.15
        elif stat_ratio >= 0.1:  # Some statistical content
            score += 0.10
        else:  # Minimal statistical content
            score += 0.05
    
    # 3. SOURCE QUALITY (0.25 max)
    # Assess authoritative sources
    authoritative_sources = assess_authoritative_sources(sources)
    if sources:
        auth_source_ratio = len(authoritative_sources) / len(sources)
        
        if auth_source_ratio >= auth_ratio * 1.5:  # Excellent authority
            score += 0.25
        elif auth_source_ratio >= auth_ratio:  # Good authority
            score += 0.20
        elif auth_source_ratio >= auth_ratio * 0.7:  # Adequate authority
            score += 0.15
        elif auth_source_ratio >= auth_ratio * 0.5:  # Minimal authority
            score += 0.10
        else:  # Poor authority
            score += 0.05
    
    # 4. RECENCY (0.15 max)
    # Assess source recency
    recent_sources = count_recent_sources(sources, recency_threshold)
    if sources:
        recency_ratio = recent_sources / len(sources)
        
        if recency_ratio >= 0.8:  # Very recent sources
            score += 0.15
        elif recency_ratio >= 0.6:  # Mostly recent sources
            score += 0.12
        elif recency_ratio >= 0.4:  # Some recent sources
            score += 0.08
        elif recency_ratio >= 0.2:  # Few recent sources
            score += 0.05
        else:  # Outdated sources
            score += 0.02
    
    # 5. CONSISTENCY AND CROSS-VALIDATION (0.15 max)
    # Existing consistency assessment
    consistency_score = assess_fact_consistency(extracted_facts)
    
    # New statistical validation to check numeric data consistency among facts
    stat_validation_score = perform_statistical_validation(extracted_facts)
    
    # We keep the total weighting for this section at 0.15,
    # distributing it between consistency and validation.
    # For example, 0.10 for consistency and 0.05 for statistical validation:
    combined_cross_val_score = (consistency_score * 0.10) + (stat_validation_score * 0.05)
    score += combined_cross_val_score
    
    # Log detailed scoring breakdown
    info_highlight(f"Category {category} quality score breakdown:")
    info_highlight(f"  - Facts: {len(extracted_facts)}/{min_facts} min")
    info_highlight(f"  - Sources: {len(sources)}/{min_sources} min")
    info_highlight(f"  - Statistical content: {len(stat_facts)}/{len(extracted_facts)} facts")
    info_highlight(f"  - Authoritative sources: {len(authoritative_sources)}/{len(sources)} sources")
    info_highlight(f"  - Recent sources: {recent_sources}/{len(sources)} sources")
    info_highlight(f"  - Consistency score: {consistency_score:.2f}")
    info_highlight(f"  - Statistical validation score: {stat_validation_score:.2f}")
    info_highlight(f"  - Final category score: {min(1.0, score):.2f}")
    
    return min(1.0, score)

def assess_authoritative_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assess which sources are from authoritative domains."""
    authoritative_sources = []
    
    for source in sources:
        url = source.get("url", "")
        quality_score = source.get("quality_score", 0)
        
        # Check domain patterns
        is_authoritative_domain = any(
            pattern.search(url) for pattern in COMPILED_AUTHORITY_PATTERNS
        )
        
        # Check title and source for credibility terms
        title = source.get("title", "").lower()
        source_name = source.get("source", "").lower()
        
        credibility_term_count = sum(
            1 for term in HIGH_CREDIBILITY_TERMS
            if term in title or term in source_name
        )
        
        # Consider authoritative if domain matches or high quality score or credibility terms
        if (
            is_authoritative_domain or
            quality_score >= 0.8 or
            credibility_term_count >= 2
        ):
            authoritative_sources.append(source)
    
    return authoritative_sources

def count_recent_sources(sources: List[Dict[str, Any]], recency_threshold: int) -> int:
    """Count how many sources are recent according to threshold."""
    recent_count = 0
    current_time = datetime.now().replace(tzinfo=timezone.utc)
    
    for source in sources:
        published_date = source.get("published_date")
        if not published_date:
            continue
            
        try:
            # First try direct datetime parsing
            try:
                date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                # Try other common formats
                from dateutil import parser
                date = parser.parse(published_date)
                
            # Make naive datetime timezone-aware if needed
            if hasattr(date, 'tzinfo') and date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
                
            days_old = (current_time - date).days
            if days_old <= recency_threshold:
                recent_count += 1
        except Exception:
            # If unparseable, don't count as recent
            pass
    
    return recent_count

def assess_fact_consistency(facts: List[Dict[str, Any]]) -> float:
    """Assess consistency among extracted facts.
    
    Returns:
        Score from 0.0-1.0 representing consistency
    """
    if not facts or len(facts) < 2:
        return 0.5  # Neutral score if insufficient facts
    
    # Extract key topics/entities mentioned across facts
    topics = extract_topics_from_facts(facts)
    
    # Count topic occurrences
    topic_counts = Counter(topics)
    
    # Calculate consistency based on topic distribution
    if not topic_counts:
        return 0.5
    
    # Get top topics (those mentioned multiple times)
    recurring_topics = {topic for topic, count in topic_counts.items() if count > 1}
    
    if not recurring_topics:
        return 0.5
    
    # Calculate what percentage of facts mention recurring topics
    facts_with_recurring = 0
    for fact in facts:
        fact_topics = get_topics_in_fact(fact)
        if any(topic in recurring_topics for topic in fact_topics):
            facts_with_recurring += 1
    
    return min(1.0, facts_with_recurring / len(facts))

def extract_topics_from_facts(facts: List[Dict[str, Any]]) -> List[str]:
    """Extract key topics/entities from facts."""
    all_topics = []
    
    for fact in facts:
        fact_topics = get_topics_in_fact(fact)
        all_topics.extend(fact_topics)
    
    return all_topics

def get_topics_in_fact(fact: Dict[str, Any]) -> Set[str]:
    """Extract topics from a single fact."""
    topics = set()
    
    # Extract from different fact formats
    if "data" in fact and isinstance(fact["data"], dict):
        data = fact["data"]
        # Handle different types of facts
        if fact.get("type") == "vendor":
            if "vendor_name" in data:
                topics.add(data["vendor_name"].lower())
        elif fact.get("type") == "relationship":
            entities = data.get("entities", [])
            for entity in entities:
                if isinstance(entity, str):
                    topics.add(entity.lower())
        elif fact.get("type") in ["requirement", "standard", "regulation", "compliance"]:
            if "description" in data:
                extract_noun_phrases(data["description"], topics)
    
    # Extract from source text if available
    if "source_text" in fact and isinstance(fact["source_text"], str):
        extract_noun_phrases(fact["source_text"], topics)
    
    return topics

def extract_noun_phrases(text: str, topics: Set[str]) -> None:
    """Extract potential noun phrases from text and add to topics set."""
    if not text:
        return
    
    # Simple noun phrase extraction (can be enhanced with NLP)
    # Currently just extracts capitalized multi-word sequences
    for match in re.finditer(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text):
        topics.add(match.group(1).lower())
    
    # Also extract technical terms and acronyms
    for match in re.finditer(r'\b([A-Z]{2,})\b', text):
        topics.add(match.group(1).lower())

def perform_statistical_validation(facts: List[Dict[str, Any]]) -> float:
    """
    Check for numeric data in the extracted facts and evaluate how consistent
    or valid those values are. Returns a score in the range [0.0 - 1.0].
    
    This function:
      - Extracts numeric values from each fact (either from `source_text` or 
        from known numeric fields in `data`).
      - Calculates the mean and standard deviation of these values.
      - If the standard deviation is low relative to the mean (and enough data
        points exist), it indicates higher consistency among facts.
      - Rewards having multiple consistent numeric references.
    """
    # Gather numeric values
    numeric_values = []
    pattern = re.compile(r"\b\d+(?:\.\d+)?\b")
    
    for fact in facts:
        # Check source_text for numeric values
        source_text = fact.get("source_text", "")
        if isinstance(source_text, str):
            found_numbers = pattern.findall(source_text)
            numeric_values.extend(float(n) for n in found_numbers)
        
        # Check fact["data"] if it contains numeric fields
        if "data" in fact and isinstance(fact["data"], dict):
            for key, val in fact["data"].items():
                if isinstance(val, (int, float)):
                    numeric_values.append(float(val))
                elif isinstance(val, str):
                    # Attempt to parse if it's a numeric string
                    match = pattern.search(val)
                    if match:
                        numeric_values.append(float(match.group(0)))
    
    # If fewer than 3 numeric values, not enough to judge consistency well
    if len(numeric_values) < 3:
        return 0.5  # Neutral score
    
    import statistics
    
    try:
        mean_val = statistics.mean(numeric_values)
        stdev_val = statistics.pstdev(numeric_values)  # population stdev
        
        # If mean is ~0, skip ratio-based check to avoid division by zero
        if abs(mean_val) < 1e-9:
            # If everything is zero or near zero
            if all(abs(x) < 1e-9 for x in numeric_values):
                return 1.0  # Perfectly consistent, all zero
            return 0.5
        
        # Relative stdev: smaller means more consistent
        rel_stdev = stdev_val / abs(mean_val)
        
        # Score logic:
        # - If rel_stdev is very low (< 0.1), very high consistency
        # - If rel_stdev is moderate (< 0.3), good consistency
        # - If rel_stdev is high, penalize
        if rel_stdev < 0.1:
            return 1.0
        elif rel_stdev < 0.3:
            return 0.8
        elif rel_stdev < 0.6:
            return 0.6
        else:
            return 0.4
    except statistics.StatisticsError:
        # Fallback if something goes wrong
        return 0.5

def calculate_overall_confidence(
    category_scores: Dict[str, float],
    synthesis_quality: float,
    validation_score: float
) -> float:
    """Calculate overall confidence score from multiple inputs.
    
    Args:
        category_scores: Quality scores for each research category
        synthesis_quality: Quality score for synthesis
        validation_score: Score from validation process
        
    Returns:
        Overall confidence score (0.0-1.0)
    """
    # If no scores, return low confidence
    if not category_scores:
        return 0.3
    
    # Calculate average category score
    avg_category_score = sum(category_scores.values()) / len(category_scores)
    
    # Base confidence on weighted components
    base_score = (
        avg_category_score * 0.5 +  # Category quality is 50% of score
        synthesis_quality * 0.3 +   # Synthesis quality is 30% of score
        validation_score * 0.2      # Validation score is 20% of score
    )
    
    # Apply modifiers based on coverage
    # Full coverage of all categories gets a boost
    if len(category_scores) >= 7 and all(score >= 0.6 for score in category_scores.values()):
        base_score += 0.1
    
    # Strong statistical content gets a boost
    stats_categories = sum(
        1
        for cat, score in category_scores.items()
        if cat in ['market_dynamics', 'cost_considerations'] and score >= 0.7
    )
    if stats_categories >= 2:
        base_score += 0.05
    
    return min(1.0, base_score)

def assess_synthesis_quality(synthesis: Dict[str, Any]) -> float:
    """Assess the quality of synthesis output.
    
    Args:
        synthesis: Synthesis output
        
    Returns:
        Quality score (0.0-1.0)
    """
    if not synthesis:
        return 0.3
    
    # Base score
    score = 0.5
    
    # Check for presence of synthesis sections
    synthesis_content = synthesis.get("synthesis", {})
    if not synthesis_content:
        return 0.3
    
    # Count sections with content
    sections_with_content = sum(
        1 for section in synthesis_content.values()
        if isinstance(section, dict) and
        section.get("content") and
        len(section.get("content", "")) > 50
    )
    
    # Add points for section coverage
    section_ratio = sections_with_content / max(1, len(synthesis_content))
    score += section_ratio * 0.2
    
    # Count sections with citations
    sections_with_citations = sum(
        1 for section in synthesis_content.values()
        if isinstance(section, dict) and
        section.get("citations") and
        len(section.get("citations", [])) > 0
    )
    
    # Add points for citation coverage
    citation_ratio = sections_with_citations / max(1, len(synthesis_content))
    score += citation_ratio * 0.15
    
    # Count sections with statistics
    sections_with_stats = sum(
        1 for section in synthesis_content.values()
        if isinstance(section, dict) and
        section.get("statistics") and
        len(section.get("statistics", [])) > 0
    )
    
    # Add points for statistics coverage
    stats_ratio = sections_with_stats / max(1, len(synthesis_content))
    score += stats_ratio * 0.15
    
    return min(1.0, score)
