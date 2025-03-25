"""Improved confidence scoring with statistical validation.

This module enhances the confidence scoring logic to focus on statistical
validation, source quality assessment, and cross-validation to achieve
confidence scores above 80%.

Examples:
    >>> # Example usage of calculate_category_quality_score:
    >>> category = "market_dynamics"
    >>> extracted_facts = [{"text": "Fact 1", "source_text": "Report by Gov.", "data": {}},
    ...                    {"text": "Fact 2", "source_text": "Study from Uni.", "data": {}}]
    >>> sources = [{"url": "https://example.gov/report", "quality_score": 0.9, "title": "Gov Report", "source": "Gov"},
    ...            {"url": "https://university.edu/study", "quality_score": 0.85, "title": "University Study", "source": "Uni"}]
    >>> thresholds = {"min_facts": 3, "min_sources": 2, "authoritative_source_ratio": 0.5, "recency_threshold_days": 365}
    >>> score = calculate_category_quality_score(category, extracted_facts, sources, thresholds)
    >>> print(score)  # Outputs a quality score between 0.0 and 1.0
"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timezone
from dateutil import parser
import re
from urllib.parse import urlparse
from collections import Counter

from react_agent.utils.logging import get_logger, info_highlight, warning_highlight, error_highlight

# Initialize logger
logger = get_logger(__name__)

# Authority domain patterns for authoritative sources.
AUTHORITY_DOMAINS = [
    r'\.gov($|/)',    # Government domains
    r'\.edu($|/)',    # Educational institutions
    r'\.org($|/)',    # Non-profit organizations
    r'research\.',    # Research organizations
    r'\.ac\.($|/)',   # Academic institutions
    r'journal\.',     # Academic journals
    r'university\.',   # Universities
    r'institute\.',    # Research institutes
    r'association\.'   # Professional associations
]

# Compile patterns for efficiency.
COMPILED_AUTHORITY_PATTERNS = [re.compile(pattern) for pattern in AUTHORITY_DOMAINS]

# High-credibility source terms.
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
    """
    Calculate an enhanced quality score for a research category based on extracted facts and sources.

    The score is built from several weighted components:
      1. Quantity assessment of facts and sources.
      2. Presence of statistical content.
      3. Authoritative source evaluation.
      4. Recency of the sources.
      5. Consistency and cross-validation of the extracted facts.

    Args:
        category (str): The research category (e.g., "market_dynamics").
        extracted_facts (List[Dict[str, Any]]): A list of fact dictionaries extracted from content.
        sources (List[Dict[str, Any]]): A list of source dictionaries.
        thresholds (Dict[str, Any]): A dictionary of threshold values including:
            - "min_facts": Minimum number of facts expected.
            - "min_sources": Minimum number of sources expected.
            - "authoritative_source_ratio": Desired ratio of authoritative sources.
            - "recency_threshold_days": Maximum age (in days) for a source to be considered recent.

    Returns:
        float: The final quality score between 0.0 and 1.0.

    Examples:
        >>> score = calculate_category_quality_score("market_dynamics", extracted_facts, sources, thresholds)
        >>> print(score)
    """
    score = 0.35  # Start with a slightly higher base score.

    # Retrieve thresholds.
    min_facts = thresholds.get("min_facts", 3)
    min_sources = thresholds.get("min_sources", 2)
    auth_ratio = thresholds.get("authoritative_source_ratio", 0.5)
    recency_threshold = thresholds.get("recency_threshold_days", 365)

    # 1. Quantity Assessment (up to 0.25)
    if len(extracted_facts) >= min_facts * 3:
        score += 0.15
    elif len(extracted_facts) >= min_facts * 2:
        score += 0.12
    elif len(extracted_facts) >= min_facts:
        score += 0.08
    else:
        fact_ratio = len(extracted_facts) / min_facts if min_facts else 0
        score += fact_ratio * 0.06

    if len(sources) >= min_sources * 3:
        score += 0.10
    elif len(sources) >= min_sources * 2:
        score += 0.08
    elif len(sources) >= min_sources:
        score += 0.05
    else:
        source_ratio = len(sources) / min_sources if min_sources else 0
        score += source_ratio * 0.03

    # 2. Statistical Content (up to 0.20)
    stat_facts = [
        f for f in extracted_facts
        if "statistics" in f or
        (isinstance(f.get("source_text"), str) and re.search(r'\d+', f.get("source_text", "")) is not None)
    ]
    if stat_facts:
        stat_ratio = len(stat_facts) / len(extracted_facts) if extracted_facts else 0
        if stat_ratio >= 0.5:
            score += 0.20
        elif stat_ratio >= 0.3:
            score += 0.15
        elif stat_ratio >= 0.1:
            score += 0.10
        else:
            score += 0.05

    # 3. Source Quality (up to 0.25)
    authoritative_sources = assess_authoritative_sources(sources)
    if sources:
        auth_source_ratio = len(authoritative_sources) / len(sources)
        if auth_source_ratio >= auth_ratio * 1.5:
            score += 0.25
        elif auth_source_ratio >= auth_ratio:
            score += 0.20
        elif auth_source_ratio >= auth_ratio * 0.7:
            score += 0.15
        elif auth_source_ratio >= auth_ratio * 0.5:
            score += 0.10
        else:
            score += 0.05

    # 4. Recency (up to 0.15)
    recent_sources = count_recent_sources(sources, recency_threshold)
    if sources:
        recency_ratio = recent_sources / len(sources)
        if recency_ratio >= 0.8:
            score += 0.15
        elif recency_ratio >= 0.6:
            score += 0.12
        elif recency_ratio >= 0.4:
            score += 0.08
        elif recency_ratio >= 0.2:
            score += 0.05
        else:
            score += 0.02

    # 5. Consistency and Cross-Validation (up to 0.15)
    consistency_score = assess_fact_consistency(extracted_facts)
    stat_validation_score = perform_statistical_validation(extracted_facts)
    combined_cross_val_score = (consistency_score * 0.10) + (stat_validation_score * 0.05)
    score += combined_cross_val_score

    # Log detailed breakdown.
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
    """
    Assess and return sources considered authoritative based on domain and credibility terms.

    A source is considered authoritative if its URL domain matches known patterns,
    if it has a high quality score, or if its title/source field contains multiple
    high-credibility terms.

    Args:
        sources (List[Dict[str, Any]]): A list of source dictionaries.

    Returns:
        List[Dict[str, Any]]: A list of sources deemed authoritative.

    Examples:
        >>> auth_sources = assess_authoritative_sources(sources)
        >>> print(len(auth_sources))
    """
    authoritative_sources = []
    for source in sources:
        url = source.get("url", "")
        quality_score = source.get("quality_score", 0)
        is_authoritative_domain = any(
            pattern.search(url) for pattern in COMPILED_AUTHORITY_PATTERNS
        )
        title = source.get("title", "").lower()
        source_name = source.get("source", "").lower()
        credibility_term_count = sum(
            1 for term in HIGH_CREDIBILITY_TERMS if term in title or term in source_name
        )
        if is_authoritative_domain or quality_score >= 0.8 or credibility_term_count >= 2:
            authoritative_sources.append(source)
    return authoritative_sources


def count_recent_sources(sources: List[Dict[str, Any]], recency_threshold: int) -> int:
    """
    Count how many sources are considered recent based on a recency threshold (in days).

    A source is recent if its published date (as ISO string or parseable format) is within
    the specified number of days from the current time.

    Args:
        sources (List[Dict[str, Any]]): A list of source dictionaries.
        recency_threshold (int): The maximum age in days for a source to be considered recent.

    Returns:
        int: The count of recent sources.

    Examples:
        >>> recent_count = count_recent_sources(sources, 365)
        >>> print(recent_count)
    """
    recent_count = 0
    current_time = datetime.now().replace(tzinfo=timezone.utc)
    for source in sources:
        published_date = source.get("published_date")
        if not published_date:
            continue
        try:
            try:
                date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                date = parser.parse(published_date)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            days_old = (current_time - date).days
            if days_old <= recency_threshold:
                recent_count += 1
        except Exception:
            pass
    return recent_count


def assess_fact_consistency(facts: List[Dict[str, Any]]) -> float:
    """
    Assess consistency among extracted facts based on common topics.

    The function extracts topics from each fact and calculates what percentage
    of the facts mention recurring topics. A higher percentage indicates higher consistency.

    Args:
        facts (List[Dict[str, Any]]): A list of fact dictionaries.

    Returns:
        float: A consistency score between 0.0 and 1.0.

    Examples:
        >>> consistency = assess_fact_consistency(extracted_facts)
        >>> print(consistency)
    """
    if not facts or len(facts) < 2:
        return 0.5  # Neutral score if insufficient facts
    topics = extract_topics_from_facts(facts)
    topic_counts = Counter(topics)
    if not topic_counts:
        return 0.5
    recurring_topics = {topic for topic, count in topic_counts.items() if count > 1}
    if not recurring_topics:
        return 0.5
    facts_with_recurring = sum(
        1 for fact in facts if any(topic in get_topics_in_fact(fact) for topic in recurring_topics)
    )
    return min(1.0, facts_with_recurring / len(facts))


def extract_topics_from_facts(facts: List[Dict[str, Any]]) -> List[str]:
    """
    Extract key topics or entities from a list of facts.

    This function aggregates topics from individual facts and returns a combined list.

    Args:
        facts (List[Dict[str, Any]]): A list of fact dictionaries.

    Returns:
        List[str]: A list of topics extracted from the facts.

    Examples:
        >>> topics = extract_topics_from_facts(extracted_facts)
        >>> print(topics)
    """
    all_topics: List[str] = []
    for fact in facts:
        fact_topics = get_topics_in_fact(fact)
        all_topics.extend(fact_topics)
    return all_topics


def get_topics_in_fact(fact: Dict[str, Any]) -> Set[str]:
    """
    Extract topics from a single fact.

    Topics are extracted from the 'data' field or 'source_text' if available.
    For example, vendor names or technical terms.

    Args:
        fact (Dict[str, Any]): A fact dictionary.

    Returns:
        Set[str]: A set of topics found in the fact.

    Examples:
        >>> topics = get_topics_in_fact(fact)
        >>> print(topics)
    """
    topics = set()
    if "data" in fact and isinstance(fact["data"], dict):
        data = fact["data"]
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
    if "source_text" in fact and isinstance(fact["source_text"], str):
        extract_noun_phrases(fact["source_text"], topics)
    return topics


def extract_noun_phrases(text: str, topics: Set[str]) -> None:
    """
    Extract potential noun phrases from text and add them to a topics set.

    This basic extraction finds capitalized multi-word sequences and acronyms.

    Args:
        text (str): The text from which to extract noun phrases.
        topics (Set[str]): A set to which the extracted phrases will be added.

    Returns:
        None

    Examples:
        >>> topics = set()
        >>> extract_noun_phrases("Cloud Computing Trends", topics)
        >>> print(topics)
        {'cloud computing trends'}
    """
    if not text:
        return
    for match in re.finditer(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text):
        topics.add(match.group(1).lower())
    for match in re.finditer(r'\b([A-Z]{2,})\b', text):
        topics.add(match.group(1).lower())


def perform_statistical_validation(facts: List[Dict[str, Any]]) -> float:
    """
    Validate numeric data consistency among extracted facts.

    The function extracts numeric values from each fact's 'source_text' and 'data' fields,
    then computes the relative standard deviation. A lower relative standard deviation indicates
    higher consistency.

    Returns a score between 0.0 and 1.0 based on the consistency.

    Args:
        facts (List[Dict[str, Any]]): A list of fact dictionaries.

    Returns:
        float: The statistical validation score.

    Examples:
        >>> validation_score = perform_statistical_validation(extracted_facts)
        >>> print(validation_score)
    """
    numeric_values: List[float] = []
    pattern = re.compile(r"\b\d+(?:\.\d+)?\b")
    for fact in facts:
        source_text = fact.get("source_text", "")
        if isinstance(source_text, str):
            found_numbers = pattern.findall(source_text)
            numeric_values.extend(float(n) for n in found_numbers)
        if "data" in fact and isinstance(fact["data"], dict):
            for key, val in fact["data"].items():
                if isinstance(val, (int, float)):
                    numeric_values.append(float(val))
                elif isinstance(val, str):
                    match = pattern.search(val)
                    if match:
                        numeric_values.append(float(match.group(0)))
    if len(numeric_values) < 3:
        return 0.5  # Neutral score if insufficient numeric data.
    import statistics
    try:
        mean_val = statistics.mean(numeric_values)
        stdev_val = statistics.pstdev(numeric_values)
        if abs(mean_val) < 1e-9:
            if all(abs(x) < 1e-9 for x in numeric_values):
                return 1.0
            return 0.5
        rel_stdev = stdev_val / abs(mean_val)
        if rel_stdev < 0.1:
            return 1.0
        elif rel_stdev < 0.3:
            return 0.8
        elif rel_stdev < 0.6:
            return 0.6
        else:
            return 0.4
    except statistics.StatisticsError:
        return 0.5


def calculate_overall_confidence(
    category_scores: Dict[str, float],
    synthesis_quality: float,
    validation_score: float
) -> float:
    """
    Calculate an overall confidence score from category scores, synthesis quality, and validation score.

    The overall score is a weighted average of:
      - Average category score (50%)
      - Synthesis quality (30%)
      - Validation score (20%)

    Additional boosts are applied for full category coverage and strong statistical content.

    Args:
        category_scores (Dict[str, float]): A mapping of category names to quality scores.
        synthesis_quality (float): The quality score of the synthesis process.
        validation_score (float): The validation score from statistical checks.

    Returns:
        float: The overall confidence score between 0.0 and 1.0.

    Examples:
        >>> overall = calculate_overall_confidence(category_scores, 0.8, 0.7)
        >>> print(overall)
    """
    if not category_scores:
        return 0.3
    avg_category_score = sum(category_scores.values()) / len(category_scores)
    base_score = (
        avg_category_score * 0.5 +
        synthesis_quality * 0.3 +
        validation_score * 0.2
    )
    if len(category_scores) >= 7 and all(score >= 0.6 for score in category_scores.values()):
        base_score += 0.1
    stats_categories = sum(
        1 for cat, score in category_scores.items() if cat in ['market_dynamics', 'cost_considerations'] and score >= 0.7
    )
    if stats_categories >= 2:
        base_score += 0.05
    return min(1.0, base_score)


def assess_synthesis_quality(synthesis: Dict[str, Any]) -> float:
    """
    Assess the quality of synthesis output based on section content, citations, and statistics.

    The function checks for the presence and coverage of synthesis sections, their content,
    and associated citations and statistics to determine a quality score.

    Args:
        synthesis (Dict[str, Any]): A dictionary representing the synthesis output.

    Returns:
        float: A synthesis quality score between 0.0 and 1.0.

    Examples:
        >>> quality = assess_synthesis_quality(synthesis_output)
        >>> print(quality)
    """
    if not synthesis:
        return 0.3
    score = 0.5
    synthesis_content = synthesis.get("synthesis", {})
    if not synthesis_content:
        return 0.3
    sections_with_content = sum(
        1 for section in synthesis_content.values()
        if isinstance(section, dict) and section.get("content") and len(section.get("content", "")) > 50
    )
    section_ratio = sections_with_content / max(1, len(synthesis_content))
    score += section_ratio * 0.2
    sections_with_citations = sum(
        1 for section in synthesis_content.values()
        if isinstance(section, dict) and section.get("citations") and len(section.get("citations", [])) > 0
    )
    citation_ratio = sections_with_citations / max(1, len(synthesis_content))
    score += citation_ratio * 0.15
    sections_with_stats = sum(
        1 for section in synthesis_content.values()
        if isinstance(section, dict) and section.get("statistics") and len(section.get("statistics", [])) > 0
    )
    stats_ratio = sections_with_stats / max(1, len(synthesis_content))
    score += stats_ratio * 0.15
    return min(1.0, score)
