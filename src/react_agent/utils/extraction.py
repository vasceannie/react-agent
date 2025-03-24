"""Enhanced extraction module for research categories with statistics focus.

This module improves the extraction of facts and statistics from search results,
with a particular emphasis on numerical data, trends, and statistical information.

Examples:
    Input text example:
        >>> text = '''
        According to a recent survey by TechCorp, 75% of enterprises adopted cloud 
        computing in 2023, up from 60% in 2022. The global cloud market reached 
        $483.3 billion in revenue, with AWS maintaining a 32% market share. 
        A separate study by MarketWatch revealed that cybersecurity spending 
        increased by 15% year-over-year.
        '''

    Extracting citations:
        >>> citations = extract_citations(text)
        >>> citations
        [
            {
                "source": "TechCorp",
                "context": "According to a recent survey by TechCorp, 75% of enterprises"
            },
            {
                "source": "MarketWatch",
                "context": "A separate study by MarketWatch revealed that cybersecurity"
            }
        ]

    Extracting statistics:
        >>> stats = extract_statistics(text)
        >>> stats
        [
            {
                "text": "75% of enterprises adopted cloud computing in 2023",
                "type": "percentage",
                "citations": [{"source": "TechCorp", "context": "...survey by TechCorp, 75% of enterprises..."}],
                "quality_score": 0.95,
                "year_mentioned": 2023
            },
            {
                "text": "global cloud market reached $483.3 billion in revenue",
                "type": "financial",
                "citations": [],
                "quality_score": 0.85,
                "year_mentioned": None
            },
            {
                "text": "AWS maintaining a 32% market share",
                "type": "market",
                "citations": [],
                "quality_score": 0.75,
                "year_mentioned": None
            }
        ]

    Rating statistic quality:
        >>> stat_text = "According to Gartner's 2023 survey, 78.5% of Fortune 500 companies..."
        >>> quality = rate_statistic_quality(stat_text)
        >>> quality
        0.95  # High score due to specific percentage, citation, and year

    Inferring statistic type:
        >>> text1 = "Market share increased to 45%"
        >>> infer_statistic_type(text1)
        'percentage'
        >>> text2 = "$50 million in revenue"
        >>> infer_statistic_type(text2)
        'financial'

    Extracting year:
        >>> text_with_year = "In 2023, cloud adoption grew by 25%"
        >>> extract_year(text_with_year)
        2023

    Finding JSON objects:
        >>> text_with_json = 'Some text {"key": "value", "nested": {"data": 123}} more text'
        >>> json_obj = find_json_object(text_with_json)
        >>> json_obj
        '{"key": "value", "nested": {"data": 123}}'

    Enriching extracted facts:
        >>> fact = {
        ...     "text": "Cloud adoption grew by 25% in 2023",
        ...     "confidence": 0.8,
        ...     "source_text": "According to AWS, cloud adoption grew by 25% in 2023"
        ... }
        >>> enriched = enrich_extracted_fact(
        ...     fact,
        ...     url="https://example.com/report",
        ...     source_title="Cloud Market Report"
        ... )
        >>> enriched
        {
            "text": "Cloud adoption grew by 25% in 2023",
            "confidence": 0.8,
            "source_text": "According to AWS, cloud adoption grew by 25% in 2023",
            "source_url": "https://example.com/report",
            "source_title": "Cloud Market Report",
            "source_domain": "example.com",
            "extraction_timestamp": "2024-03-14T10:30:00",
            "statistics": [...],
            "additional_citations": [...],
            "confidence_score": 0.9  # Adjusted based on statistics and citations
        }

    Full category information extraction:
        >>> facts, relevance = await extract_category_information(
        ...     content=text,
        ...     url="https://example.com/cloud-report",
        ...     title="Cloud Computing Trends 2023",
        ...     category="market_dynamics",
        ...     original_query="cloud computing adoption trends",
        ...     prompt_template="...",
        ...     extraction_model=model
        ... )
        >>> facts
        [
            {
                "type": "fact",
                "data": {
                    "text": "Enterprise cloud adoption increased to 75% in 2023",
                    "source_url": "https://example.com/cloud-report",
                    "source_title": "Cloud Computing Trends 2023",
                    "source_domain": "example.com",
                    "extraction_timestamp": "2024-03-14T10:30:00",
                    "confidence_score": 0.9,
                    "statistics": [...],
                    "additional_citations": [...]
                }
            }
        ]
        >>> relevance
        0.95

Functions:
    extract_citations(text: str) -> List[Dict[str, str]]
        Extract citation information from text.

    extract_statistics(text: str, url: str = "", source_title: str = "") -> List[Dict[str, Any]]
        Extract statistics and numerical data from text.

    rate_statistic_quality(stat_text: str) -> float
        Rate the quality of a statistic on a scale of 0.0 to 1.0.

    extract_year(text: str) -> Optional[int]
        Extract year mentioned in text, if any.

    infer_statistic_type(text: str) -> str
        Infer the type of statistic mentioned.

    enrich_extracted_fact(fact: Dict[str, Any], url: str, source_title: str) -> Dict[str, Any]
        Enrich an extracted fact with additional context and metadata.

    find_json_object(text: str) -> Optional[str]
        Find a JSON object in text using balanced brace matching.

    extract_category_information(content: str, url: str, title: str, category: str,
                               original_query: str, prompt_template: str,
                               extraction_model: Any) -> Tuple[List[Dict[str, Any]], float]
        Extract information from content for a specific category with enhanced statistics focus.
"""


import contextlib
from typing import Dict, List, Any, Optional, Union, Tuple
import re
import json
from datetime import datetime, timezone
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from react_agent.utils.logging import get_logger, info_highlight, warning_highlight, error_highlight
from react_agent.utils.validations import is_valid_url
from react_agent.utils.content import chunk_text, preprocess_content, merge_chunk_results
from react_agent.utils.defaults import get_default_extraction_result
from react_agent.utils.cache import ProcessorCache, create_checkpoint, load_checkpoint, cache_result
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from react_agent.utils.statistics import (
    calculate_category_quality_score,
    assess_authoritative_sources,
    HIGH_CREDIBILITY_TERMS
)

# Initialize logger and cache
logger = get_logger(__name__)
json_cache = ProcessorCache(thread_id="json_parser")

# Regular expressions for identifying statistical content
STAT_PATTERNS = [
    r'\d+%',  # Percentage
    r'\$\d+(?:,\d+)*(?:\.\d+)?(?:\s?(?:million|billion|trillion))?',  # Currency
    r'\d+(?:\.\d+)?(?:\s?(?:million|billion|trillion))?',  # Numbers with scale
    r'increased by|decreased by|grew by|reduced by|rose|fell',  # Trend language
    r'majority|minority|fraction|proportion|ratio',  # Proportion language
    r'survey|respondents|participants|study found',  # Research language
    r'statistics show|data indicates|report reveals',  # Statistical citation
    r'market share|growth rate|adoption rate|satisfaction score',  # Business metrics
    r'average|mean|median|mode|range|standard deviation'  # Statistical terms
]

COMPILED_STAT_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in STAT_PATTERNS]

def extract_citations(text: str) -> List[Dict[str, str]]:
    """Extract citation information from text."""
    citations = []
    
    # Find patterns like (Source: X), [X], cited from X, etc.
    citation_patterns = [
        r'\(Source:?\s+([^)]+)\)',
        r'\[([^]]+)\]',
        r'cited\s+from\s+([^,.;]+)',
        r'according\s+to\s+([^,.;]+)',
        r'reported\s+by\s+([^,.;]+)',
        r'([^,.;]+)\s+reports'
    ]
    
    for pattern in citation_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            citation = match.group(1).strip()
            # Skip dates that might be captured in brackets like [2022]
            if not re.match(r'^(19|20)\d{2}$', citation):
                citations.append({
                    "source": citation,
                    "context": text[max(0, match.start() - 50):min(len(text), match.end() + 50)]
                })
    
    return citations

def extract_statistics(text: str, url: str = "", source_title: str = "") -> List[Dict[str, Any]]:
    """Extract statistics and numerical data from text.
    
    This function identifies and extracts statistical information from text,
    including percentages, currency amounts, numerical data, and their context.
    It also assesses the quality of each statistic based on various factors.
    
    Args:
        text: The input text to extract statistics from
        url: Optional URL source of the statistics
        source_title: Optional title of the source document
        
    Returns:
        List of dictionaries containing extracted statistics with metadata
    """
    statistics = []
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sentence in sentences:
        for pattern in COMPILED_STAT_PATTERNS:
            if pattern.search(sentence):
                # Avoid duplicate statistics
                stat_text = sentence.strip()
                if all(s["text"] != stat_text for s in statistics):
                    # Create statistic object with enhanced metadata
                    statistic = {
                        "text": stat_text,
                        "type": infer_statistic_type(stat_text),
                        "citations": extract_citations(stat_text),
                        "year_mentioned": extract_year(stat_text),
                        "source_quality": assess_source_quality(stat_text),
                        "quality_score": rate_statistic_quality(stat_text),
                        "credibility_terms": extract_credibility_terms(stat_text)
                    }

                    if enriched := enrich_extracted_fact(
                        statistic, url, source_title
                    ):
                        statistic |= enriched

                    statistics.append(statistic)
                break

    return statistics

def rate_statistic_quality(stat_text: str) -> float:
    """Rate the quality of a statistic on a scale of 0.0 to 1.0."""
    score = 0.5  # Base score
    
    # Higher quality if includes specific numbers
    if re.search(r'\d+(?:\.\d+)?%', stat_text):
        score += 0.15  # Specific percentage
    elif re.search(r'\$\d+(?:,\d+)*(?:\.\d+)?', stat_text):
        score += 0.15  # Specific currency amount
    
    # Higher quality if cites source
    if re.search(r'according to|reported by|cited from|source|study|survey', stat_text, re.IGNORECASE):
        score += 0.2
    
    # Higher quality if includes year
    if extract_year(stat_text):
        score += 0.1
    
    # Lower quality if uses vague language
    if re.search(r'may|might|could|possibly|potentially|estimated', stat_text, re.IGNORECASE):
        score -= 0.1
    
    # Cap between 0.0 and 1.0
    return max(0.0, min(1.0, score))

def extract_year(text: str) -> Optional[int]:
    """Extract year mentioned in text, if any."""
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
    return int(year_match[1]) if year_match else None

def infer_statistic_type(text: str) -> str:
    """Infer the type of statistic mentioned."""
    if re.search(r'%|percent|percentage', text, re.IGNORECASE):
        return "percentage"
    elif re.search(r'\$|\beuro\b|\beur\b|\bgbp\b|\bjpy\b|cost|price|spend|budget', text, re.IGNORECASE):
        return "financial"
    elif re.search(r'time|duration|period|year|month|week|day|hour', text, re.IGNORECASE):
        return "temporal"
    elif re.search(r'ratio|proportion|fraction', text, re.IGNORECASE):
        return "ratio"
    elif re.search(r'increase|decrease|growth|decline|trend', text, re.IGNORECASE):
        return "trend"
    elif re.search(r'survey|respondent|participant', text, re.IGNORECASE):
        return "survey"
    elif re.search(r'market share|market size', text, re.IGNORECASE):
        return "market"
    else:
        return "general"

def enrich_extracted_fact(fact: Dict[str, Any], url: str, source_title: str) -> Dict[str, Any]:
    """Enrich an extracted fact with additional context and metadata."""
    # Add source information
    fact["source_url"] = url
    fact["source_title"] = source_title

    # Extract domain from URL
    try:
        domain = urlparse(url).netloc
        fact["source_domain"] = domain
    except Exception:
        fact["source_domain"] = ""

    # Add timestamp
    fact["extraction_timestamp"] = datetime.now().isoformat()

    # Extract statistics if present in source_text
    if "source_text" in fact and isinstance(fact["source_text"], str):
        if statistics := extract_statistics(fact["source_text"], url, source_title):
            fact["statistics"] = statistics

    # Extract citations if present in source_text
    if "source_text" in fact and isinstance(fact["source_text"], str):
        if citations := extract_citations(fact["source_text"]):
            fact["additional_citations"] = citations

    # Add confidence score based on enriched data
    confidence_score = fact.get("confidence", 0.5)
    if isinstance(confidence_score, str):
        # Convert string confidence to float
        if confidence_score.lower() == "high":
            confidence_score = 0.9
        elif confidence_score.lower() == "medium":
            confidence_score = 0.7
        elif confidence_score.lower() == "low":
            confidence_score = 0.4
        else:
            confidence_score = 0.5

    # Adjust confidence based on statistics and citations
    if "statistics" in fact and len(fact["statistics"]) > 0:
        confidence_score = min(1.0, confidence_score + 0.1)

    if "additional_citations" in fact and len(fact["additional_citations"]) > 0:
        confidence_score = min(1.0, confidence_score + 0.1)

    fact["confidence_score"] = confidence_score

    return fact

def find_json_object(text: str) -> Optional[str]:
    """Find a JSON object in text using balanced brace matching.
    
    This is more robust than simple regex for finding JSON objects as it:
    1. Handles nested braces correctly
    2. Supports both objects and arrays as root elements
    3. Finds the longest valid JSON-like structure
    
    Args:
        text: Text that may contain a JSON object
        
    Returns:
        Extracted JSON-like text or None if not found
    """
    # Look for both object and array patterns
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        # Find potential starting positions
        start_positions = [pos for pos, char in enumerate(text) if char == start_char]
        
        for start_pos in start_positions:
            # Track nesting level
            level = 0
            # Track position
            pos = start_pos
            
            # Scan through text tracking brace/bracket balance
            while pos < len(text):
                char = text[pos]
                if char == start_char:
                    level += 1
                elif char == end_char:
                    level -= 1
                    # If we've found a balanced structure, extract it
                    if level == 0:
                        return text[start_pos:pos+1]
                pos += 1
    
    # No balanced JSON-like structure found
    return None

def _check_cache(response, category):
    # Clean response string
    response = _clean_json_string(response)

    # Check cache
    cache_key = f"json_parse_{hash(response)}"
    if cached_result := json_cache.get(cache_key):
        if isinstance(cached_result, dict) and cached_result.get("data"):
            return cached_result["data"]

    # Find and parse JSON
    json_text = find_json_object(response)
    if not json_text:
        return get_default_extraction_result(category)

    parsed = json.loads(json_text)
    result = _merge_with_default(parsed, category)

    # Cache result
    json_cache.put(
        cache_key,
        {
            "data": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ttl": 3600
        }
    )

    return result

def _clean_json_string(text: str) -> str:
    """Clean and normalize JSON string."""
    text = re.sub(r'```(?:json)?\s*|\s*```', '', text)  # Remove code blocks
    text = text.strip()
    text = text.replace("'", '"')  # Handle single quotes
    text = re.sub(r',(\s*[}\]])', r'\1', text)  # Handle trailing commas
    
    # Add missing braces
    if not text.startswith('{'): text = '{' + text
    if not text.endswith('}'): text = text + '}'
    
    return re.sub(r'"(\w+)":\s*"', r'"\1": []', text)  # Handle quoted keys without values

def _merge_with_default(parsed: Dict[str, Any], category: str) -> Dict[str, Any]:
    """Merge parsed result with default template."""
    result = get_default_extraction_result(category)
    
    for key, value in parsed.items():
        if key not in result:
            continue
            
        if isinstance(value, type(result[key])):
            if isinstance(value, (list, dict)):
                result[key] = value
            elif isinstance(value, (int, float)) and key in ['relevance_score', 'confidence_score']:
                result[key] = float(value)
    
    return result

def assess_source_quality(text: str) -> float:
    """Assess the quality of the source mentioned in the statistic."""
    score = 0.5  # Base score

    # Check for credibility terms
    credibility_count = sum(
        term.lower() in text.lower() for term in HIGH_CREDIBILITY_TERMS
    )

    # Add points for credibility terms
    if credibility_count >= 2:
        score += 0.3
    elif credibility_count == 1:
        score += 0.15

    # Add points for specific source citations
    if re.search(r'according to|reported by|cited from|source|study|survey', text, re.IGNORECASE):
        score += 0.2

    # Add points for authoritative sources
    if any(domain in text.lower() for domain in ['.gov', '.edu', '.org', 'research', 'university']):
        score += 0.2

    return min(1.0, score)

def extract_credibility_terms(text: str) -> List[str]:
    """Extract credibility-indicating terms from the text."""
    return [
        term for term in HIGH_CREDIBILITY_TERMS
        if term.lower() in text.lower()
    ]

async def extract_category_information(
    content: str,
    url: str,
    title: str,
    category: str,
    original_query: str,
    prompt_template: str,
    extraction_model: Any,
    config: Optional[RunnableConfig] = None
) -> Tuple[List[Dict[str, Any]], float]:
    """Extract information from content for a specific category with enhanced statistics focus."""
    if not _validate_inputs(content, url):
        return [], 0.0

    info_highlight(f"Extracting from {url} for {category}")
    
    try:
        content = preprocess_content(content, url)
        prompt = prompt_template.format(query=original_query, url=url, content=content)
        
        extraction_result = await _process_content(
            content=content,
            prompt=prompt,
            category=category,
            extraction_model=extraction_model,
            config=config,
            url=url,
            title=title
        )
        
        facts = _get_category_facts(category, extraction_result)
        enriched_facts = [enrich_extracted_fact(fact, url, title) for fact in facts]
        
        return sorted(
            enriched_facts,
            key=lambda x: x.get("confidence_score", 0),
            reverse=True
        ), extraction_result.get("relevance_score", 0.0)
        
    except Exception as e:
        error_highlight(f"Error extracting from {url}: {str(e)}")
        return [], 0.0

def _validate_inputs(content: str, url: str) -> bool:
    """Validate input content and URL."""
    if not content or not url or not is_valid_url(url):
        warning_highlight(f"Invalid content or URL for extraction: {url}")
        return False
    return True

async def _process_content(
    content: str,
    prompt: str,
    category: str,
    extraction_model: Any,
    config: Optional[RunnableConfig],
    url: str,
    title: str
) -> Dict[str, Any]:
    """Process content and extract information using the model."""
    if len(content) > 40000:
        return await _process_chunked_content(content, prompt, category, extraction_model, config, url, title)
    
    model_response = await extraction_model(
        messages=[{"role": "human", "content": prompt}],
        config=config
    )
    
    extraction_result = safe_json_parse(model_response, category)
    if statistics := extract_statistics(content, url, title):
        extraction_result["statistics"] = statistics
        
    return extraction_result

async def _process_chunked_content(
    content: str,
    prompt: str,
    category: str,
    extraction_model: Any,
    config: Optional[RunnableConfig],
    url: str,
    title: str
) -> Dict[str, Any]:
    """Process content in chunks when it exceeds size limit."""
    info_highlight(f"Content too large ({len(content)} chars), chunking...")
    chunks = chunk_text(content)
    
    all_statistics = []
    chunk_results = []
    
    for chunk_idx, chunk in enumerate(chunks):
        info_highlight(f"Processing chunk {chunk_idx + 1}/{len(chunks)}")
        chunk_prompt = prompt.format(content=chunk)
        
        chunk_response = await extraction_model(
            messages=[{"role": "human", "content": chunk_prompt}],
            config=config
        )
        
        if chunk_result := safe_json_parse(chunk_response, category):
            chunk_statistics = extract_statistics(chunk, url, title)
            all_statistics.extend(chunk_statistics)
            chunk_results.append(chunk_result)
    
    result = merge_chunk_results(chunk_results, category)
    if all_statistics:
        result["statistics"] = all_statistics
        
    return result

def _get_category_facts(category: str, extraction_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract facts from extraction result based on category structure."""
    if not extraction_result or not isinstance(extraction_result, dict):
        return []

    # Mapping of categories to their corresponding fact types and keys
    category_mapping = {
        "market_dynamics": [("fact", "extracted_facts")],
        "provider_landscape": [
            ("vendor", "extracted_vendors"),
            ("relationship", "vendor_relationships")
        ],
        "technical_requirements": [
            ("requirement", "extracted_requirements"),
            ("standard", "standards")
        ],
        "regulatory_landscape": [
            ("regulation", "extracted_regulations"),
            ("compliance", "compliance_requirements")
        ],
        "cost_considerations": [
            ("cost", "extracted_costs"),
            ("pricing_model", "pricing_models")
        ],
        "best_practices": [
            ("practice", "extracted_practices"),
            ("methodology", "methodologies")
        ],
        "implementation_factors": [
            ("factor", "extracted_factors"),
            ("challenge", "challenges")
        ]
    }

    facts = []
    for fact_type, key in category_mapping.get(category, [("fact", "extracted_facts")]):
        items = extraction_result.get(key, [])
        facts.extend([{"type": fact_type, "data": item} for item in items])
    
    return facts

@cache_result(ttl=3600)
def safe_json_parse(response: Union[str, Dict[str, Any]], category: str) -> Dict[str, Any]:
    """Safely parse JSON response with enhanced error handling and cleanup."""
    # Return if already a dict
    if isinstance(response, dict):
        return response

    # Handle non-string input
    if not isinstance(response, str):
        return get_default_extraction_result(category)

    try:
        return _check_cache(response, category)
    except Exception as e:
        error_highlight(f"Error parsing JSON: {str(e)}")
        return get_default_extraction_result(category)