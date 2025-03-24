"""Enhanced extraction module for research categories with statistics focus.

This module improves the extraction of facts and statistics from search results,
with a particular emphasis on numerical data, trends, and statistical information.
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
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Initialize logger
logger = get_logger(__name__)

# Initialize memory saver for caching
memory_saver = MemorySaver()

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

def extract_statistics(text: str) -> List[Dict[str, Any]]:
    """Extract statistics and numerical data from text."""
    statistics = []
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sentence in sentences:
        for pattern in COMPILED_STAT_PATTERNS:
            if pattern.search(sentence):
                # Avoid duplicate statistics
                stat_text = sentence.strip()
                if all(s["text"] != stat_text for s in statistics):
                    # Attempt to rate the quality of the statistic
                    quality = rate_statistic_quality(stat_text)

                    statistics.append({
                        "text": stat_text,
                        "type": infer_statistic_type(stat_text),
                        "citations": extract_citations(stat_text),
                        "quality_score": quality,
                        "year_mentioned": extract_year(stat_text)
                    })
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
        if statistics := extract_statistics(fact["source_text"]):
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

def safe_json_parse(response: Union[str, Dict[str, Any]], category: str) -> Dict[str, Any]:
    """Safely parse JSON response with enhanced error handling and cleanup."""
    try:
        # If already a dict, return it
        if isinstance(response, dict):
            return response
            
        # Clean up the response string
        if isinstance(response, str):
            # Remove any markdown code blocks
            response = re.sub(r'```(?:json)?\s*|\s*```', '', response)
            
            # Check checkpoint with TTL
            cache_key = f"json_parse_{hash(response)}"
            if cached_state := memory_saver.get(RunnableConfig(configurable={"thread_id": cache_key})):
                cached_result = cached_state.get("result")
                if isinstance(cached_result, dict) and cached_result:
                    timestamp = datetime.fromisoformat(cached_state.get("timestamp", ""))
                    if (datetime.now() - timestamp).total_seconds() < 3600:  # 1 hour TTL
                        return cached_result
            
            # Remove any leading/trailing whitespace
            response = response.strip()
            
            # Handle empty or invalid responses
            if not response or response in ['{}', '[]', 'null']:
                return get_default_extraction_result(category)
                
            # Handle single-quoted strings
            response = response.replace("'", '"')
            
            # Handle trailing commas
            response = re.sub(r',(\s*[}\]])', r'\1', response)
            
            # Handle missing braces
            if not response.startswith('{'):
                response = '{' + response
            if not response.endswith('}'):
                response = response + '}'
                
            # Handle quoted keys without values
            response = re.sub(r'"(\w+)":\s*"', r'"\1": []', response)
            
            # Find JSON object in text
            json_text = find_json_object(response)
            if not json_text:
                warning_highlight("No valid JSON object found in response")
                return get_default_extraction_result(category)
                
            # Parse JSON
            try:
                parsed = json.loads(json_text)
            except json.JSONDecodeError as e:
                warning_highlight(f"JSON decode error: {str(e)}")
                return get_default_extraction_result(category)
                
            # Validate and clean up the parsed structure
            result = get_default_extraction_result(category)
            
            # Copy valid fields from parsed result
            for key, value in parsed.items():
                if key in result:
                    if isinstance(value, list) and isinstance(result[key], list):
                        result[key] = value
                    elif isinstance(value, dict) and isinstance(result[key], dict):
                        result[key].update(value)
                    elif isinstance(value, (int, float)) and key in ['relevance_score', 'confidence_score']:
                        result[key] = float(value)
                        
            # Save to checkpoint with TTL
            memory_saver.save(RunnableConfig(configurable={
                "thread_id": cache_key,
                "data": {
                    "result": result,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }))
            
            return result
            
    except Exception as e:
        error_highlight(f"Error parsing JSON: {str(e)}")
        return get_default_extraction_result(category)

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
    """Extract information from content for a specific category with enhanced statistics focus.
    
    Args:
        content: The content to extract information from
        url: URL of the content
        title: Title of the content
        category: Research category
        original_query: Original search query
        prompt_template: Template for extraction prompt
        extraction_model: Model to use for extraction
        config: Optional configuration
        
    Returns:
        Tuple of (extracted facts, relevance score)
    """
    if not content or not url or not is_valid_url(url):
        warning_highlight(f"Invalid content or URL for extraction: {url}")
        return [], 0.0

    info_highlight(f"Extracting from {url} for {category}")

    try:
        # Preprocess content to reduce size
        content = preprocess_content(content, url)

        # Apply extraction prompt template
        extraction_prompt = prompt_template.format(
            query=original_query,
            url=url,
            content=content
        )

        # If content is too large, chunk it and process each chunk
        if len(content) > 40000:  # Threshold for chunking
            info_highlight(f"Content too large ({len(content)} chars), chunking...")
            chunks = chunk_text(content)

            all_statistics = []
            chunk_results = []

            for chunk_idx, chunk in enumerate(chunks):
                info_highlight(f"Processing chunk {chunk_idx + 1}/{len(chunks)}")
                chunk_prompt = prompt_template.format(
                    query=original_query,
                    url=url,
                    content=chunk
                )

                # Extract from this chunk
                chunk_response = await extraction_model(
                    messages=[{"role": "human", "content": chunk_prompt}],
                    config=config
                )

                if chunk_result := safe_json_parse(chunk_response, category):
                    # Extract statistics from this chunk directly
                    chunk_statistics = extract_statistics(chunk)
                    all_statistics.extend(chunk_statistics)

                    # Add to chunk results for merging
                    chunk_results.append(chunk_result)

            # Merge results from all chunks
            extraction_result = merge_chunk_results(chunk_results, category)

            # Add all statistics to the merged result
            if all_statistics:
                extraction_result["statistics"] = all_statistics
        else:
            # Process single chunk
            model_response = await extraction_model(
                messages=[{"role": "human", "content": extraction_prompt}],
                config=config
            )

            # Handle model response
            if isinstance(model_response, str):
                # Clean up the response
                model_response = model_response.strip()
                
                # If it starts with a quoted key, wrap it in braces
                if model_response.startswith('"') or model_response.startswith('\n  "'):
                    # Extract the key name
                    match = re.search(r'"(extracted_[^"]+)"', model_response)
                    if match:
                        key = match.group(1)
                        model_response = f'{{"{key}": []}}'
                    else:
                        model_response = f'{{"extracted_facts": []}}'
                
                # If it's an error message, use default
                if model_response == "' and ending with '" or model_response == '" and ending with "':
                    model_response = '{"extracted_facts": []}'

            # Safely parse the model response
            extraction_result = safe_json_parse(model_response, category)

            if statistics := extract_statistics(content):
                extraction_result["statistics"] = statistics

        # Get relevance score
        relevance_score = extraction_result.get("relevance_score", 0.0)

        # Get extracted facts based on category
        facts = _get_category_facts(category, extraction_result)

        # Enrich each fact with additional context
        enriched_facts = [
            enrich_extracted_fact(fact, url, title)
            for fact in facts
        ]

        # Sort facts by confidence score
        sorted_facts = sorted(
            enriched_facts, 
            key=lambda x: x.get("confidence_score", 0), 
            reverse=True
        )

        return sorted_facts, relevance_score

    except Exception as e:
        error_highlight(f"Error extracting from {url}: {str(e)}")
        return [], 0.0

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