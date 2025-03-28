"""Type definitions for the React Agent research module.

This module defines the types used by the research graph and related components.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union, Sequence


class TextContent(TypedDict):
    """Text content extracted from a source."""
    content: str
    url: Optional[str]
    title: Optional[str]


class WebContent(TextContent):
    """Content extracted from a web source."""
    source: str
    published_date: Optional[str]


class DocumentContent(TextContent):
    """Content extracted from a document."""
    document_type: str
    metadata: Dict[str, Any]


class ContentChunk(TypedDict):
    """A chunk of content from a document or web page."""
    content: str
    source_url: str
    source_title: Optional[str]


class DocumentPage(TypedDict):
    """A page from a document."""
    page_number: int
    content: str
    metadata: Dict[str, Any]


class ProcessedDocument(TypedDict):
    """A processed document with extracted content."""
    url: str
    title: str
    content: str
    pages: List[DocumentPage]
    metadata: Dict[str, Any]


class SearchResult(TypedDict):
    """A search result from a search engine."""
    url: str
    title: str
    snippet: str
    source: str
    quality_score: float
    published_date: Optional[str]
    content_type: str


class EnhancedSearchResult(SearchResult):
    """Enhanced search result with additional metadata for better tracking."""
    retrieval_timestamp: str
    category: str
    query: str
    success: bool
    retry_count: int
    processing_time_ms: float
    source_quality: float
    relevance_score: float


class ExtractedFact(TypedDict):
    """A fact extracted from content."""
    fact: str
    source_url: str
    source_title: str
    category: str
    confidence: float
    data: Dict[str, Any]


class Statistics(TypedDict):
    """Statistics extracted from content."""
    statistic: str
    value: Union[str, float, int]
    context: str
    source_url: Optional[str]
    source_title: Optional[str]
    confidence: float
    unit: Optional[str]
    date: Optional[str]
    category: Optional[str]


class ResearchSettings(TypedDict):
    """Settings for the research process."""
    max_sources: int
    recency_threshold_days: int
    min_quality_score: float
    extraction_model: str
    search_type: Literal["general", "authoritative", "recent", "comprehensive", "technical"]