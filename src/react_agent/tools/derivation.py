import asyncio
from typing import Any, Dict, List, Optional

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from react_agent.utils.extraction import (
    extract_category_information,
    extract_citations,
    extract_statistics,
)
from react_agent.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# Pydantic models for input validation
class ExtractStatisticsInput(BaseModel):
    text: str = Field(..., description="The text to extract statistics from")
    url: str = Field(default="", description="Optional source URL")
    source_title: str = Field(default="", description="Optional source title")

class ExtractCategoryInfoInput(BaseModel):
    content: str = Field(..., description="The raw content to extract information from")
    url: str = Field(..., description="The source URL")
    title: str = Field(..., description="The source title")
    category: str = Field(..., description="The extraction category (e.g., 'market_dynamics')")
    original_query: str = Field(..., description="The original search query")
    prompt_template: Optional[str] = Field(default=None, description="Optional custom prompt template")

# Tool implementations
class StatisticsExtractionTool(BaseTool):
    name = "statistics_extractor"
    description = "Extract statistics and numerical data from text with metadata"
    args_schema = ExtractStatisticsInput
    
    def _run(self, text: str, url: str = "", source_title: str = "") -> List[Dict[str, Any]]:
        """Extract statistics from text synchronously."""
        return extract_statistics(text, url, source_title)
        
    async def _arun(self, text: str, url: str = "", source_title: str = "") -> List[Dict[str, Any]]:
        """Extract statistics from text asynchronously."""
        return extract_statistics(text, url, source_title)

class CitationExtractionTool(BaseTool):
    name = "citation_extractor"
    description = "Extract citation information from text"
    
    def _run(self, text: str) -> List[Dict[str, str]]:
        """Extract citations from text synchronously."""
        return extract_citations(text)
        
    async def _arun(self, text: str) -> List[Dict[str, str]]:
        """Extract citations from text asynchronously."""
        return extract_citations(text)

class CategoryExtractionTool(BaseTool):
    name = "category_information_extractor"
    description = "Extract category-specific information from content with enhanced statistical focus"
    args_schema = ExtractCategoryInfoInput
    
    def __init__(self, extraction_model, default_prompt_template: str):
        """Initialize the category extraction tool."""
        self.extraction_model = extraction_model
        self.default_prompt_template = default_prompt_template
        super().__init__()
    
    def _run(self, content: str, url: str, title: str, category: str, 
             original_query: str, prompt_template: Optional[str] = None) -> Dict[str, Any]:
        """Extract category information synchronously."""
        template = prompt_template or self.default_prompt_template
        
        loop = asyncio.get_event_loop()
        facts, relevance = loop.run_until_complete(extract_category_information(
            content=content,
            url=url,
            title=title,
            category=category,
            original_query=original_query,
            prompt_template=template,
            extraction_model=self.extraction_model
        ))
        return {"facts": facts, "relevance": relevance}
    
    async def _arun(self, content: str, url: str, title: str, category: str, 
                   original_query: str, prompt_template: Optional[str] = None) -> Dict[str, Any]:
        """Extract category information asynchronously."""
        template = prompt_template or self.default_prompt_template
        
        facts, relevance = await extract_category_information(
            content=content,
            url=url,
            title=title,
            category=category,
            original_query=original_query,
            prompt_template=template,
            extraction_model=self.extraction_model
        )
        return {"facts": facts, "relevance": relevance}