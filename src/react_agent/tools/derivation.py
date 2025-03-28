import asyncio
from typing import Any, Callable, Dict, List, Literal, Union
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import InjectedToolArg
from langgraph.prebuilt import InjectedState, InjectedStore, ToolNode
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from react_agent.utils.content import should_skip_content, validate_content
from react_agent.utils.extraction import (
    enrich_extracted_fact,
    extract_category_information,
    extract_citations,
    extract_statistics,
)
from react_agent.utils.logging import get_logger
from react_agent.utils.statistics import (
    assess_authoritative_sources,
    assess_fact_consistency,
    assess_synthesis_quality,
    calculate_category_quality_score,
    calculate_overall_confidence,
    perform_statistical_validation,
)

# Initialize logger
logger = get_logger(__name__)


# Original Pydantic models for backward compatibility
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
    prompt_template: str | None = Field(default=None, description="Optional custom prompt template")


# New consolidated Pydantic models
class ExtractionInput(BaseModel):
    """Input model for the ExtractionTool."""
    operation: Literal["statistics", "citations", "category", "enrich"] = Field(
        ..., description="The extraction operation to perform"
    )
    text: str | None = Field(
        default=None, description="The text to extract from"
    )
    url: str | None = Field(
        default="", description="Optional source URL"
    )
    source_title: str | None = Field(
        default="", description="Optional source title"
    )
    category: str | None = Field(
        default=None, description="The extraction category (e.g., 'market_dynamics')"
    )
    original_query: str | None = Field(
        default=None, description="The original search query"
    )
    prompt_template: str | None = Field(
        default=None, description="Optional custom prompt template"
    )
    fact: Dict[str, Any] | None = Field(
        default=None, description="Fact to enrich"
    )
    extraction_model: Any | None = Field(
        default=None, description="Model to use for extraction"
    )


class StatisticsAnalysisInput(BaseModel):
    """Input model for the StatisticsAnalysisTool."""
    operation: Literal["quality", "sources", "confidence", "synthesis", "validate", "consistency"] = Field(
        ..., description="The statistics analysis operation to perform"
    )
    category: str | None = Field(
        default=None, description="The category to analyze"
    )
    facts: List[Dict[str, Any]] | None = Field(
        default=None, description="Extracted facts to analyze"
    )
    sources: List[Dict[str, Any]] | None = Field(
        default=None, description="Sources to analyze"
    )
    thresholds: Dict[str, Any] | None = Field(
        default=None, description="Thresholds for quality assessment"
    )
    category_scores: Dict[str, float] | None = Field(
        default=None, description="Category scores for confidence calculation"
    )
    synthesis_quality: float | None = Field(
        default=None, description="Synthesis quality score"
    )
    validation_score: float | None = Field(
        default=None, description="Validation score"
    )
    synthesis: Dict[str, Any] | None = Field(
        default=None, description="Synthesis data to assess"
    )


class ResearchValidationInput(BaseModel):
    """Input model for the ResearchValidationTool."""
    operation: Literal["content", "facts", "sources", "overall"] = Field(
        ..., description="The validation operation to perform"
    )
    content: str | None = Field(
        default=None, description="Content to validate"
    )
    url: str | None = Field(
        default=None, description="URL to validate"
    )
    facts: List[Dict[str, Any]] | None = Field(
        default=None, description="Facts to cross-validate"
    )
    sources: List[Dict[str, Any]] | None = Field(
        default=None, description="Sources to assess quality"
    )
    scores: Dict[str, float] | None = Field(
        default=None, description="Scores to calculate overall score"
    )


# Original tool implementations (kept for backward compatibility)
class StatisticsExtractionTool(BaseTool):
    name: str = "statistics_extractor"
    description: str = "Extract statistics and numerical data from text with metadata"
    args_schema: type[ExtractStatisticsInput] = ExtractStatisticsInput
    name: str = "statistics_extractor"
    description: str = "Extract statistics and numerical data from text with metadata"
    args_schema: type[ExtractStatisticsInput] = ExtractStatisticsInput
    
    def _run(self, text: str, url: str = "", source_title: str = "") -> List[Dict[str, Any]]:
        """Extract statistics from text synchronously.
        
        Note: This tool is maintained for backward compatibility.
        Consider using ExtractionTool with operation="statistics" instead.
        """
        return extract_statistics(text, url, source_title)
        
    async def _arun(self, text: str, url: str = "", source_title: str = "") -> List[Dict[str, Any]]:
        """Extract statistics from text asynchronously.
        
        Note: This tool is maintained for backward compatibility.
        Consider using ExtractionTool with operation="statistics" instead.
        """
        return extract_statistics(text, url, source_title)


class CitationExtractionTool(BaseTool):
    name: str = "citation_extractor"
    description: str = "Extract citation information from text"
    name: str = "citation_extractor"
    description: str = "Extract citation information from text"
    
    def _run(self, text: str) -> List[Dict[str, str]]:
        """Extract citations from text synchronously.
        
        Note: This tool is maintained for backward compatibility.
        Consider using ExtractionTool with operation="citations" instead.
        """
        return extract_citations(text)
        
    async def _arun(self, text: str) -> List[Dict[str, str]]:
        """Extract citations from text asynchronously.
        
        Note: This tool is maintained for backward compatibility.
        Consider using ExtractionTool with operation="citations" instead.
        """
        return extract_citations(text)


class CategoryExtractionTool(BaseTool):
    name: str = "category_information_extractor"
    description: str = "Extract category-specific information with enhanced statistical focus"
    args_schema: type[ExtractCategoryInfoInput] = ExtractCategoryInfoInput
    
    def __init__(self, extraction_model: Any, default_prompt_template: str = ""):
        """Initialize the category extraction tool."""
        super().__init__()
        self._extraction_model = extraction_model
        self._default_prompt_template = default_prompt_template
    
    @property
    def extraction_model(self) -> Any:
        """Get the extraction model used by this tool."""
        return self._extraction_model
        
    @property
    def default_prompt_template(self) -> str:
        """Get the default prompt template used by this tool."""
        return self._default_prompt_template
    
    def _run(self, content: str, url: str, title: str, category: str, 
             original_query: str, prompt_template: str | None = None) -> Dict[str, Any]:
        """Extract category information synchronously.
        
        Note: This tool is maintained for backward compatibility.
        Consider using ExtractionTool with operation="category" instead.
        """
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
                   original_query: str, prompt_template: str | None = None) -> Dict[str, Any]:
        """Extract category information asynchronously.
        
        Note: This tool is maintained for backward compatibility.
        Consider using ExtractionTool with operation="category" instead.
        """
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


# New consolidated tool implementations
class ExtractionTool(BaseTool):
    name: str = "extraction_tool"
    description: str = "Extract information from text including statistics, citations, and data"
    args_schema: type[ExtractionInput] = ExtractionInput
    
    def _run(
        self, 
        operation: Literal["statistics", "citations", "category", "enrich"],
        text: str | None = None,
        url: str = "",
        source_title: str = "",
        category: str | None = None,
        original_query: str | None = None,
        prompt_template: str | None = None,
        fact: Dict[str, Any] | None = None,
        extraction_model: Any | None = None,
    ) -> Any:
        """Run the appropriate extraction function based on operation."""
        if operation == "statistics":
            if text is None:
                raise ValueError("Text is required for statistics extraction")
            return extract_statistics(text, url, source_title)
            
        elif operation == "citations":
            if text is None:
                raise ValueError("Text is required for citation extraction")
            return extract_citations(text)
            
        elif operation == "category":
            if text is None or category is None or original_query is None:
                raise ValueError("Text, category, and original_query are required for category extraction")
            if extraction_model is None:
                raise ValueError("extraction_model is required for category extraction")
                
            # For synchronous execution, we need to run the async function in an event loop
            loop = asyncio.get_event_loop()
            facts, relevance = loop.run_until_complete(extract_category_information(
                content=text,
                url=url,
                title=source_title,
                category=category,
                original_query=original_query,
                prompt_template=prompt_template or "",
                extraction_model=extraction_model
            ))
            return {"facts": facts, "relevance": relevance}
            
        elif operation == "enrich":
            if fact is None:
                raise ValueError("Fact is required for enrichment")
            return enrich_extracted_fact(fact, url, source_title)
            
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def _arun(
        self, 
        operation: Literal["statistics", "citations", "category", "enrich"],
        text: str | None = None,
        url: str = "",
        source_title: str = "",
        category: str | None = None,
        original_query: str | None = None,
        prompt_template: str | None = None,
        fact: Dict[str, Any] | None = None,
        extraction_model: Any | None = None,
        config: RunnableConfig | None = None,
        state: Annotated[Dict[str, Any], InjectedState] = None,
        store: Annotated[BaseStore, InjectedStore] = None,
    ) -> Any:
        """Run the appropriate extraction function asynchronously."""
        if operation == "statistics":
            if text is None:
                raise ValueError("Text is required for statistics extraction")
            return extract_statistics(text, url, source_title)
            
        elif operation == "citations":
            if text is None:
                raise ValueError("Text is required for citation extraction")
            return extract_citations(text)
            
        elif operation == "category":
            if text is None or category is None or original_query is None:
                raise ValueError("Text, category, and original_query are required for category extraction")
            if extraction_model is None:
                raise ValueError("extraction_model is required for category extraction")
                
            facts, relevance = await extract_category_information(
                content=text,
                url=url,
                title=source_title,
                category=category,
                original_query=original_query,
                prompt_template=prompt_template or "",
                extraction_model=extraction_model,
                config=config
            )
            return {"facts": facts, "relevance": relevance}
            
        elif operation == "enrich":
            if fact is None:
                raise ValueError("Fact is required for enrichment")
            return enrich_extracted_fact(fact, url, source_title)
            
        else:
            raise ValueError(f"Unknown operation: {operation}")


class StatisticsAnalysisTool(BaseTool):
    name: str = "statistics_analyzer"
    description: str = "Analyze statistics and calculate quality scores"
    args_schema: type[StatisticsAnalysisInput] = StatisticsAnalysisInput
    
    def _run(
        self, 
        operation: Literal["quality", "sources", "confidence", "synthesis", "validate", "consistency"],
        **kwargs
    ) -> Any:
        """Run the appropriate statistics analysis function based on operation."""
        if operation == "quality":
            required = ["category", "facts", "sources", "thresholds"]
            self._validate_required_params(required, kwargs)
            return calculate_category_quality_score(
                kwargs["category"], 
                kwargs["facts"], 
                kwargs["sources"], 
                kwargs["thresholds"]
            )
            
        elif operation == "sources":
            self._validate_required_params(["sources"], kwargs)
            return assess_authoritative_sources(kwargs["sources"])
            
        elif operation == "confidence":
            required = ["category_scores", "synthesis_quality", "validation_score"]
            self._validate_required_params(required, kwargs)
            return calculate_overall_confidence(
                kwargs["category_scores"],
                kwargs["synthesis_quality"],
                kwargs["validation_score"]
            )
            
        elif operation == "synthesis":
            self._validate_required_params(["synthesis"], kwargs)
            return assess_synthesis_quality(kwargs["synthesis"])
            
        elif operation == "validate":
            self._validate_required_params(["facts"], kwargs)
            return perform_statistical_validation(kwargs["facts"])
            
        elif operation == "consistency":
            self._validate_required_params(["facts"], kwargs)
            return assess_fact_consistency(kwargs["facts"])
            
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def _arun(
        self, 
        operation: Literal["quality", "sources", "confidence", "synthesis", "validate", "consistency"],
        **kwargs
    ) -> Any:
        """Run the appropriate statistics analysis function asynchronously."""
        # For these synchronous functions, we can just call the synchronous implementation
        return self._run(operation, **kwargs)
        
    def _validate_required_params(self, required: List[str], params: Dict[str, Any]) -> None:
        """Validate that required parameters are present."""
        missing = [param for param in required if param not in params or params[param] is None]
        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")


class ResearchValidationTool(BaseTool):
    name: str = "research_validator"
    description: str = "Validate research content, facts, and sources"
    args_schema: type[ResearchValidationInput] = ResearchValidationInput
    
    def _run(
        self, 
        operation: Literal["content", "facts", "sources", "overall"],
        **kwargs
    ) -> Any:
        """Run the appropriate validation function based on operation."""
        if operation == "content":
            required = ["content", "url"]
            self._validate_required_params(required, kwargs)
            
            # First check if we should skip this content
            if should_skip_content(kwargs["url"]):
                return {"valid": False, "reason": "Content type should be skipped"}
                
            # Then validate the content
            is_valid = validate_content(kwargs["content"])
            return {"valid": is_valid}
            
        elif operation == "facts":
            self._validate_required_params(["facts"], kwargs)
            consistency_score = assess_fact_consistency(kwargs["facts"])
            return {"consistency_score": consistency_score}
            
        elif operation == "sources":
            self._validate_required_params(["sources"], kwargs)
            # Return quality scores for sources
            quality_scores = [
                {"url": source.get("url", ""), 
                 "quality": source.get("quality_score", 0.5)}
                for source in kwargs["sources"]
            ]
            return {"quality_scores": quality_scores}
            
        elif operation == "overall":
            self._validate_required_params(["scores"], kwargs)
            # Calculate overall score from individual scores
            scores = kwargs["scores"]
            avg_score = sum(scores.values()) / len(scores) if scores else 0.0
            return {"overall_score": avg_score}
            
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def _arun(
        self, 
        operation: Literal["content", "facts", "sources", "overall"],
        **kwargs
    ) -> Any:
        """Run the appropriate validation function asynchronously."""
        # For these synchronous functions, we can just call the synchronous implementation
        return self._run(operation, **kwargs)
        
    def _validate_required_params(self, required: List[str], params: Dict[str, Any]) -> None:
        """Validate that required parameters are present."""
        missing = [param for param in required if param not in params or params[param] is None]
        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")


def create_derivation_toolnode(
    include_tools: List[str] | None = None
) -> ToolNode:
    """Create a ToolNode with the derivation tools.
    
    Args:
        include_tools: Optional list of tool names to include.
                      If None, all tools are included.
    
    Returns:
        A ToolNode with the specified tools.
    """
    # Initialize all tools
    extraction_tool = ExtractionTool()
    statistics_tool = StatisticsAnalysisTool()
    validation_tool = ResearchValidationTool()
    
    # Map tool names to instances
    all_tools = {
        "extraction": extraction_tool,
        "statistics": statistics_tool,
        "validation": validation_tool
    }
    
    # Filter tools if specified
    if include_tools:
        tools = [all_tools[name] for name in include_tools if name in all_tools]
    else:
        tools = list(all_tools.values())
    
    # Create the base ToolNode
    base_tool_node = ToolNode(tools)
    
    # Create a wrapper function to ensure the last message is an AIMessage
    async def tool_node_wrapper(state: Dict[str, Any], config: RunnableConfig | None = None) -> Dict[str, Any]:
        try:
            # Try to run the ToolNode as is
            return await base_tool_node.ainvoke(state, config)
        except ValueError as e:
            # Check if it's the specific error we're trying to handle
            if "Last message is not an AIMessage" in str(e):
                logger.warning("Caught 'Last message is not an AIMessage' error, attempting to fix")
                
                # If we have messages, ensure the last one is an AIMessage
                if "messages" in state and state["messages"]:
                    # Get the last message
                    last_message = state["messages"][-1]
                    
                    # If it's not an AIMessage, convert it
                    if not isinstance(last_message, AIMessage):
                        # Create a new AIMessage with the content from the last message
                        if hasattr(last_message, "content"):
                            content = last_message.content
                        elif isinstance(last_message, dict) and "content" in last_message:
                            content = last_message["content"]
                        else:
                            content = str(last_message)
                        
                        # Replace the last message with an AIMessage
                        state["messages"] = list(state["messages"][:-1]) + [AIMessage(content=content)]
                        logger.info("Converted last message to AIMessage")
                        
                        # Try again with the fixed state
                        return await base_tool_node.ainvoke(state, config)
            
            # If it's a different error or we couldn't fix it, re-raise
            raise e
    
    # Create a new ToolNode that uses our wrapper function
    wrapped_node = RunnableLambda(tool_node_wrapper)
    
    return wrapped_node