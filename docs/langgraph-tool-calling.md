# LangGraph Tool Calling Integration Guide

## Overview

This document provides guidelines for integrating custom tools with LangGraph's tool calling capabilities, with a focus on handling complex data types. The document specifically addresses the `jina.py` tools implementation in the `react_agent` project and provides best practices for ensuring compatibility with LangGraph's `ToolNode`.

## Table of Contents

1. [Current Architecture](#current-architecture)
2. [LangGraph Tool Integration Requirements](#langgraph-tool-integration-requirements)
3. [Handling Complex Data Types](#handling-complex-data-types)
4. [Tool Implementation Patterns](#tool-implementation-patterns)
5. [Error Handling](#error-handling)
6. [Integration Examples](#integration-examples)

## Current Architecture

The `react_agent` project implements several tools for interacting with Jina AI services in `src/react_agent/tools/jina.py`. These tools use the Langchain `BaseTool` class and provide both synchronous `run` and asynchronous `_arun` methods.

Each tool uses Pydantic models to define request objects for input validation:

```python
class EmbeddingsRequest(BaseModel):
    """Request model for Jina's Embeddings API."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(
        ...,
        description="Identifier of the model to use."
    )
    input: List[str] = Field(
        ...,
        min_length=1,
        description="Array of input strings to be embedded."
    )
    # ... more fields
```

And TypedDict objects to define response structure:

```python
class EmbeddingData(TypedDict):
    embedding: List[float]
    index: int
    object: str

class EmbeddingsResponse(TypedDict):
    data: List[EmbeddingData]
    model: str
    object: str
    usage: Dict[str, int]
```

## LangGraph Tool Integration Requirements

LangGraph's `ToolNode` is designed to use LangChain's tool interface for execution. However, it requires specific patterns to work effectively, especially when dealing with complex data types:

1. **Message-Based State**: LangGraph maintains a message history in the state under the `messages` key.
2. **Tool Invocation Format**: The `ToolNode` expects to find tool calls in the most recent `AIMessage` in the state.
3. **Complex Parameters**: When tools require complex structured inputs, special handling is needed.

## Handling Complex Data Types

When integrating tools that require complex data types with LangGraph, there are two primary approaches:

### 1. Using InjectedToolArg for Runtime Values

For parameters that should not be exposed to the LLM or require runtime values:

```python
from typing import Annotated
from langchain_core.tools import InjectedToolArg
from langgraph.prebuilt import InjectedState

async def _arun(
    self,
    tool_input: Union[str, Dict[str, Any]],
    request: Annotated[EmbeddingsRequest, InjectedToolArg] = None,
    state: Annotated[Dict[str, Any], InjectedState] = None,
    **kwargs: Any
) -> Any:
    # Implementation
```

### 2. Schema Definitions for LLM-Controlled Parameters

For parameters that the LLM should control, define clear schemas:

```python
# For parameter schemas visible to the LLM:
class EmbeddingsInput(BaseModel):
    """Input for the embeddings tool."""
    model: str = Field(
        description="Identifier of the model to use."
    )
    input_text: List[str] = Field(
        description="Text to generate embeddings for."
    )
```

### 3. Converting Between Schemas

Convert LLM-provided inputs to your internal request objects:

```python
def _convert_to_request(llm_input: EmbeddingsInput) -> EmbeddingsRequest:
    """Convert LLM input to API request object."""
    return EmbeddingsRequest(
        model=llm_input.model,
        input=llm_input.input_text
    )
```

## Tool Implementation Patterns

### Pattern 1: Schema Conversion

```python
class EnhancedEmbeddingsTool(BaseTool):
    name: str = "EmbeddingsTool"
    description: str = "Convert text to vector embeddings."
    
    # Schema exposed to LLM
    args_schema: Type[BaseModel] = EmbeddingsInput
    
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    
    async def _arun(
        self, 
        model: str,
        input_text: List[str],
        **kwargs: Any
    ) -> Any:
        """Process LLM-provided arguments directly."""
        # Convert to internal request
        request = EmbeddingsRequest(
            model=model,
            input=input_text
        )
        
        # Make API call
        return await self._make_api_call(request)
    
    async def _make_api_call(self, request: EmbeddingsRequest) -> Any:
        """Internal method for API calls."""
        # Implementation
```

### Pattern 2: Runtime Injection

```python
class ContextAwareEmbeddingsTool(BaseTool):
    name: str = "ContextAwareEmbeddingsTool"
    description: str = "Convert text to vector embeddings with context awareness."
    
    # Schema exposed to LLM
    args_schema: Type[BaseModel] = EmbeddingsInputSimplified
    
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    
    async def _arun(
        self, 
        input_text: List[str],
        state: Annotated[Dict[str, Any], InjectedState] = None,
        **kwargs: Any
    ) -> Any:
        """Process with runtime context."""
        # Get context from state
        context = self._extract_context(state)
        
        # Create full request with context
        request = EmbeddingsRequest(
            model=context.get("preferred_model", "jina-embeddings-v3"),
            input=input_text,
            normalized=context.get("normalize_embeddings", False)
        )
        
        # Make API call
        return await self._make_api_call(request)
```

## Error Handling

LangGraph requires proper error handling to maintain graph state integrity:

```python
async def _arun(self, tool_input: Union[str, Dict[str, Any]], **kwargs: Any) -> Any:
    try:
        # Tool implementation
        return successful_response
    except ValueError as e:
        # Validation errors
        return {"error": {"type": "validation", "message": str(e)}}
    except aiohttp.ClientError as e:
        # Network errors
        return {"error": {"type": "network", "message": str(e)}}
    except Exception as e:
        # Unexpected errors
        return {"error": {"type": "unexpected", "message": str(e)}}
```

## Integration Examples

### Example 1: Basic Integration with ToolNode

```python
from langgraph.prebuilt import ToolNode

# Define tools
tools = [
    EmbeddingsTool(),
    RerankerTool(),
    # Other tools
]

# Create ToolNode
tool_node = ToolNode(tools)

# Add to graph
builder = StateGraph(State)
builder.add_node("tools", tool_node)
```

### Example 2: Conditional Routing with Tool Results

```python
def route_after_tools(state: State) -> Literal["process_embeddings", "llm_chat"]:
    """Route based on which tool was called."""
    last_message = state.messages[-1]
    if not isinstance(last_message, ToolMessage):
        return "llm_chat"
        
    # Check tool name
    if last_message.name == "EmbeddingsTool":
        return "process_embeddings"
    return "llm_chat"

builder.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "process_embeddings": "embedding_processor", 
        "llm_chat": "llm_chat"
    }
)
```

By following these guidelines, you can ensure that your Jina tools integrate seamlessly with LangGraph's tool calling capabilities, providing a robust foundation for building complex AI agent workflows. 