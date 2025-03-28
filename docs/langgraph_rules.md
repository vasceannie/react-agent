# LangGraph Tool Calling Rules

## Introduction

These rules define the standards for implementing and integrating tools with LangGraph in the `react_agent` project. The focus is on ensuring consistent patterns for handling complex data types and maintaining compatibility with LangGraph's `ToolNode`.

## Rules

### Tool Definition Rules

1. **Consistent Base Class**
   - ALL tools MUST inherit from `BaseTool` from the `langchain.tools` package
   - ALL tools MUST implement both `run` and `_arun` methods

2. **Schema Exposure**
   - ALL tools MUST define an `args_schema` class variable when complex inputs are required
   - PREFER using `TypedDict` for `args_schema` over Pydantic models for LLM-facing parameters

3. **Parameter Separation**
   - ALWAYS use the `InjectedToolArg` annotation for parameters that should NOT be exposed to the LLM
   - NEVER allow sensitive information to be supplied by the LLM

4. **Method Signatures**
   - ALL tool methods MUST accept `**kwargs` to handle additional parameters
   - ALL `_arun` methods MUST be `async def` functions

### Type Safety Rules

5. **Input Validation**
   - ALWAYS use Pydantic models for internal request validation
   - ALWAYS validate inputs before making external API calls

6. **Output Consistency**
   - ALL tools MUST return outputs in a consistent format
   - PREFER structured data types (dicts with defined schemas) for responses
   - DOCUMENT the structure of output data with TypedDict classes

### Integration Rules

7. **ToolNode Integration**
   - ALWAYS define tools in a list that can be passed to `ToolNode` constructor
   - ALWAYS ensure tool names are unique across the project

8. **Error Handling**
   - ALWAYS return errors as structured data with at least `{"error": {"message": str}}` format
   - NEVER raise exceptions from tools that integrate with LangGraph - catch and structure them
   - ALWAYS include type information in error responses: `{"error": {"type": str, "message": str}}`

9. **State Management**
   - WHEN tools need access to graph state, use `state: Annotated[Dict[str, Any], InjectedState]` parameter
   - NEVER mutate graph state directly from tools (return changes to be applied)

## Example Pattern

```python
# Definition of schema exposed to LLM
class SearchInput(TypedDict):
    """Input for search tool."""
    query: Annotated[str, "The search query to execute"]
    max_results: Annotated[int, "Maximum number of results to return"]

# Internal request model
class SearchRequest(BaseModel):
    """Internal request model."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    query: str = Field(...)
    max_results: PositiveInt = Field(default=5)
    search_provider: str = Field(default="default")

# Tool implementation
class SearchTool(BaseTool):
    """Tool for performing searches."""
    name: str = "SearchTool"
    description: str = "Search for information on the web"
    args_schema: Type[SearchInput] = SearchInput
    
    async def _arun(
        self,
        query: str,
        max_results: int = 5,
        state: Annotated[Dict[str, Any], InjectedState] = None,
        **kwargs: Any
    ) -> Any:
        """Execute search asynchronously."""
        try:
            # Create internal request
            request = SearchRequest(
                query=query, 
                max_results=max_results,
                search_provider=self._get_provider(state)
            )
            
            # Execute search
            results = await self._execute_search(request)
            
            # Return structured response
            return {
                "results": results,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "count": len(results)
                }
            }
        except Exception as e:
            return {"error": {"type": "search_error", "message": str(e)}}
```

## Project Integration

To use tools with the LangGraph agent in this project:

1. Define your tools following the rules above
2. Create a `ToolNode` instance with your tools:
   ```python
   from langgraph.prebuilt import ToolNode
   tool_node = ToolNode([tool1, tool2, ...])
   ```
3. Add the node to your graph:
   ```python
   builder.add_node("tools", tool_node)
   ```
4. Configure routing based on tool outputs:
   ```python
   builder.add_conditional_edges(
       "tools",
       route_after_tools,
       {"next_node": "next_node", ...}
   )
   ``` 