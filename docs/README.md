# React Agent Documentation

This directory contains documentation and rules for the `react_agent` project, focusing on standardizing tool implementations and LangGraph integration.

## Documentation Files

- [LangGraph Tool Calling Integration Guide](langgraph-tool-calling.md): Comprehensive guide on integrating custom tools with LangGraph.
- [LangGraph Tool Calling Rules](langgraph_rules.md): Concise rules for implementing tools that work seamlessly with LangGraph.
- [Updated Jina Tool Example](updated_jina_tool_example.py): Example implementation showing how to update a Jina tool to follow the rules.

## Using These Rules

The documentation in this directory defines the standards and patterns to follow when implementing or modifying tools in this project. When working with tools:

1. Use the **LangGraph Tool Calling Rules** as a checklist for tool implementations
2. Refer to the **Integration Guide** for detailed explanations and rationale
3. Use the **Example Implementation** as a template for your own tools

## Key Principles

The rules prioritize:

1. **Clear separation** between LLM-controlled parameters and runtime parameters
2. **Consistent error handling** to maintain graph state integrity
3. **Type safety** with well-defined schemas for inputs and outputs
4. **Structured responses** that follow project-wide conventions

## Project Structure

```
react_agent/
├── graphs/           # LangGraph implementations
├── prompts/          # Prompt templates
├── tools/            # Tool implementations (including jina.py)
│   ├── jina.py       # Jina AI service integrations
│   └── ...
├── utils/            # Utility modules
└── state.py          # State definitions
```

## Implementing New Tools

1. Create a new file in `src/react_agent/tools/` if needed
2. Define your tools following the patterns in [langgraph_rules.md](langgraph_rules.md)
3. Expose both synchronous and asynchronous interfaces
4. Use the proper annotations for parameters
5. Handle errors consistently

## Integration with Graphs

1. Import your tools in the appropriate graph file
2. Create a `ToolNode` with your tool list
3. Add the node to your graph
4. Configure appropriate routing based on tool results

## Testing Tools

When writing tests for tools:

1. Test both synchronous and asynchronous interfaces
2. Test error handling for various failure modes
3. Test state injection if your tool uses state
4. Verify response format consistency 