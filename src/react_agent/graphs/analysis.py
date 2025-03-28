"""Define a data analysis pipeline using LangGraph.

Works with a chat model to perform structured data analysis tasks.
"""

import contextlib
import json
from datetime import datetime
from typing import (
    Annotated,
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    TypedDict,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from openai import AsyncClient
from pydantic import BaseModel, Field

from react_agent.utils.llm import call_model_json

# Sample prompts (assuming these are imported from somewhere else)
ANALYSIS_PLAN_PROMPT = "..."  # Your existing prompt
COMPILE_REPORT_PROMPT = "..."  # Your existing prompt
INTERPRET_RESULTS_PROMPT = "..."  # Your existing prompt

# Define message class
class AIMessage(BaseModel):
    content: str

###############################################################################
# 1) Define state using TypedDict compatible with LangGraph
###############################################################################
class AnalysisState(TypedDict, total=False):
    task: str
    data: Dict[str, Any]
    analysis_plan: Dict[str, Any]
    prepared_data: Dict[str, Any]
    analysis_results: Dict[str, Any]
    interpretations: Dict[str, Any]
    code_snippets: List[str]
    visualizations: List[Dict[str, Any]]
    messages: List[AIMessage]
    report: Dict[str, Any]
    error: Optional[Dict[str, Any]]

###############################################################################
# 2) LangGraph node functions (no decorators needed)
###############################################################################
async def formulate_analysis_plan(state: AnalysisState, config: Optional[RunnableConfig] = None) -> AnalysisState:
    """Create an analysis plan based on the task and data.
    
    Args:
        state (AnalysisState): The current state of the analysis pipeline.
        config (Optional[RunnableConfig]): Configuration for the model run.
        
    Returns:
        AnalysisState: Updated state with analysis plan or error.
    """
    task: str = state.get("task", "")
    data: Dict[str, Any] = state.get("data", {})
    
    if not task or not data:
        return {
            **state,
            "error": {"phase": "formulate_plan", "message": "Missing task or data for analysis plan"}
        }

    # Create prompt with data schema
    data_schema: Dict[str, str] = {k: type(v).__name__ for k, v in data.items()}
    prompt = f"""
    Task: {task}
    
    Available data schema:
    {json.dumps(data_schema, indent=2)}
    
    Create a detailed analysis plan including:
    1. Data preparation steps
    2. Analysis methods to apply
    3. Expected insights
    4. Visualization suggestions
    
    Format the response as a JSON object.
    """
    
    try:
        analysis_plan = await call_model_json(
            messages=[{"role": "user", "content": prompt}],
            config=config
        )
        
        # Create an AI message to record this step
        message = AIMessage(content=f"Formulated analysis plan: {json.dumps(analysis_plan, indent=2)}")
        
        # Get existing messages and add new message
        existing_messages: List[AIMessage] = state.get("messages", [])
        messages: List[AIMessage] = existing_messages + [message] if existing_messages else [message]
        
        return {
            **state,
            "analysis_plan": analysis_plan,
            "messages": messages
        }
    except Exception as e:
        return {
            **state,
            "error": {"phase": "formulate_plan", "message": str(e)}
        }

async def prepare_data(state: AnalysisState, config: Optional[RunnableConfig] = None) -> AnalysisState:
    """Prepare data for analysis by cleaning and transforming."""
    data: Dict[str, Any] = state.get("data", {})
    
    if not data:
        return {
            **state,
            "error": {"phase": "prepare_data", "message": "No data to prepare"}
        }
    
    prepared_data: Dict[str, Any] = {}
    code_snippets: List[str] = state.get("code_snippets", [])
    
    if isinstance(data, dict):
        for key, dataset in data.items():
            if isinstance(dataset, pd.DataFrame):
                df: pd.DataFrame = dataset.copy()
                df = df.dropna()
                df = df.drop_duplicates()
                for col in df.columns:
                    if df[col].dtype == "object":
                        with contextlib.suppress(ValueError):
                            df[col] = pd.to_numeric(df[col])

                prepared_data[key] = df
                snippet: str = (
                    f"# Data cleaning for {key}:\n"
                    f"# - Dropped rows with missing values\n"
                    f"# - Dropped duplicate rows\n"
                    f"# - Attempted type conversion for object columns"
                )
            else:
                prepared_data[key] = dataset
                snippet = (
                    f"# No specific preparation for {key} "
                    f"(type: {type(dataset).__name__})"
                )
            code_snippets.append(snippet)
    
    # Create an AI message to record this step
    message = AIMessage(content=f"Prepared data for analysis:\n" + "\n".join(code_snippets))
    
    # Get existing messages and add new message
    existing_messages: List[AIMessage] = state.get("messages", [])
    messages: List[AIMessage] = existing_messages + [message] if existing_messages else [message]
    
    return {
        **state,
        "prepared_data": prepared_data,
        "code_snippets": code_snippets,
        "messages": messages
    }

async def analyze_data(state: AnalysisState, config: Optional[RunnableConfig] = None) -> AnalysisState:
    """Analyze prepared data according to the analysis plan.
    
    Args:
        state (AnalysisState): The current state of the analysis pipeline.
        config (Optional[RunnableConfig]): Configuration for the model run.
        
    Returns:
        AnalysisState: Updated state with analysis results or error.
    """
    prepared_data: Dict[str, Any] = state.get("prepared_data", {})
    analysis_plan: Dict[str, Any] = state.get("analysis_plan", {})
    
    if not prepared_data:
        return {
            **state,
            "error": {"phase": "analyze_data", "message": "No prepared data to analyze"}
        }
    
    if not analysis_plan:
        return {
            **state,
            "error": {"phase": "analyze_data", "message": "No analysis plan available"}
        }

    analysis_results: Dict[str, Any] = {}
    code_snippets: List[str] = state.get("code_snippets", [])
    
    # Process each dataset according to analysis plan
    for key, dataset in prepared_data.items():
        if isinstance(dataset, pd.DataFrame):
            # Generate analysis code with LLM
            prompt = f"""
            Dataset: {key}
            Columns: {list(dataset.columns)}
            Analysis plan: {json.dumps(analysis_plan.get(key, {}), indent=2)}
            
            Generate Python code to analyze this dataset according to the plan.
            Include descriptive statistics, correlations, and any relevant statistical tests.
            """
            
            try:
                analysis_code_response = await call_model_json(
                    messages=[{"role": "user", "content": prompt}],
                    config=config
                )
                analysis_code = analysis_code_response.get("code", "")
                code_snippets.append(f"# Analysis code for {key}:\n{analysis_code}")
                
                # Execute analysis code safely
                try:
                    local_vars = {"df": dataset, "np": np, "pd": pd}
                    exec(analysis_code, {}, local_vars)
                    analysis_results[key] = local_vars.get("results", {})
                except Exception as e:
                    analysis_results[key] = {"error": str(e)}
            except Exception as e:
                analysis_results[key] = {"error": str(e)}
    
    # Create an AI message to record this step
    message = AIMessage(content=f"Completed data analysis with {len(analysis_results)} datasets analyzed")
    
    # Get existing messages and add new message
    existing_messages: List[AIMessage] = state.get("messages", [])
    messages: List[AIMessage] = existing_messages + [message] if existing_messages else [message]
    
    return {
        **state,
        "analysis_results": analysis_results,
        "code_snippets": code_snippets,
        "messages": messages
    }

async def interpret_results(state: AnalysisState, config: Optional[RunnableConfig] = None) -> AnalysisState:
    """Interpret the analysis results in the context of the original task.
    
    Args:
        state (AnalysisState): The current state of the analysis pipeline.
        config (Optional[RunnableConfig]): Configuration for the model run.
        
    Returns:
        AnalysisState: Updated state with interpretations or error.
    """
    task: str = state.get("task", "")
    analysis_results: Dict[str, Any] = state.get("analysis_results", {})
    analysis_plan: Dict[str, Any] = state.get("analysis_plan", {})
    
    if not analysis_results:
        return {
            **state,
            "error": {"phase": "interpret_results", "message": "No analysis results to interpret"}
        }
    
    # Create prompt with task and results
    prompt: str = INTERPRET_RESULTS_PROMPT.format(
        task=task,
        analysis_results=json.dumps(analysis_results, indent=2),
        analysis_plan=json.dumps(analysis_plan, indent=2)
    )
    
    try:
        interpretations = await call_model_json(
            messages=[{"role": "user", "content": prompt}],
            config=config
        )
        
        # Create an AI message to record this step
        message = AIMessage(content=f"Interpreted analysis results:\n{json.dumps(interpretations, indent=2)}")
        
        # Get existing messages and add new message
        existing_messages: List[AIMessage] = state.get("messages", [])
        messages: List[AIMessage] = existing_messages + [message] if existing_messages else [message]
        
        return {
            **state,
            "interpretations": interpretations,
            "messages": messages
        }
    except Exception as e:
        return {
            **state,
            "error": {"phase": "interpret_results", "message": str(e)}
        }


async def generate_visualizations(state: AnalysisState, config: Optional[RunnableConfig] = None) -> AnalysisState:
    """Generate visualizations based on analysis results."""
    prepared_data: Dict[str, Any] = state.get("prepared_data", {})
    analysis_results: Dict[str, Any] = state.get("analysis_results", {})
    analysis_plan: Dict[str, Any] = state.get("analysis_plan", {})
    
    if not prepared_data or not analysis_results:
        return {
            **state,
            "error": {"phase": "generate_visualizations", "message": "Missing prepared data or analysis results"}
        }
    
    visualizations: List[Dict[str, Any]] = state.get("visualizations", [])
    code_snippets: List[str] = state.get("code_snippets", [])
    
    # Generate visualizations based on analysis plan
    if isinstance(analysis_plan, dict) and "visualizations" in analysis_plan:
        plan_visualizations = analysis_plan.get("visualizations", [])
        if isinstance(plan_visualizations, list):
            for viz in plan_visualizations:
                if not isinstance(viz, dict):
                    continue
                
                viz_type = viz.get("type", "")
                target = viz.get("target", "")
                
                if not viz_type or not target or target not in prepared_data:
                    continue
                
                df = prepared_data[target]
                if not isinstance(df, pd.DataFrame):
                    continue
                
                # Add visualization code and metadata
                visualizations.append({
                    "type": viz_type,
                    "target": target,
                    "data": df.to_dict(),
                    "config": viz
                })
                code_snippets.append(f"# Generated {viz_type} visualization for {target}")
    
    # Create an AI message to record this step
    message = AIMessage(content=f"Generated {len(visualizations)} visualizations")
    
    # Get existing messages and add new message
    existing_messages: List[AIMessage] = state.get("messages", [])
    messages: List[AIMessage] = existing_messages + [message] if existing_messages else [message]
    
    return {
        **state,
        "visualizations": visualizations,
        "code_snippets": code_snippets,
        "messages": messages
    }

async def compile_report(state: AnalysisState, config: Optional[RunnableConfig] = None) -> AnalysisState:
    """Compile a final report based on all analysis results."""
    task: str = state.get("task", "")
    analysis_plan: Dict[str, Any] = state.get("analysis_plan", {})
    analysis_results: Dict[str, Any] = state.get("analysis_results", {})
    interpretations: Dict[str, Any] = state.get("interpretations", {})
    visualizations: List[Dict[str, Any]] = state.get("visualizations", [])
    
    if not task or not analysis_results or not interpretations:
        return {
            **state,
            "error": {"phase": "compile_report", "message": "Missing required components for report compilation"}
        }
    
    # Create prompt with all components
    prompt: str = COMPILE_REPORT_PROMPT.format(
        task=task,
        analysis_plan=json.dumps(analysis_plan, indent=2),
        analysis_results=json.dumps(analysis_results, indent=2),
        interpretations=json.dumps(interpretations, indent=2),
        visualizations=json.dumps(visualizations, indent=2)
    )
    
    try:
        report = await call_model_json(
            messages=[{"role": "user", "content": prompt}],
            config=config
        )
        
        # Create an AI message to record this step
        message = AIMessage(content=f"Compiled final analysis report:\n{json.dumps(report, indent=2)}")
        
        # Get existing messages and add new message
        existing_messages: List[AIMessage] = state.get("messages", [])
        messages: List[AIMessage] = existing_messages + [message] if existing_messages else [message]
        
        return {
            **state,
            "report": report,
            "messages": messages
        }
    except Exception as e:
        return {
            **state,
            "error": {"phase": "compile_report", "message": str(e)}
        }

###############################################################################
# 3) Error handling function
###############################################################################
def handle_error(state: AnalysisState) -> str:
    """Route based on presence of error."""
    return "handle_error" if "error" in state else "continue"

def error_handler(state: AnalysisState) -> AnalysisState:
    """Handle errors in the pipeline."""
    error: Dict[str, Any] = state.get("error", {}) or {}
    
    # Create an error message
    message = AIMessage(content=f"Error in phase '{error.get('phase', 'unknown')}': {error.get('message', 'Unknown error')}")
    
    # Get existing messages and add new message
    existing_messages: List[AIMessage] = state.get("messages", [])
    messages: List[AIMessage] = existing_messages + [message] if existing_messages else [message]
    
    return {
        **state,
        "messages": messages
    }

###############################################################################
# 4) Define the flow router
###############################################################################
def flow_router(state: AnalysisState) -> str:
    """Determine the next step in the flow based on the current state."""
    if "error" in state:
        return "handle_error"

    if "report" in state:
        return END

    if "interpretations" in state and "visualizations" in state:
        return "compile_report"

    if "analysis_results" in state:
        if "interpretations" in state:
            return "generate_visualizations"

        return "interpret_results"

    if "prepared_data" in state:
        return "analyze_data"

    if "analysis_plan" in state:
        return "prepare_data"

    if "task" in state and "data" in state:
        return "formulate_analysis_plan"

    return "handle_error"

###############################################################################
# 5) Build the LangGraph
###############################################################################
def create_analysis_graph(config: Optional[RunnableConfig] = None) -> StateGraph:
    """Create the analysis workflow graph."""
    # Create nodes dictionary with partially applied config
    nodes: Dict[str, Union[Callable[[AnalysisState], AnalysisState], 
                          Callable[[AnalysisState], Coroutine[Any, Any, AnalysisState]]]] = {
        "formulate_analysis_plan": lambda state: formulate_analysis_plan(state, config),
        "prepare_data": lambda state: prepare_data(state, config),
        "analyze_data": lambda state: analyze_data(state, config),
        "interpret_results": lambda state: interpret_results(state, config),
        "generate_visualizations": lambda state: generate_visualizations(state, config),
        "compile_report": lambda state: compile_report(state, config),
        "handle_error": error_handler
    }
    
    # Create the graph with the state
    workflow = StateGraph(state_schema=AnalysisState)
    
    # Add nodes
    for name, func in nodes.items():
        workflow.add_node(name, func)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "formulate_analysis_plan",
        handle_error,
        {
            "continue": "prepare_data",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "prepare_data",
        handle_error,
        {
            "continue": "analyze_data",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "analyze_data",
        handle_error,
        {
            "continue": "interpret_results",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "interpret_results",
        handle_error,
        {
            "continue": "generate_visualizations",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "generate_visualizations",
        handle_error,
        {
            "continue": "compile_report",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "compile_report",
        handle_error,
        {
            "continue": END,
            "handle_error": "handle_error"
        }
    )
    
    # Add final edge from error handler
    workflow.add_edge("handle_error", END)
    
    # Set the entry point
    workflow.set_entry_point("formulate_analysis_plan")
    
    return workflow

###############################################################################
# 6) Function to run the analysis
###############################################################################
async def run_analysis(task: str, data: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Run the full analysis pipeline.
    
    Args:
        task: The analysis task description
        data: Dictionary of datasets to analyze
        config: Optional configuration for the pipeline
        
    Returns:
        The final state with analysis results and report
    """
    # Create the graph
    graph: StateGraph = create_analysis_graph(config)

    # Create the initial state
    initial_state: AnalysisState = {
        "task": task,
        "data": data,
        "messages": [],
        "code_snippets": [],
        "visualizations": []
    }

    # Execute the graph
    async_graph: CompiledStateGraph = graph.compile()
    return await async_graph.ainvoke(input=initial_state)

# Create default graph instance with default configuration
graph: StateGraph = create_analysis_graph()