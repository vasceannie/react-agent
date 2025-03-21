from typing import TypedDict, List, Dict, Any, Optional, Union
from typing_extensions import Annotated
from langgraph.graph import StateGraph
from datetime import datetime
from langchain_core.messages import AIMessage

# Import existing utilities
from ..utils.logging import error_highlight, warning_highlight, info_success, get_logger

logger = get_logger(__name__)

# Define state schema for error handling
class ErrorState(TypedDict):
    errors: List[Dict[str, Any]]  # List of error objects
    critical_error: Optional[bool]  # Flag for critical errors
    workflow_status: Optional[str]  # Status like "failed", "recovered", "degraded"
    error_recovery: Optional[List[Dict[str, Any]]]  # List of recovery actions
    validation_error_summary: Optional[Dict[str, Any]]  # Summary of validation errors
    validation_passed: Optional[bool]  # Flag for validation status
    error_summary: Optional[str]  # Text summary of errors
    error_timestamp: Optional[str]  # When the error occurred
    messages: Optional[List[AIMessage]]  # Messages related to errors
    task_complete: Optional[bool]  # Flag to indicate completion

async def handle_general_error(state: ErrorState) -> Dict[str, Any]:
    """Handle errors reported in the state by implementing appropriate recovery actions."""
    errors = state.get("errors", [])
    if not errors:
        return {}
        
    # Log each error
    for error in errors:
        phase = error.get("phase", "unknown")
        message = error.get("message", "unknown error")
        error_highlight(f"Error in phase {phase}: {message}")
        
    critical_phases = ["initialization", "recovery", "persistence"]
    has_critical_error = any(
        error.get("phase", "") in critical_phases for error in errors
    )
    
    if has_critical_error:
        error_highlight("Critical error detected, cannot continue workflow")
        return {
            "critical_error": True,
            "workflow_status": "failed",
        }
        
    recovery_actions: List[Dict[str, Any]] = []
    for error in errors:
        phase = error.get("phase", "")
        if phase == "search_query":
            recovery_actions.append({
                "action": "retry_with_broader_query",
                "phase": phase,
                "details": "Retrying with a more general search query",
            })
        elif phase == "fact_check":
            recovery_actions.append({
                "action": "mark_uncertain",
                "phase": phase,
                "details": "Marking questionable facts as uncertain",
            })
        elif phase in ["synthesis", "validation"]:
            recovery_actions.append({
                "action": "simplify_scope",
                "phase": phase,
                "details": "Reducing scope to focus on verified information",
            })
        elif phase == "human_feedback":
            recovery_actions.append({
                "action": "proceed_without_feedback",
                "phase": phase,
                "details": "Continuing without human feedback",
            })
        else:
            recovery_actions.append({
                "action": "log_and_continue",
                "phase": phase,
                "details": "Logging error and attempting to continue",
            })
            
    if recovery_actions:
        info_success(f"Attempting {len(recovery_actions)} recovery actions")
        return {
            "error_recovery": recovery_actions,
            "workflow_status": "recovered",
            "critical_error": False,
        }
        
    warning_highlight("No recovery actions available for errors")
    return {
        "workflow_status": "degraded",
    }

async def handle_validation_error(state: ErrorState) -> Dict[str, Any]:
    """Handle validation-specific errors."""
    errors = state.get("errors", [])
    validation_phases = [
        "criteria",
        "fact_check",
        "logic_check",
        "consistency_check",
        "human_feedback",
    ]
    validation_errors = [
        error for error in errors if error.get("phase", "") in validation_phases
    ]
    
    if not validation_errors:
        return {}
        
    for error in validation_errors:
        phase = error.get("phase", "unknown")
        message = error.get("message", "unknown error")
        error_highlight(f"Validation error in phase {phase}: {message}")
        
    error_counts: Dict[str, int] = {}
    for error in validation_errors:
        phase = error.get("phase", "unknown")
        error_counts[phase] = error_counts.get(phase, 0) + 1
        
    if error_counts.get("fact_check", 0) > 0:
        validation_status = "failed_fact_check"
    elif error_counts.get("logic_check", 0) > 0:
        validation_status = "failed_logic_check"
    elif error_counts.get("consistency_check", 0) > 0:
        validation_status = "failed_consistency_check"
    elif error_counts.get("human_feedback", 0) > 0:
        validation_status = "failed_human_feedback"
    else:
        validation_status = "failed_validation"
        
    summary = {
        "validation_status": validation_status,
        "error_count": len(validation_errors),
        "error_distribution": error_counts,
        "recommendation": "Revise content based on validation feedback",
    }
    
    warning_highlight("Validation failed")
    return {
        "validation_error_summary": summary,
        "validation_passed": False,
    }


def create_error_message(state: ErrorState) -> Dict[str, Any]:
    """Create user-facing error messages."""
    errors = state.get("errors", [])
    
    # Log all errors
    for i, error in enumerate(errors):
        error_msg = f"Error {i+1}/{len(errors)}: {error.get('message', 'Unknown error')}"
        error_highlight(error_msg)
        if "traceback" in error:
            logger.error(f"Traceback: {error['traceback']}")
    
    # Create a summarized error message
    error_messages = [e.get("message", "Unknown error") for e in errors]
    summary = f"Encountered {len(errors)} error(s): {', '.join(error_messages)}"
    
    # Create an AI message for the error
    error_message = AIMessage(content=f"Error: {summary}")
    
    # Return updated state with error message
    return {
        "task_complete": True,  # Mark as complete to end the graph
        "error_summary": summary,
        "error_timestamp": datetime.now().isoformat(),
        "messages": [error_message],
    }

def should_handle_validation(state: ErrorState) -> str:
    """Determine if we should route to validation error handler."""
    errors = state.get("errors", [])
    validation_phases = [
        "criteria", "fact_check", "logic_check", "consistency_check", "human_feedback",
    ]
    
    has_validation_errors = any(
        error.get("phase", "") in validation_phases for error in errors
    )
    
    return "handle_validation_error" if has_validation_errors else "handle_general_error"

def decide_next_step(state: ErrorState) -> str:
    """Decide what to do after error handling."""
    # If we have a critical error, go straight to creating error messages
    if state.get("critical_error", False):
        return "create_error_message"
    
    # If validation failed, create error message
    if state.get("validation_passed", None) is False:
        return "create_error_message"
        
    # If we have recovery actions, we might want to retry the workflow
    if state.get("error_recovery", None):
        # Here we could potentially return to a "retry" node
        # For simplicity, we'll just create an error message
        return "create_error_message"
        
    # Default path: just create error message
    return "create_error_message"

# Create the error handling graph
def create_error_graph():
    """Create and compile the error handling graph."""
    error_graph = StateGraph(ErrorState)
    
    # Add nodes to the graph
    error_graph.add_node("error_router", should_handle_validation)
    error_graph.add_node("handle_general_error", handle_general_error)
    error_graph.add_node("handle_validation_error", handle_validation_error)
    error_graph.add_node("create_error_message", create_error_message)
    error_graph.add_node("next_step_router", decide_next_step)
    
    # Define the graph structure (edges)
    error_graph.set_entry_point("error_router")
    error_graph.add_conditional_edges(
        "error_router",
        should_handle_validation,
        {
            "handle_validation_error": "handle_validation_error",
            "handle_general_error": "handle_general_error"
        }
    )
    error_graph.add_edge("handle_validation_error", "next_step_router")
    error_graph.add_edge("handle_general_error", "next_step_router")
    error_graph.add_conditional_edges(
        "next_step_router",
        decide_next_step,
        {
            "create_error_message": "create_error_message"
        }
    )
    
    # Set the terminal node
    error_graph.add_edge("create_error_message", "END")
    
    # Compile the graph
    return error_graph.compile()