"""Logging utilities.

This module provides enhanced logging utilities and convenience methods
for the agent framework. It builds upon the basic logging configuration
defined in log_config.py.
"""

import logging
from typing import Any, Mapping, Optional, Dict


"""Module: log_config.py. This module provides logging configuration for the enrichment agent.

-------------------------
It includes functions for setting up and configuring loggers with rich formatting.
"""

# Standard library imports
import logging
import threading
from typing import Optional  # noqa: F401

# Third-party imports
from rich.console import Console
from rich.logging import RichHandler

# Create console for rich output - use stderr for logs
console = Console(stderr=True)

# Logging configuration
LOG_FORMAT = "%(message)s"  # Added back for backward compatibility with tests
DATE_FORMAT = "[%X]"

# Logger lock for thread safety
_logger_lock = threading.Lock()


def setup_logger(
    name: str = "enrichment_agent", level: int = logging.INFO
) -> logging.Logger:
    """Set up a logger with rich formatting for beautiful console output.

    Args:
        name: Name of the logger. Defaults to "enrichment_agent".
        level: Logging level. Defaults to logging.INFO.

    Returns:
        Configured logger instance with rich formatting.
    """
    with _logger_lock:
        logger = logging.getLogger(name)

        # Only configure the logger if it hasn't been configured yet
        if not logger.handlers:
            logger.setLevel(level)

            # Create rich handler with proper parameter passing
            handler = RichHandler(
                console=console,
                rich_tracebacks=True,
                tracebacks_show_locals=True,
                show_time=True,
                show_path=True,
                markup=True,
                log_time_format=DATE_FORMAT,
                omit_repeated_times=False,
                level=level,
            )

            # Add handler and disable propagation to prevent duplicate logs
            logger.addHandler(handler)
            logger.propagate = False

        return logger


# Create a default logger instance
logger = setup_logger()

def set_level(level: int) -> None:
    """Set the logging level for the default logger and root logger.
    
    Args:
        level: The logging level to set (e.g., logging.DEBUG, logging.INFO)
    """
    with _logger_lock:
        # Set level for the default logger
        logger.setLevel(level)
        
        # Update handler level
        for handler in logger.handlers:
            handler.setLevel(level)
        
        # Also set for root logger to affect other loggers in hierarchy
        root_logger = logging.getLogger()
        root_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    This function returns a logger configured with rich formatting and proper
    log levels. It uses the setup_logger function from log_config to ensure
    consistent logging configuration across the application.

    Args:
        name (str): The name for the logger, typically __name__ from the calling module

    Returns:
        logging.Logger: A configured logger instance with rich formatting
    """
    return setup_logger(name)


# Convenience methods for formatted logging with Rich markup
def info_success(message: str, exc_info: bool | BaseException | None = None) -> None:
    """Log a success message with green formatting.

    Args:
        message (str): The message to log
        exc_info (bool | BaseException | None): Exception information to include in the log
    """
    logger.info("[bold green]✓ %s[/bold green]", message, exc_info=exc_info)


def info_highlight(message: str, category: Optional[str] = None, progress: Optional[str] = None, exc_info: bool | BaseException | None = None) -> None:
    """Log a highlighted informational message with optional category and progress tracking.
    
    Args:
        message: The message to log
        category: Optional category for the message
        progress: Optional progress indicator
        exc_info: Exception information to include in the log
    """
    if progress:
        message = f"[{progress}] {message}"
    if category:
        message = f"[{category}] {message}"
    logger.info("[bold blue]ℹ %s[/bold blue]", message, exc_info=exc_info)


def warning_highlight(message: str, category: Optional[str] = None, exc_info: bool | BaseException | None = None) -> None:
    """Log a highlighted warning message with optional category.
    
    Args:
        message: The message to log
        category: Optional category for the message
        exc_info: Exception information to include in the log
    """
    if category:
        message = f"[{category}] {message}"
    logger.warning("[bold yellow]⚠ %s[/bold yellow]", message, exc_info=exc_info)


def error_highlight(message: str, category: Optional[str] = None, exc_info: bool | BaseException | None = None) -> None:
    """Log a highlighted error message with optional category.
    
    Args:
        message: The message to log
        category: Optional category for the message
        exc_info: Exception information to include in the log
    """
    if category:
        message = f"[{category}] {message}"
    logger.error("[bold red]✗ %s[/bold red]", message, exc_info=exc_info)


def log_dict(
    data: Mapping[str, Any], level: int = logging.INFO, title: str | None = None
) -> None:
    """Log a dictionary with pretty formatting.

    Args:
        data (Mapping[str, Any]): The dictionary to log
        level (int): The logging level to use. Defaults to logging.INFO
        title (str | None): Optional title to display before the dictionary

    Raises:
        ValueError: If level is not a valid logging level
    """
    # Validate logging level
    if level not in (
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ):
        raise ValueError(f"Invalid logging level: {level}")

    if title:
        logger.log(level, "[bold]%s[/bold]", title)

    # Use rich's pretty printing through the logger
    for key, value in data.items():
        logger.log(level, "  [cyan]%s[/cyan]: %s", key, value)


def log_step(
    step_name: str, step_number: int | None = None, total_steps: int | None = None
) -> None:
    """Log a processing step with optional step counter.

    Args:
        step_name (str): The name of the step
        step_number (int | None): Current step number if part of a sequence
        total_steps (int | None): Total number of steps in the sequence

    Raises:
        ValueError: If step_number is provided without total_steps or vice versa
        ValueError: If step_number is greater than total_steps
    """
    # Validate step numbers
    if (step_number is None) != (total_steps is None):
        raise ValueError("Both step_number and total_steps must be provided together")

    if step_number is None or total_steps is None:
        logger.info("[bold magenta]Step:[/bold magenta] %s", step_name)

    elif not 1 <= step_number <= total_steps:
        raise ValueError(f"Invalid step numbers: {step_number}/{total_steps}")
    else:
        logger.info(
            "[bold magenta]Step %s/%s:[/bold magenta] %s",
            step_number,
            total_steps,
            step_name,
        )


def log_progress(current: int, total: int, category: str, operation: str):
    """Log progress for long-running operations.
    
    Args:
        current: Current progress value
        total: Total number of items
        category: Category being processed
        operation: Type of operation (e.g., "processing", "extracting")
    """
    if total > 0:
        percentage = (current / total) * 100
        info_highlight(
            f"{operation} {current}/{total} ({percentage:.1f}%)",
            category=category
        )


def log_performance_metrics(
    operation: str,
    start_time: float,
    end_time: float,
    category: Optional[str] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """Log performance metrics for operations.
    
    Args:
        operation: Name of the operation being measured
        start_time: Start time of the operation
        end_time: End time of the operation
        category: Optional category for the operation
        additional_info: Optional dictionary of additional metrics
    """
    duration = end_time - start_time
    message = f"{operation} completed in {duration:.2f}s"
    
    if additional_info:
        info_parts = [f"{k}: {v}" for k, v in additional_info.items()]
        message += f" ({', '.join(info_parts)})"
    
    info_highlight(message, category=category)
