"""Logging utilities.

This module provides enhanced logging utilities and convenience methods for the agent framework.
It builds upon the logging configuration defined in log_config.py to enable rich, formatted logging output.
"""

import logging
from typing import Any, Mapping, Optional, Dict

# Module: log_config.py
# This module provides logging configuration for the enrichment agent, including functions
# for setting up and configuring loggers with rich formatting.

import threading
from typing import Optional  # noqa: F401

from rich.console import Console
from rich.logging import RichHandler

# Create a rich console for formatted output (logs will be printed to stderr).
console = Console(stderr=True)

# Logging configuration constants.
LOG_FORMAT = "%(message)s"  # Maintained for backward compatibility with tests.
DATE_FORMAT = "[%X]"

# A thread-safe lock to ensure logger configuration is not accessed concurrently.
_logger_lock = threading.Lock()


def setup_logger(name: str = "enrichment_agent", level: int = logging.INFO) -> logging.Logger:
    """
    Set up and configure a logger with rich formatting.

    Args:
        name (str): The name of the logger. Defaults to "enrichment_agent".
        level (int): The logging level to set (e.g., logging.DEBUG, logging.INFO). Defaults to logging.INFO.

    Returns:
        logging.Logger: A configured logger instance with rich formatting enabled.

    Examples:
        >>> logger = setup_logger("my_agent", level=logging.DEBUG)
        >>> logger.info("This is an info message.")
    """
    with _logger_lock:
        logger = logging.getLogger(name)
        # Configure the logger only if it hasn't been set up already.
        if not logger.handlers:
            logger.setLevel(level)
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
            logger.addHandler(handler)
            logger.propagate = False
        return logger


# Create a default logger instance.
logger = setup_logger()


def set_level(level: int) -> None:
    """
    Set the logging level for both the default logger and the root logger.

    Args:
        level (int): The logging level to set (e.g., logging.DEBUG, logging.INFO).

    Returns:
        None

    Examples:
        >>> set_level(logging.DEBUG)
    """
    with _logger_lock:
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
        # Update the root logger's level to affect the entire logging hierarchy.
        root_logger = logging.getLogger()
        root_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Retrieve a logger with the specified name, configured with rich formatting.

    Args:
        name (str): The name for the logger (typically __name__ from the calling module).

    Returns:
        logging.Logger: A logger instance with rich formatting and proper log levels.

    Examples:
        >>> my_logger = get_logger(__name__)
        >>> my_logger.info("Logger retrieved and ready to use.")
    """
    return setup_logger(name)


def info_success(message: str, exc_info: bool | BaseException | None = None) -> None:
    """
    Log a success message with green formatting.

    Args:
        message (str): The message to log.
        exc_info (bool | BaseException | None): Optional exception information to include in the log.

    Returns:
        None

    Examples:
        >>> info_success("Operation completed successfully.")
    """
    logger.info("[bold green]✓ %s[/bold green]", message, exc_info=exc_info)


def info_highlight(
    message: str, 
    category: Optional[str] = None, 
    progress: Optional[str] = None, 
    exc_info: bool | BaseException | None = None
) -> None:
    """
    Log an informational message with blue highlighting, optionally tagged with a category and progress.

    Args:
        message (str): The message to log.
        category (Optional[str]): An optional category to tag the message.
        progress (Optional[str]): An optional progress indicator to prefix the message.
        exc_info (bool | BaseException | None): Optional exception information to include.

    Returns:
        None

    Examples:
        >>> info_highlight("Data loaded successfully", category="DataLoader")
        >>> info_highlight("50% completed", progress="50%")
    """
    if progress:
        message = f"[{progress}] {message}"
    if category:
        message = f"[{category}] {message}"
    logger.info("[bold blue]ℹ %s[/bold blue]", message, exc_info=exc_info)


def warning_highlight(
    message: str, 
    category: Optional[str] = None, 
    exc_info: bool | BaseException | None = None
) -> None:
    """
    Log a warning message with yellow highlighting, optionally tagged with a category.

    Args:
        message (str): The warning message to log.
        category (Optional[str]): An optional category tag.
        exc_info (bool | BaseException | None): Optional exception information to include.

    Returns:
        None

    Examples:
        >>> warning_highlight("Low disk space", category="System")
    """
    if category:
        message = f"[{category}] {message}"
    logger.warning("[bold yellow]⚠ %s[/bold yellow]", message, exc_info=exc_info)


def error_highlight(
    message: str, 
    category: Optional[str] = None, 
    exc_info: bool | BaseException | None = None
) -> None:
    """
    Log an error message with red highlighting, optionally tagged with a category.

    Args:
        message (str): The error message to log.
        category (Optional[str]): An optional category tag.
        exc_info (bool | BaseException | None): Optional exception information to include.

    Returns:
        None

    Examples:
        >>> error_highlight("Failed to connect to database", category="Database")
    """
    if category:
        message = f"[{category}] {message}"
    logger.error("[bold red]✗ %s[/bold red]", message, exc_info=exc_info)


def log_dict(data: Mapping[str, Any], level: int = logging.INFO, title: Optional[str] = None) -> None:
    """
    Log a dictionary with pretty formatting for easier readability.

    Args:
        data (Mapping[str, Any]): The dictionary data to log.
        level (int): The logging level to use (e.g., logging.INFO). Defaults to logging.INFO.
        title (Optional[str]): An optional title to display before the dictionary output.

    Raises:
        ValueError: If an invalid logging level is provided.

    Returns:
        None

    Examples:
        >>> log_dict({"key1": "value1", "key2": 42}, level=logging.DEBUG, title="Config Data")
    """
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

    for key, value in data.items():
        logger.log(level, "  [cyan]%s[/cyan]: %s", key, value)


def log_step(step_name: str, step_number: Optional[int] = None, total_steps: Optional[int] = None) -> None:
    """
    Log a processing step, optionally including the step number within a sequence.

    Args:
        step_name (str): The name or description of the step.
        step_number (Optional[int]): The current step number in the sequence.
        total_steps (Optional[int]): The total number of steps in the sequence.

    Raises:
        ValueError: If one of step_number or total_steps is provided without the other.
        ValueError: If step_number is not within the valid range (1 to total_steps).

    Returns:
        None

    Examples:
        >>> log_step("Loading data")
        >>> log_step("Processing data", step_number=2, total_steps=5)
    """
    if (step_number is None) != (total_steps is None):
        raise ValueError("Both step_number and total_steps must be provided together")

    if step_number is None or total_steps is None:
        logger.info("[bold magenta]Step:[/bold magenta] %s", step_name)
    elif not 1 <= step_number <= total_steps:
        raise ValueError(f"Invalid step numbers: {step_number}/{total_steps}")
    else:
        logger.info("[bold magenta]Step %s/%s:[/bold magenta] %s", step_number, total_steps, step_name)


def log_progress(current: int, total: int, category: str, operation: str) -> None:
    """
    Log progress for long-running operations.

    Args:
        current (int): The current progress count.
        total (int): The total count representing completion.
        category (str): Category of the operation for grouping logs.
        operation (str): Description of the operation (e.g., "processing", "extracting").

    Returns:
        None

    Examples:
        >>> log_progress(5, 20, category="DataLoad", operation="Loading")
    """
    if total > 0:
        percentage = (current / total) * 100
        info_highlight(f"{operation} {current}/{total} ({percentage:.1f}%)", category=category)


def log_performance_metrics(
    operation: str,
    start_time: float,
    end_time: float,
    category: Optional[str] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log performance metrics for a given operation.

    Args:
        operation (str): The name or description of the operation.
        start_time (float): The start time (e.g., as returned by time.time()).
        end_time (float): The end time (e.g., as returned by time.time()).
        category (Optional[str]): An optional category for grouping metrics.
        additional_info (Optional[Dict[str, Any]]): Optional additional metrics to log.

    Returns:
        None

    Examples:
        >>> import time
        >>> start = time.time()
        >>> # ... perform operation ...
        >>> end = time.time()
        >>> log_performance_metrics("Data Processing", start, end, category="Performance", additional_info={"records": 1000})
    """
    duration = end_time - start_time
    message = f"{operation} completed in {duration:.2f}s"
    if additional_info:
        info_parts = [f"{k}: {v}" for k, v in additional_info.items()]
        message += f" ({', '.join(info_parts)})"
    info_highlight(message, category=category)
