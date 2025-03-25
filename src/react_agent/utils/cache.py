"""Type-safe caching and checkpointing utilities.

This module provides a unified interface for caching and checkpointing with LangGraph,
ensuring type safety and consistent behavior across the application.

Examples:
    Basic usage with simple function:
    >>> @cache_result(ttl=600)
    >>> def compute(a: int, b: int) -> int:
    >>>     return a + b
    >>> result = compute(3, 4)  # Returns 7, either from cache or computed

    Using the ProcessorCache directly:
    >>> cache = ProcessorCache(thread_id="test-thread")
    >>> cache.put("user:123", {"name": "John", "age": 30}, ttl=3600)
    >>> user_data = cache.get("user:123")
    >>> print(user_data)  # Output: {'name': 'John', 'age': 30}

    Cache statistics example:
    >>> cache = ProcessorCache()
    >>> cache.get("missing_key")  # Returns None (cache miss)
    >>> cache.cache_hits  # Returns 0
    >>> cache.cache_misses  # Returns 1
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from functools import wraps
from typing import Any, Callable, Dict, TypeVar, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from .logging import get_logger, log_performance_metrics

# Get module logger
logger = get_logger(__name__)

# Generic type variables for strict type safety
T = TypeVar('T')
R = TypeVar('R')


# Use TypedDict for better type safety
class CacheEntryData(TypedDict):
    """Data structure for individual cache entries.

    Example:
        {
            "data": {"product": "Widget", "price": 19.99},  # The actual cached data
            "timestamp": "2023-10-15T14:30:00.000000+00:00",  # ISO format timestamp
            "ttl": 3600,  # Time-to-live in seconds
            "version": 1,  # Cache version
            "metadata": {"source": "API", "tags": ["popular"]}  # Optional metadata
        }
    """
    data: Any
    timestamp: str
    ttl: int
    version: int
    metadata: Dict[str, Any]


class CacheState(TypedDict):
    """State representation for the cache system."""
    entries: Dict[str, CacheEntryData]
    stats: Dict[str, int]


class CheckpointMetadataDict(TypedDict):
    """Metadata structure for checkpoints."""
    source: str
    function: str | None
    ttl: int


class ProcessorCache:
    """A type-safe processor cache utilizing LangGraph checkpointing for persistent storage.

    Examples:
        Initialization:
        >>> cache = ProcessorCache(thread_id="user-session-123", version=2)

        Basic operations:
        >>> cache.put("product:456", {"name": "Gadget", "stock": 42}, ttl=300)
        >>> item = cache.get("product:456")
        >>> print(item)  # Output: {'name': 'Gadget', 'stock': 42}

        With metadata:
        >>> cache.put(
        ...     "config:app",
        ...     {"theme": "dark", "locale": "en_US"},
        ...     ttl=86400,
        ...     metadata={"updated_by": "system"}
        ... )
    """
    
    def __init__(self, thread_id: str = "default-processor", version: int = 1) -> None:
        """Initialize the cache with a specific thread ID and version."""
        self.memory_saver = MemorySaver()
        self.thread_id = thread_id
        self.version = version
        self.cache_hits = 0
        self.cache_misses = 0
        # In-memory cache for fallback
        self.memory_cache: Dict[str, CacheEntryData] = {}
        logger.info(f"Initialized ProcessorCache with thread_id={thread_id}, version={version}")
    
    def _get_config(self, checkpoint_id: str) -> RunnableConfig:
        """Create a configuration object for checkpoint operations."""
        return RunnableConfig(configurable={
            "thread_id": self.thread_id,
            "checkpoint_id": checkpoint_id,
        })
    
    def get(self, key: str) -> Any | None:
        """Retrieve data from the cache for a given key.

        Args:
            key: The unique identifier for the cached data

        Returns:
            The cached data if found and valid, otherwise None

        Examples:
            >>> cache.get("user:789")
            {'name': 'Alice', 'email': 'alice@example.com'}

            >>> cache.get("nonexistent_key")
            None
        """
        start_time = time.time()
        
        # First check memory cache for fallback
        if key in self.memory_cache:
            logger.info(f"Cache hit from memory for key: {key}")
            self.cache_hits += 1
            return self.memory_cache[key]["data"]
        
        try:
            # Use LangGraph checkpoint system to retrieve data
            checkpoint = self.memory_saver.get(self._get_config(key))
            
            if checkpoint is not None and (isinstance(checkpoint, dict) and "values" in checkpoint):
                cached_entry = checkpoint["values"].get("entry")
                if cached_entry and self._is_cache_valid(cached_entry):
                    self.cache_hits += 1
                    logger.info(f"Cache hit from checkpoint for key: {key}")
                    return cached_entry["data"]
            
            self.cache_misses += 1
            logger.info(f"Cache miss for key: {key}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}", exc_info=True)
            return None
        finally:
            end_time = time.time()
            log_performance_metrics("Cache retrieval", start_time, end_time, category="Cache")
    
    def put(
        self,
        key: str,
        data: Any,
        ttl: int = 3600,
        metadata: Dict[str, Any] | None = None
    ) -> None:
        """Store data in the cache under the provided key.

        Args:
            key: Unique identifier for the data
            data: The data to be cached (any serializable type)
            ttl: Time-to-live in seconds (default: 3600)
            metadata: Optional dictionary of metadata (default: None)

        Examples:
            >>> cache.put("session:abc123", {"user_id": 42, "logged_in": True}, ttl=1800)

            With metadata:
            >>> cache.put(
            ...     "report:q3",
            ...     {"revenue": 1_000_000, "growth": 0.15},
            ...     ttl=86400,
            ...     metadata={"generated_by": "report_service", "format": "v2"}
            ... )
        """
        start_time = time.time()

        # Create cache entry
        entry: CacheEntryData = {
            "data": data,
            "timestamp": datetime.now(UTC).isoformat(),
            "ttl": ttl,
            "version": self.version,
            "metadata": metadata or {}
        }

        # Store in memory cache for fallback
        self.memory_cache[key] = entry

        try:
            # Create checkpoint using LangGraph patterns
            values = {"entry": entry}

            # Use LangGraph's checkpoint system
            checkpoint_data: Dict[str, Any] = {
                "id": key,
                "ts": datetime.now(UTC).isoformat(),
                "v": self.version,
                "channel_values": values,
                "channel_versions": {k: self.version for k in values},
                "versions_seen": {self.thread_id: self.version},
                "pending_sends": [],
                "metadata": metadata or {},
            }

            checkpoint = Checkpoint(**checkpoint_data)

            self.memory_saver.put(
                self._get_config(key),
                checkpoint,
                metadata={
                    "source": "input",
                    "step": self.version,
                    "writes": metadata or {},
                    "parents": {},
                },
                new_versions={k: self.version for k in values},
            )

            logger.info(f"Stored data in cache for key: {key}")
        except Exception as e:
            logger.error(f"Error saving to checkpoint system: {str(e)}", exc_info=True)
            logger.warning("Falling back to memory cache only")
        finally:
            end_time = time.time()
            log_performance_metrics("Cache storage", start_time, end_time, category="Cache")
    
    def cache_result(
        self,
        ttl: int = 3600
    ) -> Callable[[Callable[..., R]], Callable[..., R]]:
        """Decorate a function to cache its results with type preservation.

        Args:
            ttl: Time-to-live in seconds for cached results (default: 3600)

        Returns:
            A decorated function that caches its results

        Examples:
            Basic usage:
            >>> @cache.cache_result(ttl=300)
            >>> def get_user_details(user_id: int) -> dict:
            >>>     # Expensive database call here
            >>>     return {"id": user_id, "name": "John"}

            First call (executes function):
            >>> get_user_details(42)  # Returns {'id': 42, 'name': 'John'}

            Subsequent call (returns cached result):
            >>> get_user_details(42)  # Returns cached result immediately
        """
        def decorator(func: Callable[..., R]) -> Callable[..., R]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> R:
                start_time = time.time()
                cache_key = self._generate_cache_key(func, args, kwargs)
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    end_time = time.time()
                    log_performance_metrics(f"Cache hit for {func.__name__}", start_time, end_time, category="Cache")
                    return cast(R, cached_result)
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                
                self.put(
                    cache_key,
                    result,
                    ttl=ttl,
                    metadata={"function": func.__name__}
                )
                
                end_time = time.time()
                log_performance_metrics(f"Cache miss for {func.__name__}", start_time, end_time, category="Cache")
                return result
            
            return wrapper
        return decorator
    
    def _generate_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate a unique cache key from function and arguments.
        
        Creates a deterministic hash key based on:
        - Function name or ID
        - Arguments (positional and keyword)
        - Cache version
        
        Args:
            func: The function being cached
            args: Positional arguments passed to the function
            kwargs: Keyword arguments passed to the function
            
        Returns:
            A SHA256 hex digest string representing the unique cache key
            
        Examples:
            >>> def example(a: int, b: int = 2) -> int:
            ...     return a + b
            >>> cache._generate_cache_key(example, (1,), {})
            'a1b2c3...'  # SHA256 hash
            
            >>> cache._generate_cache_key(example, (1,), {'b': 3})
            'd4e5f6...'  # Different hash due to changed arguments
            
            With complex objects:
            >>> cache._generate_cache_key(example, ([1,2],), {'b': {'x': 1}})
            'g7h8i9...'  # Hash based on string representation
        """
        try:
            func_name = func.__name__
        except (AttributeError, TypeError):
            func_name = id(func)

        # Handle complex objects in args
        args_str = [str(arg) if isinstance(arg, (list, dict, set)) else arg for arg in args]
        cache_data = {
            'args': str(args_str),
            'kwargs': str(kwargs),
            'func': func_name,
            'version': self.version
        }
        cache_str = json.dumps(cache_data, sort_keys=True, default=str)
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    def _is_cache_valid(self, cached: Dict[str, Any]) -> bool:
        """Validate if a cached entry is still fresh based on TTL.
        
        Checks:
        - Presence of required timestamp field
        - Valid ISO format timestamp
        - Whether current time is within TTL window
        
        Args:
            cached: The cache entry dictionary containing:
                - timestamp: ISO format string
                - ttl: Time-to-live in seconds
                
        Returns:
            True if cache entry is valid and fresh, False otherwise
            
        Examples:
            Valid case:
            >>> entry = {
            ...     "timestamp": "2023-10-15T14:30:00.000000+00:00",
            ...     "ttl": 3600,
            ...     "data": {...}
            ... }
            >>> cache._is_cache_valid(entry)  # True if within 1 hour of timestamp
            
            Expired case:
            >>> old_entry = {
            ...     "timestamp": "2023-10-01T00:00:00.000000+00:00", 
            ...     "ttl": 3600,
            ...     "data": {...}
            ... }
            >>> cache._is_cache_valid(old_entry)  # False
            
            Invalid format:
            >>> bad_entry = {
            ...     "timestamp": "invalid-date",
            ...     "ttl": 3600
            ... }
            >>> cache._is_cache_valid(bad_entry)  # False
        """
        timestamp_str = cached.get("timestamp", "")
        if not timestamp_str:
            return False
        
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            elapsed = (datetime.now(UTC) - timestamp).total_seconds()
            return elapsed < cached.get("ttl", 3600)
        except ValueError:
            return False
