"""Type-safe caching and checkpointing utilities.

This module provides a unified interface for caching and checkpointing with LangGraph,
ensuring type safety and consistent behavior across the application.

Examples:
    Basic usage with result caching:
    >>> cache = ProcessorCache()
    >>> @cache.cache_result(ttl=600)
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

    Using checkpoint functions:
    >>> create_checkpoint("session:123", {"user_id": 42, "logged_in": True})
    >>> session_data = load_checkpoint("session:123")
    >>> print(session_data)  # Output: {'user_id': 42, 'logged_in': True}
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from functools import wraps
from typing import Any, Callable, Dict, TypeVar, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from .logging import (
    get_logger,
    info_highlight,
    log_performance_metrics,
    warning_highlight,
)

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


class LangGraphCheckpointMetadata(TypedDict, total=False):
    """TypedDict representation of LangGraph checkpoint metadata.
    
    This class defines the structure of metadata associated with LangGraph checkpoints.
    It uses TypedDict with total=False, meaning all fields are optional. This flexibility
    is necessary to accommodate different checkpoint scenarios and LangGraph's API
    requirements.
    
    Attributes:
        source: Identifies the origin of the checkpoint (e.g., "input", "agent")
        step: Integer representing the execution step or version
        writes: Dictionary of data written by the checkpoint operation
        parents: Dictionary mapping parent checkpoint identifiers to their versions
        checkpoint_ns: Namespace for organizing checkpoints, typically matching thread_id
        
    Examples:
        >>> metadata: LangGraphCheckpointMetadata = {
        ...     "source": "input",
        ...     "step": 1,
        ...     "writes": {"user_id": 123, "action": "login"},
        ...     "parents": {},
        ...     "checkpoint_ns": "user-session-456"
        ... }
        
        Minimal example with only required fields:
        >>> minimal_metadata: LangGraphCheckpointMetadata = {
        ...     "source": "agent",
        ...     "step": 2,
        ...     "writes": {},
        ...     "parents": {"previous-checkpoint": 1},
        ...     "checkpoint_ns": "workflow-789"
        ... }
        
        Using in MemorySaver.put():
        >>> memory_saver = MemorySaver()
        >>> memory_saver.put(
        ...     config=config,
        ...     checkpoint=checkpoint_data,
        ...     metadata={
        ...         "source": "input", 
        ...         "step": 1,
        ...         "writes": {},
        ...         "parents": {},
        ...         "checkpoint_ns": "thread-123"
        ...     },
        ...     new_versions={key: 1 for key in checkpoint_data}
        ... )
    """
    source: str
    step: int
    writes: Dict[str, Any]
    parents: Dict[str, Any]
    checkpoint_ns: str


class LangGraphCheckpoint(TypedDict):
    """TypedDict representation of a LangGraph checkpoint.
    
    This class defines the structure of a checkpoint object compatible with
    LangGraph's MemorySaver API. It encapsulates the cached data along with
    LangGraph-specific fields required for proper checkpoint management.
    
    The checkpoint structure is carefully designed to ensure compatibility with
    LangGraph's internal implementation while exposing a clean interface for
    the cache system.
    
    Attributes:
        entry: The actual cache entry containing the data, timestamp, ttl, version, 
               and metadata
        pending_sends: List of pending messages/operations (required by LangGraph)
        id: Unique identifier for the checkpoint, typically matching the cache key
        
    Examples:
        >>> cache_entry: CacheEntryData = {
        ...     "data": {"username": "alice", "status": "active"},
        ...     "timestamp": datetime.now(UTC).isoformat(),
        ...     "ttl": 3600,
        ...     "version": 1,
        ...     "metadata": {"source": "login_service"}
        ... }
        
        >>> checkpoint: LangGraphCheckpoint = {
        ...     "entry": cache_entry,
        ...     "pending_sends": [],
        ...     "id": "user:alice"
        ... }
        
        Using in memory_saver operations:
        >>> memory_saver = MemorySaver()
        >>> config = RunnableConfig(configurable={
        ...     "thread_id": "user-thread-123",
        ...     "checkpoint_id": "user:alice",
        ...     "checkpoint_ns": "user-thread-123"
        ... })
        >>> memory_saver.put(config=config, checkpoint=checkpoint, ...)
        
        Retrieving from memory_saver:
        >>> stored_checkpoint = memory_saver.get(config)
        >>> if stored_checkpoint:
        ...     entry = stored_checkpoint.channel_values.get("entry")
        ...     data = entry.get("data") if entry else None
    """
    entry: CacheEntryData
    pending_sends: list[Any]
    id: str  # Required by LangGraph's MemorySaver implementation


# Global memory saver instance for singleton pattern
_MEMORY_SAVER = MemorySaver()


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
        """Initialize the cache with a specific thread ID and version.
        
        Creates a new ProcessorCache instance configured with the given thread ID and version.
        The thread ID is used to isolate different caching contexts, ensuring that cache entries
        from different parts of the application or different users don't conflict.
        
        The cache maintains hit and miss statistics, which can be used for monitoring and
        optimization. It also uses an in-memory cache as a fallback in case the LangGraph
        checkpoint system is unavailable.
        
        Args:
            thread_id: A unique identifier for this cache instance, used to isolate 
                       caching contexts (default: "default-processor")
            version: The cache version, used for cache invalidation when the schema 
                     or data format changes (default: 1)
                     
        Returns:
            None
            
        Examples:
            Basic initialization:
            >>> cache = ProcessorCache()
            >>> print(cache.thread_id)
            default-processor
            
            With custom thread ID:
            >>> user_cache = ProcessorCache(thread_id="user-12345")
            >>> print(user_cache.thread_id)
            user-12345
            
            With custom version:
            >>> cache_v2 = ProcessorCache(version=2)
            >>> print(cache_v2.version)
            2
            
            Complete custom configuration:
            >>> custom_cache = ProcessorCache(thread_id="session-abc", version=3)
            >>> print(f"thread_id={custom_cache.thread_id}, version={custom_cache.version}")
            thread_id=session-abc, version=3
        """
        self.memory_saver = _MEMORY_SAVER
        self.thread_id = thread_id
        self.version = version
        self.cache_hits = 0
        self.cache_misses = 0
        # In-memory cache for fallback
        self.memory_cache: Dict[str, CacheEntryData] = {}
        logger.info(f"Initialized ProcessorCache with thread_id={thread_id}, version={version}")
    
    def _get_config(self, checkpoint_id: str) -> RunnableConfig:
        """Create a configuration object for LangGraph checkpoint operations.
        
        This method generates a properly structured RunnableConfig object that contains
        all the necessary information for LangGraph's MemorySaver to identify and
        retrieve checkpoints. The config includes the thread ID to maintain isolation
        between different caching contexts, a checkpoint ID to uniquely identify the
        specific data being checkpointed, and a checkpoint namespace that LangGraph
        requires for organizational purposes.
        
        Args:
            checkpoint_id: A unique identifier for the specific checkpoint,
                          typically the cache key being accessed
                          
        Returns:
            A RunnableConfig object configured for LangGraph checkpoint operations
            
        Examples:
            >>> cache = ProcessorCache(thread_id="user-session-123")
            >>> config = cache._get_config("user:456")
            >>> config_dict = config.get("configurable", {})
            >>> print(config_dict.get("thread_id"))
            user-session-123
            >>> print(config_dict.get("checkpoint_id"))
            user:456
            >>> print(config_dict.get("checkpoint_ns"))
            user-session-123
            
            # Using the config with MemorySaver:
            >>> # memory_saver = MemorySaver()
            >>> # checkpoint = memory_saver.get(config)  # Retrieve checkpoint
        """
        return RunnableConfig(configurable={
            "thread_id": self.thread_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_ns": self.thread_id,  # Add checkpoint namespace that LangGraph requires
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
            return self._retrieve_from_checkpoint(key)
        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}", exc_info=True)
            return None
        finally:
            end_time = time.time()
            log_performance_metrics("Cache retrieval", start_time, end_time, category="Cache")

    def _retrieve_from_checkpoint(self, key: str) -> Any | None:
        """Retrieve data from the LangGraph checkpoint system.
        
        This method attempts to fetch a checkpoint from the LangGraph memory saver
        and extract the cached data from it. It handles the specific structure of
        LangGraph checkpoint objects, safely accessing attributes and validating
        the cache entry before returning the data.
        
        The method increments the cache_hits counter if data is successfully retrieved
        and valid, or the cache_misses counter if no valid data is found.
        
        Args:
            key: The unique identifier for the cached data to retrieve
            
        Returns:
            The cached data if found and valid, otherwise None
            
        Examples:
            >>> cache = ProcessorCache(thread_id="test-thread")
            >>> cache.put("product:123", {"name": "Widget", "price": 19.99})
            >>> cache._retrieve_from_checkpoint("product:123")
            {'name': 'Widget', 'price': 19.99}
            
            >>> # When checkpoint doesn't exist:
            >>> cache._retrieve_from_checkpoint("nonexistent:456")
            None
            
            >>> # When checkpoint exists but is expired:
            >>> cache.put("expired:789", {"status": "old"}, ttl=0)  # Immediately expires
            >>> time.sleep(0.1)  # Ensure TTL is exceeded
            >>> cache._retrieve_from_checkpoint("expired:789")
            None
        """
        # Use LangGraph checkpoint system to retrieve data
        config: RunnableConfig = self._get_config(key)
        checkpoint = self.memory_saver.get(config)

        if checkpoint is not None:
            # Try to extract entry from the checkpoint
            # Safely access attributes without type errors
            values = getattr(checkpoint, "channel_values", {})
            if isinstance(values, dict) and "entry" in values:
                entry = values["entry"]
                if isinstance(entry, dict) and self._is_cache_valid(entry):
                    self.cache_hits += 1
                    logger.info(f"Cache hit from checkpoint for key: {key}")
                    return entry.get("data")

        self.cache_misses += 1
        logger.info(f"Cache miss for key: {key}")
        return None
    
    def put(
        self,
        key: str,
        data: Any,
        ttl: int = 3600,
        metadata: Dict[str, Any] | None = None
    ) -> None:
        """Store data in the cache under the provided key.
        
        This method stores the given data in both the in-memory cache and the 
        LangGraph checkpoint system. The data is wrapped in a CacheEntryData
        structure that includes a timestamp, TTL, version, and optional metadata.
        
        The method creates a LangGraph checkpoint that contains the cache entry,
        ensuring proper integration with LangGraph's state management. If storing
        in the checkpoint system fails, the method gracefully falls back to using
        only the in-memory cache, ensuring data availability even when LangGraph 
        functionality is limited.
        
        Performance metrics are logged for monitoring cache operations.
        
        Args:
            key: Unique identifier for the data, used to retrieve it later
            data: The data to be cached (any serializable type)
            ttl: Time-to-live in seconds, controlling how long the data remains valid
                 (default: 3600 seconds, or 1 hour)
            metadata: Optional dictionary of metadata to associate with this cache entry,
                     useful for tracking origin, purpose, or other attributes (default: None)
        
        Returns:
            None
        
        Examples:
            Basic usage:
            >>> cache = ProcessorCache(thread_id="user-session-123")
            >>> cache.put("user:456", {"name": "Alice", "role": "admin"})
            
            With custom TTL:
            >>> cache.put("temporary:key", "short-lived data", ttl=60)  # Expires in 1 minute
            
            With metadata:
            >>> cache.put(
            ...     "product:789", 
            ...     {"name": "Widget", "price": 19.99, "stock": 42},
            ...     ttl=3600,
            ...     metadata={"source": "inventory_system", "last_updated_by": "sync_job"}
            ... )
            
            Storing complex data:
            >>> user_preferences = {
            ...     "theme": "dark",
            ...     "notifications": {"email": True, "sms": False},
            ...     "recent_items": [101, 203, 305]
            ... }
            >>> cache.put(
            ...     "prefs:user-456", 
            ...     user_preferences,
            ...     ttl=86400,  # 24 hours
            ...     metadata={"version": "v2"}
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
            # Create the checkpoint configuration
            config = self._get_config(key)
            
            # Store in the checkpoint system
            # Create a properly typed checkpoint object
            checkpoint_data: LangGraphCheckpoint = {
                "entry": entry,
                "pending_sends": [],  # Required by LangGraph's MemorySaver implementation
                "id": key,  # Set the id to the key, which is required by LangGraph
            }
            
            # Store in the checkpoint system 
            self.memory_saver.put(
                config=config,
                checkpoint=cast(Any, checkpoint_data),  # Cast to Any since MemorySaver's type is different
                metadata=cast(Any, {
                    "source": "input",
                    "step": self.version,
                    "writes": metadata or {},
                    "parents": {},
                    "checkpoint_ns": self.thread_id,  # Add checkpoint namespace
                }),
                new_versions={k: self.version for k in checkpoint_data},
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
            """Wrap a function with caching behavior while preserving its signature.
            
            This decorator takes the target function and wraps it with caching logic
            that preserves the original function's type signature and metadata.
            When the wrapped function is called, it first checks the cache for a 
            previously computed result based on the function name and arguments.
            If found, it returns the cached result. Otherwise, it executes the 
            function, caches the result, and returns it.
            
            Performance metrics are logged for both cache hits and misses, allowing 
            for monitoring and optimization of cache effectiveness.
            
            Args:
                func: The function to wrap with caching behavior
                
            Returns:
                A wrapped function that implements caching while preserving the
                original function's signature, docstring, and other metadata
                
            Examples:
                >>> @cache.cache_result(ttl=300)
                >>> def compute_value(x: int) -> int:
                ...     print("Computing...")
                ...     return x * 2
                >>> 
                >>> # First call (cache miss)
                >>> result1 = compute_value(5)  # Prints "Computing..." and returns 10
                >>> 
                >>> # Second call with same args (cache hit)
                >>> result2 = compute_value(5)  # Silently returns 10 from cache
                >>> 
                >>> # Different args cause cache miss
                >>> result3 = compute_value(7)  # Prints "Computing..." and returns 14
            """
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> R:
                """Execute function with caching behavior.
                
                This wrapper function implements the actual caching logic. It first
                generates a unique cache key based on the function and its arguments,
                then checks if a valid result exists in the cache. If found, it 
                returns the cached result directly. Otherwise, it executes the original
                function, caches the result with the specified TTL, and returns it.
                
                The wrapper preserves the original function's signature, docstring,
                and other metadata thanks to the @wraps decorator. Performance metrics
                are logged for both cache hits and misses, providing visibility into
                cache effectiveness.
                
                Args:
                    *args: Variable positional arguments to pass to the original function
                    **kwargs: Variable keyword arguments to pass to the original function
                    
                Returns:
                    The result of the wrapped function call, either from cache or freshly computed
                    
                Raises:
                    Any exceptions that the original function might raise
                    
                Examples:
                    >>> # This function is not called directly by users, but through the 
                    >>> # original function that was decorated with @cache_result
                    >>> 
                    >>> @cache.cache_result(ttl=60)
                    >>> def factorial(n: int) -> int:
                    ...     if n <= 1:
                    ...         return 1
                    ...     return n * factorial(n-1)
                    >>> 
                    >>> # Behind the scenes, the wrapper handles the caching logic
                    >>> result = factorial(5)  # Returns 120, either from cache or computed
                """
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
        - Function name or numeric ID (for objects without __name__)
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
            # Returns SHA256 hash as hexadecimal string
            
            >>> cache._generate_cache_key(example, (1,), {'b': 3})
            # Returns different hash due to changed arguments
            
            >>> # With complex objects:
            >>> cache._generate_cache_key(example, ([1,2],), {'b': {'x': 1}})
            # Returns hash based on string representation of complex objects
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
        
        This method determines whether a cached entry should be considered valid
        by checking several conditions:
        
        1. It first verifies that the cache entry contains a properly formatted timestamp
           in ISO 8601 format (e.g., "2023-10-15T14:30:00.000000+00:00").
        2. It then calculates the elapsed time since the cache entry was created by
           comparing the current time with the timestamp.
        3. Finally, it checks if the elapsed time is less than the TTL (time-to-live)
           value specified in the cache entry.
        
        The method handles various error cases gracefully:
        - If the timestamp is missing or empty, the entry is considered invalid.
        - If the timestamp is malformed or cannot be parsed, the entry is considered invalid.
        - If the TTL is not present in the entry, a default of 3600 seconds (1 hour) is used.
        
        This validation ensures that cache entries are automatically invalidated once they
        exceed their intended lifespan, preventing stale data from being returned to callers.
        
        Args:
            cached: The cache entry dictionary that must contain at minimum:
                - timestamp: ISO 8601 format datetime string when the entry was created
                - ttl: Time-to-live in seconds (positive integer)
                
        Returns:
            True if the cache entry is valid and has not exceeded its TTL, False otherwise
            
        Examples:
            Fresh entry within TTL:
            >>> import time
            >>> from datetime import UTC, datetime, timedelta
            >>> # Create an entry from 30 minutes ago with 1 hour TTL
            >>> recent_time = datetime.now(UTC) - timedelta(minutes=30)
            >>> entry = {
            ...     "timestamp": recent_time.isoformat(),
            ...     "ttl": 3600,  # 1 hour
            ...     "data": {"user_id": 123, "status": "active"}
            ... }
            >>> cache._is_cache_valid(entry)
            True
            
            Expired entry (TTL exceeded):
            >>> # Create an entry from 2 hours ago with 1 hour TTL
            >>> old_time = datetime.now(UTC) - timedelta(hours=2)
            >>> expired_entry = {
            ...     "timestamp": old_time.isoformat(),
            ...     "ttl": 3600,  # 1 hour
            ...     "data": {"user_id": 123, "status": "active"}
            ... }
            >>> cache._is_cache_valid(expired_entry)
            False
            
            Entry with very short TTL (already expired):
            >>> # Create a fresh entry but with 0 TTL (immediately expires)
            >>> entry_zero_ttl = {
            ...     "timestamp": datetime.now(UTC).isoformat(),
            ...     "ttl": 0,
            ...     "data": {"temp": "value"}
            ... }
            >>> cache._is_cache_valid(entry_zero_ttl)
            False
            
            Invalid timestamp format:
            >>> bad_format = {
            ...     "timestamp": "2023/10/15 14:30:00",  # Not ISO format
            ...     "ttl": 3600,
            ...     "data": {"example": "data"}
            ... }
            >>> cache._is_cache_valid(bad_format)
            False
            
            Missing timestamp:
            >>> missing_timestamp = {
            ...     "ttl": 3600,
            ...     "data": {"example": "data"}
            ... }
            >>> cache._is_cache_valid(missing_timestamp)
            False
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


# Global functions for direct checkpoint management
def create_checkpoint(
    key: str,
    data: Any,
    ttl: int = 3600,
    metadata: Dict[str, Any] | None = None
) -> None:
    """Create a checkpoint with the given key and data.
    
    This function provides a simplified interface for creating checkpoints without
    directly managing ProcessorCache instances. It handles the creation of an appropriate
    thread ID based on the key structure, instantiates a ProcessorCache with that thread ID,
    and then stores the data.
    
    Internally, this function uses ProcessorCache to store the data in both an in-memory
    cache and the LangGraph checkpoint system. This ensures data persistence across
    application restarts if LangGraph's persistence is configured.
    
    The function automatically logs checkpoint creation success or failure, enhancing
    observability without requiring additional logging code.
    
    Args:
        key: Unique identifier for the checkpoint, preferably in a namespaced format
             like "domain:id" (e.g., "user:123")
        data: The data to store in the checkpoint (any JSON-serializable type)
        ttl: Time-to-live in seconds, controlling how long the checkpoint remains valid
             (default: 3600 seconds, or 1 hour)
        metadata: Optional dictionary of metadata to associate with this checkpoint,
                 useful for tracking origin, purpose, or other attributes (default: None)
        
    Returns:
        None
        
    Examples:
        Basic usage:
        >>> create_checkpoint("session:user123", {"logged_in": True, "last_active": "2023-10-15"})
        
        With custom TTL:
        >>> create_checkpoint(
        ...     "config:app",
        ...     {"theme": "dark", "features": {"beta": True}},
        ...     ttl=86400  # 24 hours
        ... )
        
        With metadata and shorter expiration:
        >>> create_checkpoint(
        ...     "workflow:order456",
        ...     {
        ...         "status": "processing",
        ...         "steps_completed": ["payment", "inventory"],
        ...         "next_step": "shipping"
        ...     },
        ...     ttl=1800,  # 30 minutes
        ...     metadata={
        ...         "created_by": "order_processor",
        ...         "priority": "high",
        ...         "retry_count": 0
        ...     }
        ... )
        
        Error handling is automatic:
        >>> try:
        ...     # This will log any errors but won't raise exceptions to the caller
        ...     create_checkpoint("test:error", complex_object_with_circular_reference)
        ... except:
        ...     # No need for try/except blocks for normal checkpoint operations
        ...     pass
    """
    try:
        # Use thread ID based on key for isolation
        thread_id = f"checkpoint:{key.split(':')[0]}" if ':' in key else f"checkpoint:{key}"
        cache = ProcessorCache(thread_id=thread_id)
        cache.put(key, data, ttl=ttl, metadata=metadata)
        info_highlight(f"Created checkpoint: {key}")
    except Exception as e:
        warning_highlight(f"Error creating checkpoint {key}: {str(e)}")


def load_checkpoint(key: str) -> Any:
    """Load data from a checkpoint.
    
    This function retrieves previously stored data from a checkpoint using the provided key.
    It serves as a simplified interface to access checkpointed data without directly
    managing ProcessorCache instances.
    
    The function automatically determines the appropriate thread ID based on the key's
    structure (using the namespace before the colon), creates a ProcessorCache instance
    with that thread ID, and attempts to retrieve the data.
    
    If the checkpoint exists and is still valid (within its TTL), the function returns
    the stored data. If the checkpoint doesn't exist, has expired, or cannot be accessed
    due to errors, the function returns None.
    
    The function automatically logs checkpoint retrieval success or failure, enhancing
    observability without requiring additional logging code.
    
    Args:
        key: Unique identifier for the checkpoint, typically in a namespaced format
             like "domain:id" (e.g., "user:123")
        
    Returns:
        The data stored in the checkpoint if found and valid, otherwise None
        
    Examples:
        Basic retrieval:
        >>> user_session = load_checkpoint("session:user123")
        >>> if user_session:
        ...     # Session data was found and is still valid
        ...     is_logged_in = user_session.get("logged_in", False)
        ... else:
        ...     # No valid session found
        ...     is_logged_in = False
        
        Working with complex data:
        >>> app_config = load_checkpoint("config:app")
        >>> if app_config:
        ...     theme = app_config.get("theme", "light")
        ...     beta_features = app_config.get("features", {}).get("beta", False)
        
        Error handling is automatic:
        >>> try:
        ...     # This will log any errors but won't raise exceptions to the caller
        ...     workflow_data = load_checkpoint("workflow:nonexistent")
        ...     # workflow_data will be None if checkpoint doesn't exist
        ... except:
        ...     # No need for try/except blocks for normal checkpoint operations
        ...     pass
        
        Directly accessing nested data (with fallback):
        >>> order_data = load_checkpoint("order:12345")
        >>> status = order_data.get("status", "unknown") if order_data else "unknown"
        >>> print(f"Order status: {status}")
    """
    try:
        # Use thread ID based on key for isolation
        thread_id = f"checkpoint:{key.split(':')[0]}" if ':' in key else f"checkpoint:{key}"
        cache = ProcessorCache(thread_id=thread_id)
        data = cache.get(key)
        if data is not None:
            info_highlight(f"Loaded checkpoint: {key}")
        return data
    except Exception as e:
        warning_highlight(f"Error loading checkpoint {key}: {str(e)}")
        return None
