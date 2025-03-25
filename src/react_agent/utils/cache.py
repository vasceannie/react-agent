"""Type-safe caching and checkpointing utilities.

This module provides a unified interface for caching and checkpointing with LangGraph,
ensuring type safety and consistent behavior across the application.

Example:
    @cache_result(ttl=600)
    def compute(a: int, b: int) -> int:
        return a + b

    # Using the decorated function
    result = compute(3, 4)  # Returns 7, either from cache or computed.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, cast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from functools import wraps

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from langgraph.checkpoint.memory import MemorySaver

from react_agent.utils.logging import get_logger

logger = get_logger(__name__)

# Generic type variables for strict type safety.
T = TypeVar('T')
K = TypeVar('K')

@dataclass
class CacheEntry(Generic[T]):
    """
    Represents a type-safe cache entry with metadata.
    
    Attributes:
        data (T): The cached data.
        timestamp (str): ISO formatted timestamp of when the data was cached.
        ttl (int): Time-to-live in seconds (default is 3600 seconds, i.e. 1 hour).
    """
    data: T
    timestamp: str
    ttl: int = 3600

class ProcessorCache:
    """
    A type-safe processor cache utilizing LangGraph checkpointing for persistent storage.

    Attributes:
        thread_id (str): Identifier for this cache instance.
        memory_saver (MemorySaver): An instance to handle checkpoint storage.
        cache_hits (int): Count of cache hits.
        cache_misses (int): Count of cache misses.
    """
    
    def __init__(self, thread_id: str = "default-processor") -> None:
        """
        Initialize the cache with a specific thread ID.

        Args:
            thread_id (str): Identifier for this cache instance.
        
        Example:
            >>> processor = ProcessorCache(thread_id="processor-1")
        """
        self.memory_saver: MemorySaver = MemorySaver()
        self.thread_id: str = thread_id
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    def _get_config(self, checkpoint_id: str) -> RunnableConfig:
        """
        Create a configuration object for checkpoint operations.

        Args:
            checkpoint_id (str): The unique identifier for the checkpoint.

        Returns:
            RunnableConfig: Configuration object containing thread and checkpoint IDs.
        """
        return RunnableConfig(configurable={
            "thread_id": self.thread_id,
            "checkpoint_id": checkpoint_id
        })

    def _create_checkpoint(
        self,
        key: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Checkpoint:
        """
        Generate a checkpoint object for the given data.

        Args:
            key (str): Unique identifier for the checkpoint.
            data (Dict[str, Any]): The data to store in the checkpoint.
            metadata (Optional[Dict[str, Any]]): Additional metadata if needed (default is None).

        Returns:
            Checkpoint: A checkpoint instance containing the provided data.
        """
        return Checkpoint(
            id=key,
            ts=datetime.now(timezone.utc).isoformat(),
            v=1,
            channel_values=data,
            channel_versions={},
            versions_seen={},
            pending_sends=[]
        )

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve data from the cache for a given key.

        Args:
            key (str): The cache key to lookup.

        Returns:
            Optional[Dict[str, Any]]: The cached data if present and valid, otherwise None.

        Example:
            >>> data = processor.get("some_cache_key")
            >>> if data is not None:
            ...     print("Cache hit!")
        """
        try:
            checkpoint: Optional[Any] = self.memory_saver.get(self._get_config(key))
            if checkpoint is not None and isinstance(checkpoint, dict):
                channel_values: Optional[Dict[str, Any]] = cast(Dict[str, Any], checkpoint.get("channel_values"))
                return channel_values
            return None
        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}")
            return None

    def put(
        self,
        key: str,
        data: Dict[str, Any],
        ttl: int = 3600,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store data in the cache under the provided key.

        Args:
            key (str): Unique identifier for the cache entry.
            data (Dict[str, Any]): The data to be cached.
            ttl (int): Time-to-live in seconds for the cache entry (default is 3600 seconds).
            metadata (Optional[Dict[str, Any]]): Additional metadata for checkpointing (default is None).

        Example:
            >>> processor.put("my_key", {"data": 42}, ttl=600)
        """
        checkpoint_metadata: CheckpointMetadata = CheckpointMetadata(
            source="input",
            step=-1,
            writes={},
            parents={}
        )

        if metadata is not None:
            # Assuming CheckpointMetadata is dict-like.
            checkpoint_metadata["writes"] = metadata

        checkpoint: Checkpoint = self._create_checkpoint(key, data)
        self.memory_saver.put(
            self._get_config(key),
            checkpoint,
            metadata=checkpoint_metadata,
            new_versions={}
        )

    def cache_result(self, ttl: int = 3600) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """
        Decorator to cache the result of a function call with type preservation.

        This decorator serializes the function's arguments to generate a unique key.
        If a cached result exists and is not expired, it returns the cached value.
        Otherwise, it executes the function, caches its output, and returns it.

        Args:
            ttl (int): Time-to-live in seconds for the cached result (default is 3600 seconds).

        Returns:
            Callable[[Callable[..., T]], Callable[..., T]]:
                A decorator that wraps a function with caching behavior.

        Example:
            >>> @processor.cache_result(ttl=600)
            ... def add(a: int, b: int) -> int:
            ...     return a + b
            >>>
            >>> result = add(2, 3)  # Returns 5, from cache on subsequent calls if within TTL.
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                # Generate a unique cache key from function name and arguments.
                cache_data: Dict[str, Any] = {'args': args, 'kwargs': kwargs, 'func': func.__name__}
                cache_str: str = json.dumps(cache_data, sort_keys=True)
                cache_key: str = hashlib.sha256(cache_str.encode()).hexdigest()

                cached: Optional[Dict[str, Any]] = self.get(cache_key)
                if cached is not None and cached.get("data") is not None:
                    timestamp_str: str = cached.get("timestamp", "")
                    if timestamp_str:
                        try:
                            timestamp: datetime = datetime.fromisoformat(timestamp_str)
                        except ValueError:
                            timestamp = datetime.min.replace(tzinfo=timezone.utc)
                        elapsed: float = (datetime.now(timezone.utc) - timestamp).total_seconds()
                        if elapsed < cached.get("ttl", ttl):
                            self.cache_hits += 1
                            return cast(T, cached["data"])
                self.cache_misses += 1
                result: T = func(*args, **kwargs)
                self.put(
                    cache_key,
                    {
                        "data": result,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "ttl": ttl
                    }
                )
                return result
            return wrapper
        return decorator

# Default processor instance.
default_processor: ProcessorCache = ProcessorCache()

def create_checkpoint(key: str, data: Dict[str, Any], ttl: int = 3600) -> None:
    """
    Create a checkpoint using the default processor.

    Args:
        key (str): Unique identifier for the checkpoint.
        data (Dict[str, Any]): The data to be checkpointed.
        ttl (int): Time-to-live for the checkpoint data (default is 3600 seconds).

    Example:
        >>> create_checkpoint("checkpoint1", {"result": 100})
    """
    default_processor.put(key, data, ttl=ttl)

def load_checkpoint(key: str) -> Optional[Dict[str, Any]]:
    """
    Load a checkpoint using the default processor.

    Args:
        key (str): Unique identifier for the checkpoint.

    Returns:
        Optional[Dict[str, Any]]: The checkpoint data if available, else None.

    Example:
        >>> data = load_checkpoint("checkpoint1")
        >>> if data is not None:
        ...     print("Checkpoint loaded:", data)
    """
    return default_processor.get(key)

def cache_result(ttl: int = 3600) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for caching function results using the default processor.

    Args:
        ttl (int): Time-to-live in seconds for the cached result (default is 3600 seconds).

    Returns:
        Callable[[Callable[..., T]], Callable[..., T]]:
            A decorator that applies caching to a function.

    Example:
        >>> @cache_result(ttl=600)
        ... def multiply(x: int, y: int) -> int:
        ...     return x * y
        >>>
        >>> print(multiply(3, 4))  # Outputs 12, and caches the result.
    """
    return default_processor.cache_result(ttl=ttl)
