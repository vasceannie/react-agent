"""Type-safe caching and checkpointing utilities.

This module provides a unified interface for caching and checkpointing with LangGraph,
ensuring type safety and consistent behavior across the application.
"""

from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Union, cast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import pickle
from pathlib import Path
import logging
from functools import wraps

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from langgraph.checkpoint.memory import MemorySaver

from react_agent.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# Type variables for generic type safety
T = TypeVar('T')
K = TypeVar('K')

@dataclass
class CacheEntry(Generic[T]):
    """Type-safe cache entry with metadata."""
    data: T
    timestamp: str
    ttl: int = 3600  # Default 1 hour TTL

class ProcessorCache:
    """Type-safe processor cache using LangGraph checkpointing."""
    
    def __init__(self, thread_id: str = "default-processor") -> None:
        """Initialize the cache with a specific thread ID.
        
        Args:
            thread_id: Identifier for this cache instance
        """
        self.memory_saver = MemorySaver()
        self.thread_id = thread_id

    def _get_config(self, checkpoint_id: str) -> RunnableConfig:
        """Create a RunnableConfig for checkpoint operations."""
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
        """Create a checkpoint with proper typing."""
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
        """Get data from cache with type safety."""
        try:
            checkpoint = self.memory_saver.get(self._get_config(key))
            if checkpoint and isinstance(checkpoint, dict):
                return cast(Dict[str, Any], checkpoint.get("channel_values"))
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
        """Store data in cache with type safety."""
        checkpoint_metadata = CheckpointMetadata(
            source="input",
            step=-1,
            writes={},
            parents={}
        )

        if metadata:
            checkpoint_metadata["writes"] = metadata

        checkpoint = self._create_checkpoint(key, data)
        
        self.memory_saver.put(
            self._get_config(key),
            checkpoint,
            metadata=checkpoint_metadata,
            new_versions={}
        )

    def cache_result(self, ttl: int = 3600) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator for caching function results with type preservation."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                # Generate cache key
                cache_data = {'args': args, 'kwargs': kwargs, 'func': func.__name__}
                cache_str = json.dumps(cache_data, sort_keys=True)
                cache_key = hashlib.sha256(cache_str.encode()).hexdigest()

                # Check cache
                if cached := self.get(cache_key):
                    if cached.get("data") is not None:
                        timestamp = datetime.fromisoformat(cached.get("timestamp", ""))
                        if (datetime.now(timezone.utc) - timestamp).total_seconds() < cached.get("ttl", ttl):
                            return cast(T, cached["data"])

                # Execute function
                result = func(*args, **kwargs)

                # Store in cache
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

# Default processor instance
default_processor = ProcessorCache()

# Type-safe utility functions
def create_checkpoint(key: str, data: Dict[str, Any], ttl: int = 3600) -> None:
    """Create a checkpoint with the default processor."""
    default_processor.put(key, data, ttl=ttl)

def load_checkpoint(key: str) -> Optional[Dict[str, Any]]:
    """Load a checkpoint with the default processor."""
    return default_processor.get(key)

def cache_result(ttl: int = 3600) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for caching results with the default processor."""
    return default_processor.cache_result(ttl=ttl)
