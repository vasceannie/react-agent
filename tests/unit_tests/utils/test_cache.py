"""Unit tests for the cache module.

This test suite verifies the functionality of the ProcessorCache class and its methods,
including caching behavior, checkpoint integration, and edge case handling.

Examples of test cases:
    - Basic cache operations (get/put)
    - Cache hit/miss scenarios
    - TTL expiration validation
    - Error handling cases
    - Decorator functionality
"""

from datetime import datetime, timedelta, timezone

import pytest
from langgraph.checkpoint.memory import MemorySaver


class TestCodeUnderTest:
    """Tests for the cache module.
    
    Test scenarios cover:
    - Memory cache operations
    - Checkpoint integration
    - Cache statistics tracking
    - Decorator behavior
    - Edge cases (empty values, exceptions)
    
    Example test cases:
        >>> test_memory_cache_hit_retrieves_correct_data()
        Verifies memory cache returns stored data
        
        >>> test_cache_result_decorator_caches_function_results()
        Verifies decorated functions cache results
    """

    @pytest.fixture(autouse=True)
    def setup_mocks(self, mocker):
        """Set up mocks for all tests.
        
        Initializes:
        - Mock modules to prevent import errors
        - Logger and metrics mocks
        - ProcessorCache instance for testing
        
        Example mock setup:
            >>> mocker.patch('openai.AsyncOpenAI')
            >>> mocker.patch('react_agent.utils.cache.logger')
        """
        # Mock all necessary modules to prevent import errors
        mocker.patch.dict('sys.modules', {
            'openai': mocker.MagicMock(),
            'openai.AsyncOpenAI': mocker.MagicMock(),
            'react_agent.graphs.graph': mocker.MagicMock(),
            'react_agent.utils.llm': mocker.MagicMock()
        })
        
        # Mock logger and metrics
        mocker.patch('src.react_agent.utils.cache.log_performance_metrics')
        mocker.patch('src.react_agent.utils.cache.logger')
        
        # Import ProcessorCache here to avoid module-level imports
        from src.react_agent.utils.cache import ProcessorCache
        self.ProcessorCache = ProcessorCache

    def test_memory_cache_hit_retrieves_correct_data(self, mocker):
        """Verify memory cache returns stored data for existing keys.
        
        Example:
            >>> cache.put("test-key", {"value": "test"})
            >>> cache.get("test-key") 
            {'value': 'test'}  # Returns stored data
        """
        # Arrange
        cache = self.ProcessorCache(thread_id="test-thread", version=1)
    
        # Create cache instance
        # cache = ProcessorCache(thread_id="test-thread", version=1)
    
        # Mock the logger to verify it's called
        mock_logger = mocker.patch("src.react_agent.utils.cache.logger")
    
        # Add data to memory cache directly
        test_key = "test-key"
        test_data = {"value": "test-value"}
        cache.memory_cache[test_key] = {
            "data": test_data,
            "timestamp": "2023-01-01T00:00:00+00:00",
            "ttl": 3600,
            "version": 1,
            "metadata": {}
        }
    
        # Act
        result = cache.get(test_key)
    
        # Assert
        assert result == test_data
        mock_logger.info.assert_any_call(f"Cache hit from memory for key: {test_key}")

    def test_exception_handling_during_checkpoint_retrieval(self, mocker):
        """Verify exception handling during checkpoint retrieval.
        
        Example:
            >>> cache.memory_saver.get.side_effect = Exception("Test error")
            >>> cache.get("key")  # Should handle exception gracefully
            None  # Returns None instead of raising exception
        """
        # Arrange
        # Create cache instance
        cache = self.ProcessorCache(thread_id="test-thread", version=1)
    
        # Mock the logger to verify it's called
        mock_logger = mocker.patch("src.react_agent.utils.cache.logger")
    
        # Mock memory_saver.get to raise an exception
        mock_get = mocker.patch.object(cache.memory_saver, "get")
        mock_get.side_effect = Exception("Test exception")
    
        # Mock log_performance_metrics to avoid actual logging
        mock_metrics = mocker.patch("src.react_agent.utils.cache.log_performance_metrics")
    
        # Act
        result = cache.get("non-existent-key")
    
        # Assert
        assert result is None
        mock_logger.error.assert_called_once()
        assert "Error retrieving from cache" in mock_logger.error.call_args[0][0]
        assert mock_metrics.called

    def test_checkpoint_cache_hit_retrieves_correct_data(self, mocker):
        """Verify checkpoint cache returns stored data for existing keys.
        
        Example:
            >>> mock_checkpoint = {
            ...     "values": {
            ...         "entry": {
            ...             "data": {"value": "test-value"},
            ...             "timestamp": "2023-01-01T00:00:00+00:00",
            ...             "ttl": 3600,
            ...             "version": 1,
            ...             "metadata": {}
            ...         }
            ...     }
            ... }
            >>> cache.get("test-key")  # Returns data from checkpoint
            {'value': 'test-value'}
        """
        # Arrange
        # Create cache instance
        cache = self.ProcessorCache(thread_id="test-thread", version=1)

        # Mock the logger to verify it's called
        mock_logger = mocker.patch("src.react_agent.utils.cache.logger")

        # Mock the memory_saver's get method to simulate a checkpoint hit
        test_key = "test-key"
        test_data = {"value": "test-value"}
        mock_checkpoint = {
            "values": {
                "entry": {
                    "data": test_data,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "ttl": 3600,
                    "version": 1,
                    "metadata": {}
                }
            }
        }
        mocker.patch.object(MemorySaver, 'get', return_value=mock_checkpoint)

        # Act
        result = cache.get(test_key)

        # Assert
        assert result == test_data
        mock_logger.info.assert_any_call(f"Cache hit from checkpoint for key: {test_key}")

    def test_cache_miss_increments_counter(self, mocker):
        """Verify cache miss returns None and increments counter.
        
        Example:
            >>> cache.cache_misses = 0
            >>> cache.get("nonexistent-key")  # Key doesn't exist
            None
            >>> cache.cache_misses
            1  # Counter incremented after miss
        """
        # Arrange
        # Create cache instance
        cache = self.ProcessorCache(thread_id="test-thread", version=1)
    
        # Mock the logger to verify it's called
        mock_logger = mocker.patch("src.react_agent.utils.cache.logger")
    
        # Mock the memory_saver to simulate a cache miss
        mocker.patch.object(cache.memory_saver, 'get', return_value=None)
    
        # Act
        result = cache.get("nonexistent-key")
    
        # Assert
        assert result is None
        assert cache.cache_misses == 1
        mock_logger.info.assert_any_call("Cache miss for key: nonexistent-key")

    def test_put_stores_data_in_memory_and_checkpoint(self, mocker):
        """Verify put() stores data in both memory cache and checkpoint.
        
        Example:
            >>> cache.put("test-key", {"value": "test"}, ttl=3600)
            >>> "test-key" in cache.memory_cache  # Stored in memory
            True
            >>> mock_memory_saver_put.assert_called_once()  # Stored in checkpoint
        """
        # Create cache instance
        cache = self.ProcessorCache(thread_id="test-thread", version=1)

        mock_logger = mocker.patch("src.react_agent.utils.cache.logger")
        mock_memory_saver_put = mocker.patch.object(MemorySaver, 'put')

        test_key = "test-key"
        test_data = {"value": "test-value"}
        test_ttl = 3600
        test_metadata = {"info": "test"}

        # Act
        cache.put(test_key, test_data, ttl=test_ttl, metadata=test_metadata)

        # Assert
        assert test_key in cache.memory_cache
        assert cache.memory_cache[test_key]["data"] == test_data
        assert cache.memory_cache[test_key]["ttl"] == test_ttl
        assert cache.memory_cache[test_key]["metadata"] == test_metadata

        mock_memory_saver_put.assert_called_once()
        mock_logger.info.assert_any_call(f"Stored data in cache for key: {test_key}")

    def test_cache_result_decorator_caches_function_results(self, mocker):
        """Verify decorated functions cache their return values.
        
        Example:
            >>> @cache.cache_result(ttl=600)
            >>> def add(a, b): return a + b
            >>> add(3, 4)  # First call computes and caches
            7
            >>> add(3, 4)  # Second call returns cached result
            7 (from cache)
        """
        # Create cache instance
        cache = self.ProcessorCache(thread_id="test-thread", version=1)
        
        # Arrange
        @cache.cache_result(ttl=600)
        def add(a, b):
            return a + b
            
        # Act
        result1 = add(3, 4)  # Should compute and cache
        result2 = add(3, 4)  # Should retrieve from cache
            
        # Assert
        assert result1 == 7
        assert result2 == 7
        assert cache.cache_hits == 1

    def test_decorated_function_returns_cached_result(self, mocker):
        """Verify decorated function returns cached result on subsequent calls.
        
        Example:
            >>> @cache.cache_result(ttl=600)
            >>> def expensive_operation(x): 
            ...     print("Computing...")
            ...     return x * 2
            >>> expensive_operation(5)  # Prints "Computing..." and returns 10
            >>> expensive_operation(5)  # Returns 10 without printing "Computing..."
        """
        # Create cache instance
        cache = self.ProcessorCache(thread_id="test-thread", version=1)
    
        # Mock the logger to verify it's called
        mock_logger = mocker.patch("src.react_agent.utils.cache.logger")
    
        # Mock the time module in the cache module with enough values
        # The cache.put method calls time.time() multiple times
        mocker.patch("src.react_agent.utils.cache.time.time", side_effect=[
            1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010
        ])
    
        # Define a simple function to be decorated
        @cache.cache_result(ttl=600)
        def add(a, b):
            return a + b
    
        # Act
        first_result = add(3, 4)  # This should compute and cache the result
        second_result = add(3, 4)  # This should retrieve from cache
    
        # Assert
        assert first_result == 7
        assert second_result == 7
        assert mock_logger.info.call_count >= 1  # At least one log message for cache hit

    def test_generate_cache_key_creates_unique_keys(self, mocker):
        """Verify cache keys are unique for different inputs.
        
        Example:
            >>> key1 = cache._generate_cache_key(func, (1, 2), {})
            >>> key2 = cache._generate_cache_key(func, (2, 3), {})
            >>> key1 != key2  # Different args generate different keys
            True
        """
        # Create cache instance
        cache = self.ProcessorCache(thread_id="test-thread", version=1)
        mock_func = mocker.Mock(__name__='mock_func')
    
        # Act
        key1 = cache._generate_cache_key(mock_func, (1, 2), {'param': 'value'})
        key2 = cache._generate_cache_key(mock_func, (2, 3), {'param': 'value'})
        key3 = cache._generate_cache_key(mock_func, (1, 2), {'param': 'different_value'})
    
        # Assert
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_is_cache_valid_expired_entry(self, mocker):
        """Verify cache validation correctly identifies expired entries.
        
        Example:
            >>> entry = {
            ...     "timestamp": "2023-01-01T00:00:00+00:00",  # Old timestamp
            ...     "ttl": 3600,
            ...     "data": {...}
            ... }
            >>> cache._is_cache_valid(entry)
            False  # Because entry is expired
        """
        # Create cache instance
        cache = self.ProcessorCache(thread_id="test-thread", version=1)

        # Create a fixed datetime for testing
        fixed_now = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        
        # Mock datetime.now in the cache module
        mocker.patch("src.react_agent.utils.cache.datetime", wraps=datetime)
        mocker.patch("src.react_agent.utils.cache.datetime.now", return_value=fixed_now)

        # Create a cache entry that is expired
        expired_timestamp = (fixed_now - timedelta(seconds=4000)).isoformat()
        expired_entry = {
            "data": {"value": "expired"},
            "timestamp": expired_timestamp,
            "ttl": 3600,
            "version": 1,
            "metadata": {}
        }

        # Act
        is_valid = cache._is_cache_valid(expired_entry)

        # Assert
        assert not is_valid

    def test_cache_empty_none_values(self, mocker):
        """Verify empty and None values are cached correctly.
        
        Example:
            >>> cache.put("empty-key", "")
            >>> cache.put("none-key", None)
            >>> cache.get("empty-key")
            ''  # Empty string retrieved correctly
            >>> cache.get("none-key")
            None  # None value retrieved correctly
        """
        # Create cache instance
        cache = self.ProcessorCache(thread_id="test-thread", version=1)

        # Mock the logger to verify it's called
        mock_logger = mocker.patch("src.react_agent.utils.cache.logger")

        # Add empty and None values to memory cache directly
        empty_key = "empty-key"
        none_key = "none-key"
        cache.memory_cache[empty_key] = {
            "data": "",
            "timestamp": "2023-01-01T00:00:00+00:00",
            "ttl": 3600,
            "version": 1,
            "metadata": {}
        }
        cache.memory_cache[none_key] = {
            "data": None,
            "timestamp": "2023-01-01T00:00:00+00:00",
            "ttl": 3600,
            "version": 1,
            "metadata": {}
        }

        # Act
        empty_result = cache.get(empty_key)
        none_result = cache.get(none_key)

        # Assert
        assert empty_result == ""
        assert none_result is None
        mock_logger.info.assert_any_call(f"Cache hit from memory for key: {empty_key}")
        mock_logger.info.assert_any_call(f"Cache hit from memory for key: {none_key}")
