from datetime import datetime, timedelta, timezone

import pytest
from langgraph.checkpoint.memory import MemorySaver


class TestCodeUnderTest:
    """Tests for the cache module."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, mocker):
        """Set up mocks for all tests."""
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

    # Cache hit retrieves correct data from memory cache
    def test_memory_cache_hit_retrieves_correct_data(self, mocker):
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

    # Handling exceptions during checkpoint retrieval
    def test_exception_handling_during_checkpoint_retrieval(self, mocker):
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

    # Cache hit retrieves correct data from checkpoint
    def test_checkpoint_cache_hit_retrieves_correct_data(self, mocker):
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

    # Cache miss returns None and increments cache_misses counter
    def test_cache_miss_increments_counter(self, mocker):
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

    # put() successfully stores data in both memory cache and checkpoint
    def test_put_stores_data_in_memory_and_checkpoint(self, mocker):
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

    # cache_result decorator correctly caches function results
    def test_cache_result_decorator_caches_function_results(self, mocker):
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

    # Decorated function returns cached result on subsequent calls with same parameters
    def test_decorated_function_returns_cached_result(self, mocker):
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

    # _generate_cache_key creates unique keys for different function calls
    def test_generate_cache_key_creates_unique_keys(self, mocker):
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

    # _is_cache_valid correctly determines if cached entries are expired
    def test_is_cache_valid_expired_entry(self, mocker):
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

    # Empty or None values being cached correctly
    def test_cache_empty_none_values(self, mocker):
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