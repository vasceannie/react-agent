import pytest

class TestCacheResult:

    # Function returns a decorator that caches results of the decorated function
    def test_cache_result_caches_function_output(self, mocker):
        # Arrange
        from src.react_agent.utils.cache import cache_result, default_processor
    
        # Mock the default processor's cache_result method
        mock_processor_cache_result = mocker.patch.object(
            default_processor, 'cache_result', wraps=default_processor.cache_result
        )
    
        # Create a function with a side effect to track calls
        mock_expensive_function = mocker.Mock(return_value=42)
    
        # Apply the cache decorator
        cached_function = cache_result(ttl=100)(mock_expensive_function)
    
        # Act
        result1 = cached_function(1, 2)
        result2 = cached_function(1, 2)  # Same args should use cache
        result3 = cached_function(3, 4)  # Different args should not use cache
    
        # Assert
        assert result1 == 42
        assert result2 == 42
        assert result3 == 42
        assert mock_processor_cache_result.call_count == 1
        assert mock_processor_cache_result.call_args[1]['ttl'] == 100
        assert mock_expensive_function.call_count > 1  # Called more than once due to different args

    # Caching with non-hashable function arguments
    def test_cache_result_with_non_hashable_arguments(self, mocker):
        # Arrange
        from src.react_agent.utils.cache import cache_result
    
        # Create a function that accepts non-hashable arguments (like lists)
        mock_function = mocker.Mock(return_value="result")
        cached_function = cache_result()(mock_function)
    
        # Act & Assert - Should not raise exceptions
        try:
            # Call with a non-hashable argument (list)
            result1 = cached_function([1, 2, 3])
            result2 = cached_function([1, 2, 3])  # Same list content
        
            # Both calls should return the same result
            assert result1 == "result"
            assert result2 == "result"
        
            # The underlying function should be called only once if caching works
            assert mock_function.call_count == 1
        except Exception as e:
            assert False, f"Cache should handle non-hashable arguments, but raised: {e}"

    # Cached results are returned on subsequent calls with same arguments
    def test_cached_results_returned_on_subsequent_calls(self, mocker):
        # Arrange
        from src.react_agent.utils.cache import cache_result, default_processor

        # Mock the default processor's cache_result method
        mock_processor_cache_result = mocker.patch.object(
            default_processor, 'cache_result', wraps=default_processor.cache_result
        )

        # Create a function with a side effect to track calls
        mock_function = mocker.Mock(return_value=10)

        # Apply the cache decorator
        cached_function = cache_result(ttl=200)(mock_function)

        # Act
        result1 = cached_function(5, 5)
        result2 = cached_function(5, 5)  # Same args should use cache

        # Assert
        assert result1 == 10
        assert result2 == 10
        assert mock_processor_cache_result.call_count == 1
        assert mock_processor_cache_result.call_args[1]['ttl'] == 200
        assert mock_function.call_count == 1  # Called once due to caching

    # Default TTL of 3600 seconds is applied when no TTL is specified
    def test_default_ttl_applied_when_not_specified(self, mocker):
        # Arrange
        from src.react_agent.utils.cache import cache_result, default_processor

        # Mock the default processor's cache_result method
        mock_processor_cache_result = mocker.patch.object(
            default_processor, 'cache_result', wraps=default_processor.cache_result
        )

        # Create a function with a side effect to track calls
        mock_function = mocker.Mock(return_value=42)

        # Apply the cache decorator without specifying TTL
        cached_function = cache_result()(mock_function)

        # Act
        result1 = cached_function(1, 2)
        result2 = cached_function(1, 2)  # Same args should use cache

        # Assert
        assert result1 == 42
        assert result2 == 42
        assert mock_processor_cache_result.call_count == 1
        assert mock_processor_cache_result.call_args[1]['ttl'] == 3600

    # Custom TTL is applied when specified
    def test_custom_ttl_applied(self, mocker):
        # Arrange
        from src.react_agent.utils.cache import cache_result, default_processor

        # Mock the default processor's cache_result method
        mock_processor_cache_result = mocker.patch.object(
            default_processor, 'cache_result', wraps=default_processor.cache_result
        )

        # Create a function with a side effect to track calls
        mock_function = mocker.Mock(return_value=10)

        # Apply the cache decorator with a custom TTL
        cached_function = cache_result(ttl=500)(mock_function)

        # Act
        result = cached_function(5, 5)

        # Assert
        assert result == 10
        assert mock_processor_cache_result.call_count == 1
        assert mock_processor_cache_result.call_args[1]['ttl'] == 500

    # Thread safety of the caching mechanism
    def test_cache_result_thread_safety(self, mocker):
        import threading
        from src.react_agent.utils.cache import cache_result, default_processor

        # Mock the default processor's cache_result method
        mock_processor_cache_result = mocker.patch.object(
            default_processor, 'cache_result', wraps=default_processor.cache_result
        )

        # Create a function with a side effect to track calls
        mock_expensive_function = mocker.Mock(return_value=42)

        # Apply the cache decorator
        cached_function = cache_result(ttl=100)(mock_expensive_function)

        # Define a thread target function
        def thread_target():
            for _ in range(10):
                cached_function(1, 2)

        # Create multiple threads to test thread safety
        threads = [threading.Thread(target=thread_target) for _ in range(5)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Assert that the cache was used and no race conditions occurred
        assert mock_processor_cache_result.call_count == 1
        assert mock_expensive_function.call_count == 1  # Should be called once due to caching