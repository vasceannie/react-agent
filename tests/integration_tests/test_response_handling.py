import pytest

from react_agent.utils.llm import LLMClient


class TestResponseHandling:
    """Integration tests for response handling across multiple modules."""

    # Token limit exceeded requiring content summarization
    @pytest.mark.asyncio
    async def test_token_limit_exceeded_content_summarization(self, mocker):
        # 1. Mock estimate_tokens to simulate exceeding token limit
        mock_estimate_tokens = mocker.patch("react_agent.utils.llm.estimate_tokens")
        mock_estimate_tokens.return_value = 16000 + 1000  # Exceed the limit
    
        # 2. Mock _summarize_content to return a summarized version
        mock_summarize = mocker.patch("react_agent.utils.llm._summarize_content")
        mock_summarize.return_value = "Summarized content"
    
        # 3. Mock info_highlight to track calls
        mock_info = mocker.patch("react_agent.utils.llm.info_highlight")
    
        # 4. Mock openai_client.chat.completions.create for the summarization call
        mock_openai_response = mocker.MagicMock()
        mock_openai_response.choices = [mocker.MagicMock(message=mocker.MagicMock(content="Summarized content"))]
        mock_openai_client = mocker.patch("react_agent.utils.llm.openai_client")
        mock_openai_client.chat.completions.create.return_value = mock_openai_response
    
        # 5. Create a long message that exceeds token limits
        long_message = {"role": "user", "content": "Very long content" * 1000}
        messages = [{"role": "system", "content": "System prompt"}, long_message]
    
        # 6. Mock _call_openai_api to return a successful response
        mock_call_openai = mocker.patch("react_agent.utils.llm._call_openai_api")
        mock_call_openai.return_value = {"content": "API response"}
    
        # 7. Call _format_openai_messages which should trigger summarization
        from react_agent.utils.llm import _format_openai_messages
        formatted_messages = await _format_openai_messages(messages, "System prompt")
    
        # 8. Assert that info_highlight was called indicating summarization
        mock_info.assert_called_with("Content too long, summarizing...")
    
        # 9. Assert that _summarize_content was called
        mock_summarize.assert_called_once()
    
        # 10. Assert that the formatted messages contain the summarized content
        assert any(msg["content"] == "Summarized content" for msg in formatted_messages)

    # Content chunking for large inputs with proper overlap
    @pytest.mark.asyncio
    async def test_llm_json_handles_large_content_with_chunking(self, mocker):
        # 1. Mock the chunk_text function to simulate chunking behavior
        mock_chunk_text = mocker.patch("react_agent.utils.content.chunk_text")
        mock_chunk_text.return_value = ["chunk1", "chunk2", "chunk3"]

        # 2. Mock the _process_chunk function to return a predefined response for each chunk
        mock_process_chunk = mocker.patch("react_agent.utils.llm._process_chunk")
        mock_process_chunk.side_effect = [
            {"content": "Processed chunk1"},
            {"content": "Processed chunk2"},
            {"content": "Processed chunk3"}
        ]

        # 3. Mock the merge_chunk_results function to simulate merging of chunk results
        mock_merge_chunk_results = mocker.patch("react_agent.utils.content.merge_chunk_results")
        mock_merge_chunk_results.return_value = {"content": "Merged content"}

        # 4. Create an instance of LLMClient with a default model
        client = LLMClient(default_model="openai/gpt-4o")

        # 5. Call the llm_json method with a large content prompt
        result = await client.llm_json("Large content prompt", system_prompt="Test system prompt")

        # 6. Assert that chunk_text was called with the correct parameters
        mock_chunk_text.assert_called_once_with(
            "Large content prompt", chunk_size=None, overlap=None, use_large_chunks=True
        )

        # 7. Assert that _process_chunk was called the expected number of times
        assert mock_process_chunk.call_count == 3

        # 8. Assert that merge_chunk_results was called with the correct parameters
        mock_merge_chunk_results.assert_called_once_with(
            [{"content": "Processed chunk1"}, {"content": "Processed chunk2"}, {"content": "Processed chunk3"}],
            "model_response"
        )

        # 9. Assert that the result is the expected merged content
        assert result == {"content": "Merged content"}

    # Long content is automatically chunked and results are merged
    @pytest.mark.asyncio
    async def test_llm_json_handles_long_content(self, mocker):
        # 1. Mock the chunk_text function to simulate chunking behavior
        mock_chunk_text = mocker.patch("react_agent.utils.content.chunk_text")
        mock_chunk_text.return_value = ["chunk1", "chunk2"]

        # 2. Mock the _process_chunk function to return predefined results for each chunk
        mock_process_chunk = mocker.patch("react_agent.utils.llm._process_chunk")
        mock_process_chunk.side_effect = [
            {"key": "value1"},
            {"key": "value2"}
        ]

        # 3. Mock the merge_chunk_results function to simulate merging behavior
        mock_merge_chunk_results = mocker.patch("react_agent.utils.content.merge_chunk_results")
        mock_merge_chunk_results.return_value = {"key": "merged_value"}

        # 4. Create an instance of LLMClient with a default model
        client = LLMClient(default_model="openai/gpt-4o")

        # 5. Call the llm_json method with a long prompt
        result = await client.llm_json("Long content that needs chunking", system_prompt="Test system prompt")

        # 6. Assert that chunk_text was called with the correct parameters
        mock_chunk_text.assert_called_once_with(
            "Long content that needs chunking", 
            chunk_size=None, 
            overlap=None, 
            use_large_chunks=True
        )

        # 7. Assert that _process_chunk was called twice (once for each chunk)
        assert mock_process_chunk.call_count == 2

        # 8. Assert that merge_chunk_results was called with the correct parameters
        mock_merge_chunk_results.assert_called_once_with(
            [{"key": "value1"}, {"key": "value2"}], 
            "model_response"
        )

        # 9. Assert that the result is the expected merged dictionary
        assert result == {"key": "merged_value"}

        # 10. Assert that the result is a dictionary
        assert isinstance(result, dict)

    # Performance tracking for large content processing
    @pytest.mark.asyncio
    async def test_llm_json_handles_large_content_performance(self, mocker):
        # 1. Mock the chunk_text function to simulate chunking of large content
        mock_chunk_text = mocker.patch("react_agent.utils.content.chunk_text")
        mock_chunk_text.return_value = ["chunk1", "chunk2", "chunk3"]

        # 2. Mock the _process_chunk function to return a predefined response for each chunk
        mock_process_chunk = mocker.patch("react_agent.utils.llm._process_chunk")
        mock_process_chunk.side_effect = [
            {"key": "value1"},
            {"key": "value2"},
            {"key": "value3"}
        ]

        # 3. Mock the merge_chunk_results function to simulate merging of chunk results
        mock_merge_chunk_results = mocker.patch("react_agent.utils.content.merge_chunk_results")
        mock_merge_chunk_results.return_value = {"key": "merged_value"}

        # 4. Create an instance of LLMClient with a default model
        client = LLMClient(default_model="openai/gpt-4o")

        # 5. Call the llm_json method with a large content prompt
        result = await client.llm_json("Large content prompt", system_prompt="Test system prompt")

        # 6. Assert that chunk_text was called with the correct parameters
        mock_chunk_text.assert_called_once()

        # 7. Assert that _process_chunk was called the expected number of times
        assert mock_process_chunk.call_count == 3

        # 8. Assert that merge_chunk_results was called once to merge the chunk results
        mock_merge_chunk_results.assert_called_once()

        # 9. Assert that the result is the expected merged dictionary from the mocked response
        assert result == {"key": "merged_value"}

        # 10. Assert that the result is a dictionary
        assert isinstance(result, dict)

    # Provider-specific message format conversion with summarization
    @pytest.mark.asyncio
    async def test_format_openai_messages_summarizes_long_content(self, mocker):
        # 1. Mock the _ensure_system_message function to return the input messages unchanged
        mock_ensure_system_message = mocker.patch("react_agent.utils.llm._ensure_system_message")
        mock_ensure_system_message.return_value = [{"role": "user", "content": "a" * 17000}]

        # 2. Mock the _summarize_content function to return a shortened version of the content
        mock_summarize_content = mocker.patch("react_agent.utils.llm._summarize_content")
        mock_summarize_content.return_value = "Summarized content"

        # 3. Create a list of messages with content exceeding the token limit
        messages = [{"role": "user", "content": "a" * 17000}]

        # 4. Call the _format_openai_messages function with the test messages and a system prompt
        from react_agent.utils.llm import _format_openai_messages
        result = await _format_openai_messages(messages, "Test system prompt")

        # 5. Assert that _ensure_system_message was called once with the correct parameters
        mock_ensure_system_message.assert_called_once_with(messages, "Test system prompt")

        # 6. Assert that _summarize_content was called once with the correct parameters
        mock_summarize_content.assert_called_once()

        # 7. Assert that the result contains the summarized content
        assert result == [{"role": "user", "content": "Summarized content"}]