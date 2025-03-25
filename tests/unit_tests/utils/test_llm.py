import pytest
from typing import Any, Dict, List, cast

from langchain_core.runnables import RunnableConfig
from react_agent.utils.llm import LLMClient, Message


class TestLLMModule:
    """Unit tests for the LLM module functionality."""

    # LLMClient.llm_chat returns text response from the model
    @pytest.mark.asyncio
    async def test_llm_chat_returns_text_response(self, mocker: Any) -> None:
        # 1. Mock the _call_model function to return a predefined response
        mock_call_model = mocker.patch("react_agent.utils.llm._call_model")
        mock_call_model.return_value = {"content": "This is a test response"}
    
        # 2. Create an instance of LLMClient with a default model
        client = LLMClient(default_model="openai/gpt-4o")
    
        # 3. Call the llm_chat method with a test prompt
        result = await client.llm_chat("Test prompt", system_prompt="Test system prompt")
    
        # 4. Assert that _call_model was called with the correct parameters
        mock_call_model.assert_called_once()
    
        # 5. Assert that the result is the expected string from the mocked response
        assert result == "This is a test response"
    
        # 6. Assert that the result is a string
        assert isinstance(result, str)

    # LLMClient.llm_json returns structured JSON data from the model
    @pytest.mark.asyncio
    async def test_llm_json_returns_structured_data(self, mocker: Any) -> None:
        # 1. Mock the _call_model_json function to return a predefined JSON response
        mock_json_response = {"key1": "value1", "key2": 42}
        mock_call_model_json = mocker.patch("react_agent.utils.llm._call_model_json")
        mock_call_model_json.return_value = mock_json_response
    
        # 2. Create an instance of LLMClient
        client = LLMClient(default_model="openai/gpt-4o")
    
        # 3. Call the llm_json method with a test prompt
        result = await client.llm_json("Test JSON prompt", system_prompt="Test system prompt")
    
        # 4. Assert that _call_model_json was called with the correct parameters
        mock_call_model_json.assert_called_once()
    
        # 5. Assert that the result is the expected dictionary
        assert result == mock_json_response
    
        # 6. Assert that the result is a dictionary
        assert isinstance(result, dict)

    # LLMClient.llm_embed returns embedding vectors for input text
    @pytest.mark.asyncio
    async def test_llm_embed_returns_vectors(self, mocker: Any) -> None:
        # 1. Create a mock embedding response
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    
        # 2. Mock the OpenAI client's embeddings.create method
        mock_openai_response = mocker.MagicMock()
        mock_openai_response.data = [mocker.MagicMock(embedding=mock_embedding)]
    
        # 3. Mock the openai_client instance
        mock_openai_client = mocker.patch("react_agent.utils.llm.openai_client")
        mock_openai_client.embeddings.create.return_value = mock_openai_response
    
        # 4. Mock Configuration.from_runnable_config to return a configuration with openai model
        mock_config = mocker.MagicMock()
        mock_config.model = "openai/text-embedding-ada-002"
        mock_from_config = mocker.patch("react_agent.utils.llm.Configuration.from_runnable_config")
        mock_from_config.return_value = mock_config
    
        # 5. Create an instance of LLMClient
        client = LLMClient()
    
        # 6. Call the llm_embed method with test text
        result = await client.llm_embed("Test text for embedding")
    
        # 7. Assert that the embedding method was called with the correct model and input
        mock_openai_client.embeddings.create.assert_called_once()
    
        # 8. Assert that the result matches the mock embedding
        assert result == mock_embedding
    
        # 9. Assert that the result is a list of floats
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    # Empty message list handling
    @pytest.mark.asyncio
    async def test_empty_message_list_handling(self, mocker: Any) -> None:
        # 1. Mock the error_highlight function to track calls
        mock_error_highlight = mocker.patch("react_agent.utils.llm.error_highlight")
    
        # 2. Import the functions before mocking to get the original references
        from react_agent.utils.llm import _call_model, _call_model_json
    
        # 3. Call _call_model with an empty message list
        await _call_model([])
    
        # 4. Assert that error_highlight was called with the appropriate error message
        assert mock_error_highlight.call_args_list[0][0][0] == "No messages provided to _call_model"
    
        # 5. Reset the mock to clear call history
        mock_error_highlight.reset_mock()
    
        # 6. Call _call_model_json with an empty message list
        await _call_model_json([])
    
        # 7. Assert that error_highlight was called with the appropriate error message
        assert mock_error_highlight.call_args_list[0][0][0] == "No messages provided to _call_model_json"

    # API errors and retry mechanism
    @pytest.mark.asyncio
    async def test_api_errors_and_retry_mechanism(self, mocker: Any) -> None:
        # 1. Mock the _call_model function to raise an exception on first call and succeed on second
        mock_call_model = mocker.patch("react_agent.utils.llm._call_model")
        mock_call_model.side_effect = [Exception("API Error"), {"content": '{"result": "success"}'}]
    
        # 2. Mock warning_highlight and error_highlight to track calls
        mock_warning = mocker.patch("react_agent.utils.llm.warning_highlight")
        mock_error = mocker.patch("react_agent.utils.llm.error_highlight")
    
        # 3. Mock asyncio.sleep to avoid actual delays
        mock_sleep = mocker.patch("asyncio.sleep")
    
        # 4. Mock safe_json_parse to return the parsed content
        mock_safe_json_parse = mocker.patch("react_agent.utils.llm.safe_json_parse")
        mock_safe_json_parse.return_value = {"result": "success"}
    
        # 5. Create test messages
        messages: List[Message] = [
            {"role": "system", "content": "Test system"},
            {"role": "user", "content": "Test content"}
        ]
    
        # 6. Call _process_chunk which should trigger the retry mechanism
        from react_agent.utils.llm import _process_chunk
        result = await _process_chunk("Test chunk", messages, None)
    
        # 7. Assert that warning_highlight was called indicating a retry
        mock_warning.assert_called_once()
        assert "Retrying" in mock_warning.call_args[0][0]
    
        # 8. Assert that asyncio.sleep was called for the retry delay
        mock_sleep.assert_called_once()
    
        # 9. Assert that the final result contains the expected data
        assert result == {"result": "success"}

    # Configuration is properly loaded from RunnableConfig
    @pytest.mark.asyncio
    async def test_configuration_loaded_from_runnable_config(self, mocker: Any) -> None:
        # 1. Mock the ensure_config function to return a predefined configuration
        mock_ensure_config = mocker.patch("react_agent.configuration.ensure_config")
        mock_ensure_config.return_value = {"configurable": {"model": "openai/gpt-4o"}}

        # 2. Mock the os.getenv function to return specific environment variables
        mock_getenv = mocker.patch("os.getenv")
        mock_getenv.side_effect = lambda key, default=None: {
            "FIRECRAWL_API_KEY": "test_firecrawl_key",
            "FIRECRAWL_URL": "https://test.firecrawl.url",
            "JINA_API_KEY": "test_jina_key",
            "JINA_URL": "https://test.jina.url",
            "ANTHROPIC_API_KEY": "test_anthropic_key"
        }.get(key, default)

        # 3. Create a RunnableConfig object with a specific configuration
        config: RunnableConfig = cast(RunnableConfig, {"configurable": {"model": "openai/gpt-4o"}})

        # 4. Call the from_runnable_config method to create a Configuration instance
        from react_agent.utils.llm import Configuration
        configuration = Configuration.from_runnable_config(config)

        # 5. Assert that the configuration model is correctly set from the RunnableConfig
        assert configuration.model == "openai/gpt-4o"

        # 6. Assert that the firecrawl_api_key is correctly loaded from environment variables
        assert configuration.firecrawl_api_key == "test_firecrawl_key"

        # 7. Assert that the firecrawl_url is correctly loaded from environment variables
        assert configuration.firecrawl_url == "https://test.firecrawl.url"

        # 8. Assert that the jina_api_key is correctly loaded from environment variables
        assert configuration.jina_api_key == "test_jina_key"

        # 9. Assert that the jina_url is correctly loaded from environment variables
        assert configuration.jina_url == "https://test.jina.url"

        # 10. Assert that the anthropic_api_key is correctly loaded from environment variables
        assert configuration.anthropic_api_key == "test_anthropic_key"

    # Messages are correctly formatted for different providers (OpenAI vs Anthropic)
    @pytest.mark.asyncio
    async def test_message_formatting_for_providers(self, mocker: Any) -> None:
        # 1. Mock the _format_openai_messages function to return a predefined formatted message
        mock_format_openai_messages = mocker.patch("react_agent.utils.llm._format_openai_messages")
        mock_format_openai_messages.return_value = [{"role": "user", "content": "Formatted OpenAI message"}]
    
        # 2. Mock the _ensure_system_message function to return a predefined message list
        mock_ensure_system_message = mocker.patch("react_agent.utils.llm._ensure_system_message")
        mock_ensure_system_message.return_value = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"}
        ]
        
        # 3. Mock _call_model to prevent actual API calls
        mock_call_model = mocker.patch("react_agent.utils.llm._call_model")
        mock_call_model.return_value = {"content": "Test response"}
    
        # 4. Create an instance of LLMClient with a default model
        client = LLMClient(default_model="openai/gpt-4o")
    
        # 5. Call the llm_chat method with a test prompt and system prompt
        await client.llm_chat("Test prompt", system_prompt="Test system prompt")
    
        # 6. Assert that _format_openai_messages was called with the correct parameters
        assert len(mock_format_openai_messages.call_args_list) == 1
        args = mock_format_openai_messages.call_args_list[0][0]
        assert args[0] == [{"role": "user", "content": "Test prompt"}]
        assert args[1] == "Test system prompt"

    # Invalid JSON responses from the model
    @pytest.mark.asyncio
    async def test_llm_json_handles_invalid_json_response(self, mocker: Any) -> None:
        # 1. Mock the _call_model_json function to return an invalid JSON response
        mock_call_model_json = mocker.patch("react_agent.utils.llm._call_model_json")
        mock_call_model_json.return_value = {"content": "Invalid JSON response"}

        # 2. Create an instance of LLMClient with a default model
        client = LLMClient(default_model="openai/gpt-4o")

        # 3. Call the llm_json method with a test prompt
        result = await client.llm_json("Test prompt", system_prompt="Test system prompt")

        # 4. Assert that _call_model_json was called with the correct parameters
        mock_call_model_json.assert_called_once()

        # 5. Assert that the result is an empty dictionary due to invalid JSON handling
        assert result == {}

        # 6. Assert that the result is a dictionary
        assert isinstance(result, dict)

    # System prompts are properly included in message formatting
    @pytest.mark.asyncio
    async def test_ensure_system_message_inclusion(self, mocker: Any) -> None:
        # 1. Mock the _ensure_system_message function to return a predefined list of messages
        mock_ensure_system_message = mocker.patch("react_agent.utils.llm._ensure_system_message")
        mock_ensure_system_message.return_value = [
            {"role": "system", "content": "Test system prompt"},
            {"role": "user", "content": "Test user message"}
        ]

        # 2. Create a list of messages without a system message
        messages: List[Message] = [{"role": "user", "content": "Test user message"}]

        # 3. Call the _ensure_system_message function with the messages and a system prompt
        from react_agent.utils.llm import _ensure_system_message
        result = await _ensure_system_message(messages, "Test system prompt")

        # 4. Assert that _ensure_system_message was called with the correct parameters
        mock_ensure_system_message.assert_called_once_with(messages, "Test system prompt")

        # 5. Assert that the result includes the system message
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Test system prompt"

        # 6. Assert that the result is a list of messages
        assert isinstance(result, list)
        assert all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in result)

    # Handling of empty or whitespace-only content
    @pytest.mark.asyncio
    async def test_llm_chat_handles_empty_content(self, mocker: Any) -> None:
        # 1. Mock the _call_model function to return an empty response
        mock_call_model = mocker.patch("react_agent.utils.llm._call_model")
        mock_call_model.return_value = {"content": ""}

        # 2. Create an instance of LLMClient with a default model
        client = LLMClient(default_model="openai/gpt-4o")

        # 3. Call the llm_chat method with an empty prompt
        result = await client.llm_chat("", system_prompt="Test system prompt")

        # 4. Assert that _call_model was called with the correct parameters
        mock_call_model.assert_called_once()

        # 5. Assert that the result is an empty string as expected
        assert result == ""

        # 6. Assert that the result is a string
        assert isinstance(result, str)

    # JSON response cleaning and parsing
    @pytest.mark.asyncio
    async def test_parse_json_response_handles_string_input(self, mocker: Any) -> None:
        # 1. Define a sample JSON string with extra formatting
        json_string = '{"key": "value"}'
    
        # 2. Mock the json.loads function to simulate JSON parsing
        mock_json_loads = mocker.patch("json.loads")
        mock_json_loads.return_value = {"key": "value"}
    
        # 3. Call the _parse_json_response function with the sample JSON string
        from react_agent.utils.llm import _parse_json_response
        result: Dict[str, Any] = _parse_json_response(json_string)
    
        # 4. Assert that json.loads was called once with the cleaned JSON string
        mock_json_loads.assert_called_once_with(json_string)
    
        # 5. Assert that the result is the expected dictionary from the mocked json.loads
        assert result == {"key": "value"}
    
        # 6. Assert that the result is a dictionary
        assert isinstance(result, dict)