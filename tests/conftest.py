"""Test configuration for pytest."""
import os
from unittest import mock

import pytest


# Apply the mock before any tests run and before any modules are imported
mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-mock-key-for-testing"}).start()

# Mock the OpenAI client module
mock.patch("openai.AsyncClient", return_value=mock.MagicMock()).start()


@pytest.fixture(scope="session", autouse=True)
def mock_openai_client():
    """Mock the OpenAI client to avoid API key errors during tests."""
    # This patch will be applied before any tests run
    patcher = mock.patch("react_agent.utils.llm.openai_client")
    mock_client = patcher.start()
    
    # Configure any default behaviors for the mock client here
    mock_client.chat.completions.create.return_value = mock.MagicMock(
        choices=[mock.MagicMock(message=mock.MagicMock(content="Mocked response"))]
    )
    
    mock_client.embeddings.create.return_value = mock.MagicMock(
        data=[mock.MagicMock(embedding=[0.1, 0.2, 0.3])]
    )
    
    yield mock_client
    
    # Clean up the patch after all tests are done
    patcher.stop()
