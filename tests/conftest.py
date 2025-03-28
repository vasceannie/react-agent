"""Test configuration for pytest."""
import os
import sys
from unittest import mock
from types import ModuleType, SimpleNamespace

import pytest

# Create a mock for the react_agent module (both namespaces)
mock_react_agent = mock.MagicMock()
sys.modules['react_agent'] = mock_react_agent
sys.modules['react_agent.graphs'] = mock.MagicMock()
sys.modules['react_agent.graphs.graph'] = mock.MagicMock()
sys.modules['react_agent.graphs.graph'].graph = mock.MagicMock()
sys.modules['react_agent.utils'] = mock.MagicMock()
sys.modules['react_agent.utils.llm'] = mock.MagicMock()
sys.modules['react_agent.utils.llm'].LLMClient = mock.MagicMock()
sys.modules['react_agent.utils.llm'].Message = dict
sys.modules['react_agent.utils.cache'] = mock.MagicMock()
sys.modules['react_agent.utils.cache'].ProcessorCache = mock.MagicMock()
sys.modules['react_agent.utils.extraction'] = mock.MagicMock()
sys.modules['react_agent.utils.content'] = mock.MagicMock()
sys.modules['react_agent.utils.content'].merge_chunk_results = mock.MagicMock()
sys.modules['react_agent.utils.statistics'] = mock.MagicMock()
sys.modules['react_agent.utils.logging'] = mock.MagicMock()
sys.modules['react_agent.utils.logging'].get_logger = lambda name: mock.MagicMock()
sys.modules['react_agent.configuration'] = mock.MagicMock()
sys.modules['react_agent.configuration'].Configuration = mock.MagicMock()
sys.modules['react_agent.prompts'] = mock.MagicMock()
sys.modules['react_agent.prompts.templates'] = mock.MagicMock()

# Also mock the src.react_agent namespace
sys.modules['src'] = mock.MagicMock()
sys.modules['src.react_agent'] = mock_react_agent
sys.modules['src.react_agent.configuration'] = sys.modules['react_agent.configuration']
sys.modules['src.react_agent.utils'] = sys.modules['react_agent.utils']
sys.modules['src.react_agent.utils.llm'] = sys.modules['react_agent.utils.llm']
sys.modules['src.react_agent.utils.cache'] = sys.modules['react_agent.utils.cache']
sys.modules['src.react_agent.utils.extraction'] = sys.modules['react_agent.utils.extraction']
sys.modules['src.react_agent.utils.content'] = sys.modules['react_agent.utils.content']
sys.modules['src.react_agent.utils.statistics'] = sys.modules['react_agent.utils.statistics']
sys.modules['src.react_agent.utils.logging'] = sys.modules['react_agent.utils.logging']

# Create a mock for the langgraph.prebuilt module
class MockToolNode:
    """Mock implementation of ToolNode for testing."""
    def __init__(self, tools):
        self.tools = tools

# Create a mock module for langgraph.prebuilt
mock_prebuilt_module = mock.MagicMock()
mock_prebuilt_module.ToolNode = MockToolNode
sys.modules['langgraph.prebuilt'] = mock_prebuilt_module

# Create a proper mock for nltk
class MockNLTKData:
    """Mock implementation of nltk.data module."""
    def __init__(self):
        self.__name__ = "nltk.data"
        self.__spec__ = SimpleNamespace(name="nltk.data", origin="nltk.data", loader=None, parent="nltk")
    
    def find(self, *args, **kwargs):
        return "/mock/path/to/tokenizers/punkt"
    
    def load(self, *args, **kwargs):
        # Return a mock tokenizer that can be used
        return MockTokenizer()
    
    def has_location(self, *args, **kwargs):
        return True

class MockTokenizer:
    """Mock implementation of nltk tokenizer."""
    def tokenize(self, text, *args, **kwargs):
        # Simple tokenization by splitting on spaces and periods
        return [t for t in text.replace('.', ' . ').split() if t]

class MockNLTK(ModuleType):
    """Mock implementation of nltk module."""
    def __init__(self):
        super().__init__("nltk")
        self.data = MockNLTKData()
        self.__spec__ = SimpleNamespace(name="nltk", origin="nltk", loader=None, parent=None)
        
# Create nltk mock module
mock_nltk = MockNLTK()
sys.modules['nltk'] = mock_nltk
sys.modules['nltk.data'] = mock_nltk.data

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
