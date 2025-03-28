"""React Agent.

This module defines a custom reasoning and action agent graph.
It invokes tools in a simple loop.
"""

# Import ensure_nltk_data early to ensure NLTK data is properly configured
from react_agent.graphs.graph import graph
from react_agent.utils import ensure_nltk_data

# Call the function to ensure NLTK data is available
ensure_nltk_data()

__all__ = ["graph"]
