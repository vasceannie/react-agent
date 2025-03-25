"""Define the configurable parameters for the agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig, ensure_config

from react_agent.prompts import templates


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    system_prompt: str = field(
        default=templates.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    firecrawl_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("FIRECRAWL_API_KEY"),
        metadata={
            "description": "API key for the FireCrawl service. Required for web scraping and crawling."
        },
    )

    firecrawl_url: Optional[str] = field(
        default_factory=lambda: os.getenv("FIRECRAWL_URL"),
        metadata={
            "description": "Base URL for the FireCrawl service. Use this for self-hosted instances."
        },
    )

    jina_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("JINA_API_KEY"),
        metadata={
            "description": "API key for the Jina AI service. Required for web search and summarization."
        },
    )

    jina_url: Optional[str] = field(
        default_factory=lambda: os.getenv("JINA_URL", "https://s.jina.ai"),
        metadata={
            "description": "Base URL for the Jina AI service. Use this for self-hosted instances."
        },
    )

    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"),
        metadata={
            "description": "API key for the Anthropic service. Required for chat completions."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        
        # Get environment variables for FireCrawl
        env_config = {
            "firecrawl_api_key": os.getenv("FIRECRAWL_API_KEY"),
            "firecrawl_url": os.getenv("FIRECRAWL_URL"),
        }
        
        # Merge environment variables with configurable, giving priority to configurable
        merged_config = {**env_config, **configurable}
        
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in merged_config.items() if k in _fields})
