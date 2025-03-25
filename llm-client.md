Designing an Asynchronous LLM Utility for LangGraph Nodes
Building a research assistant with LangGraph requires a clean, composable interface for making LLM calls within your flow nodes. This guide outlines a design for an async LLM utility module that wraps your existing robust functions (like call_model and call_model_json) and is agnostic to the underlying provider. We’ll cover design goals, a sample implementation, and how to use it in @tool/@step nodes.
Goals for the LLM Utility Interface
Asynchronous & Composable – The utility functions should be async so they can be awaited inside LangGraph steps/tools and composed (e.g. multiple LLM calls in parallel via asyncio.gather). This non-blocking design improves throughput and allows pipelining of LLM calls​
GITHUB.COM
.
Provider-Agnostic – The interface should not hard-code any provider (OpenAI, Anthropic, etc.). Model selection is handled via configuration and the LiteLLM proxy, so our utility simply delegates calls to the unified API. For example, the same function can call either OpenAI or Anthropic by just changing the model name​
GITHUB.COM
.
Reuse Existing Wrappers – Leverage your existing call_model and call_model_json functions for core functionality (sending requests, handling retries, chunking large inputs, etc.) rather than duplicating that logic. The utility will be a thin layer over these robust wrappers.
Clean Message Formatting – Provide a simple way to include system prompts or multiple messages. The utility should assemble the message list (system + user prompt, or full conversation) in the format required by the LLM API, abstracting away the boilerplate.
Handle Chunking & Retries Transparently – If an input is too long, the underlying call_model can chunk it and summarize or process in parts. The utility interface should make this invisible to the caller – just pass the long text and get a result. Similarly, retries on failure can be handled internally by the wrapper (e.g. via LiteLLM’s built-in retry logic​
DOCS.LITELLM.AI
).
Standard Call Patterns – Expose a few high-level async functions for common LLM tasks:
llm_chat(...) for free-form chat/completion (returning text).
llm_json(...) for structured outputs (asking the model for JSON and parsing it into Python dicts/lists).
llm_embed(...) for obtaining vector embeddings from text.
These goals ensure our LangGraph nodes remain simple and focused on the task, while the utility handles all LLM intricacies.
Implementation: Utility Functions for LLM Calls
We can implement the utility as a set of reusable async functions or encapsulate them in a class for organization. In this design, we'll use a class LLMClient to group related methods. This class doesn’t maintain state between calls (it’s essentially a namespace), so you could also implement these as standalone async def functions. The class approach makes it easy to inject configuration if needed (e.g., default model or parameters at init). Key points of implementation:
Delegation to Wrappers: Each method will call your existing call_model/call_model_json (and a new call_embedding wrapper for embeddings) under the hood. This means all provider-specific handling, API calls, chunking, and retries happen in those functions. For example, LiteLLM’s unified completion API will handle different providers uniformly​
GITHUB.COM
 and ensure consistent response format​
DOCS.LITELLM.AI
. Our utility just prepares the inputs and post-processes outputs if needed.
Message Preparation: The chat and json methods take a user prompt (and optional system prompt or full message list) and construct a messages list as expected by the LLM. Typically, if a system prompt is provided, it’s added as the first message with role "system", followed by the user message(s) with role "user". If the caller already has a list of message dicts (for multi-turn conversations), the utility can accept that as well (for simplicity, we show just prompt + system usage here).
JSON Parsing: The json method will ensure the model’s answer is returned as a Python object. Likely, your call_model_json wrapper already handles prompting the model to output valid JSON and parsing the result (possibly with error correction or re-try if the JSON is invalid). We simply return whatever call_model_json gives us. (In more advanced scenarios, you could enforce structure with a Pydantic schema – e.g. LangChain’s .with_structured_output(...) uses a BaseModel to validate the LLM output​
REALPYTHON.COM
 – but we assume call_model_json covers the basics of JSON formatting.)
Embedding Calls: The embed method will call an embedding endpoint via a wrapper (for example, using LiteLLM’s embedding support). This might use a different underlying function, or call_model could detect embedding mode based on model ID. We keep it separate for clarity. The result is typically a list of floats (the embedding vector).
Agnostic Config: The actual model names, API keys, etc., are pulled from your config or environment. For instance, call_model might use a default model set in config, or you can pass a model= argument to override. The utility functions can accept **kwargs to forward any optional parameters like model, temperature, etc., to the wrappers. This makes them flexible without being tied to specific providers.
Below is a sample implementation of the LLMClient class with these methods:
python
Copy
from my_llm_util_module import call_model, call_model_json, call_embedding  # existing wrappers

class LLMClient:
    """Asynchronous LLM utility for chat, JSON output, and embeddings."""
    def __init__(self, default_model: str = None):
        # Optionally set a default model name or other config if needed
        self.default_model = default_model

    async def llm_chat(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """Get a chat completion as plain text."""
        # Build message list for the LLM (include system prompt if provided)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        # Merge default model if set
        if self.default_model and "model" not in kwargs:
            kwargs["model"] = self.default_model
        # Call the underlying model (async)
        response = await call_model(messages=messages, **kwargs)
        # Assume call_model returns the assistant's message content (string)
        return response

    async def llm_json(self, prompt: str, system_prompt: str = None, **kwargs) -> dict:
        """Get a structured JSON response parsed into a Python dict."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        if self.default_model and "model" not in kwargs:
            kwargs["model"] = self.default_model
        # call_model_json is expected to ensure the model outputs JSON and parse it
        result = await call_model_json(messages=messages, **kwargs)
        return result  # e.g. a Python dict/list representing the JSON

    async def llm_embed(self, text: str, **kwargs) -> list[float]:
        """Get embeddings for the given text."""
        if self.default_model and "model" not in kwargs:
            kwargs["model"] = self.default_model
        embedding = await call_embedding(text=text, **kwargs)
        return embedding  # e.g. a list of floats
Some notes on the above implementation:
Each method is async and uses await to call the respective wrapper. This ensures integration with LangGraph’s async execution (LangGraph nodes can themselves be async functions, which is compatible).
We pass through any **kwargs to the underlying functions. This allows users to tweak parameters like model="anthropic/claude-2" or temperature=0.7 for a specific call without changing the utility interface. If no model is specified, you could have self.default_model set (perhaps from a config file or environment) so that all calls use a sensible default.
The design assumes call_model and call_model_json handle the heavy lifting (including splitting messages into chunks if too long, combining partial results, retrying on errors, etc.). For example, if the prompt text is extremely large, call_model might automatically break it into chunks and summarize or process sequentially. The LLMClient doesn’t need special logic for that; it simply provides the interface.
The llm_chat returns a raw text string (assistant’s reply), llm_json returns a Python object (already parsed from JSON), and llm_embed returns a list of floats. This makes it clear at the call site what to expect. If the underlying wrapper returns a full response dict (e.g. OpenAI-style), you could extract the content inside llm_chat (e.g. response['choices'][0]['message']['content']) before returning. However, if your wrapper already returns just the content text (as many utilities do), simply returning response is fine.
Because this is provider-agnostic, switching the backend model or provider doesn’t require changing the utility usage. For instance, using OpenAI vs Anthropic is just a config change (LiteLLM’s unified API makes this possible​
GITHUB.COM
). The utility is oblivious to those details.
You can easily extend this class with more specialized methods if needed (e.g., llm_chat_stream for streaming completions, or domain-specific prompts). But the three provided cover most use cases in a research assistant.