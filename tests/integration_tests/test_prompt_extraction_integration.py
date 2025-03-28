import json
import pytest
from unittest.mock import MagicMock, patch

from react_agent.prompts.research import get_extraction_prompt
from react_agent.prompts.templates import STRUCTURED_OUTPUT_VALIDATION
from react_agent.utils.extraction import (
    extract_category_information,
    safe_json_parse,
    find_json_object,
)
from react_agent.utils.content import (
    chunk_text,
    preprocess_content,
    merge_chunk_results,
)


class TestPromptExtractionIntegration:
    """Integration tests that validate how prompts get ingested with structured JSON responses."""

    @pytest.mark.asyncio
    async def test_extraction_prompt_to_json_parsing(self, mocker):
        """Test that extraction prompts generate properly structured JSON responses that can be parsed."""
        # 1. Mock the extraction model to return a structured JSON response
        mock_model = MagicMock()
        mock_model.invoke.return_value = json.dumps({
            "extracted_facts": [
                {
                    "fact": "Cloud adoption increased by 25% in 2023",
                    "source_text": "According to the report, cloud adoption saw a significant increase of 25% in 2023.",
                    "confidence": "high",
                    "data_type": "growth_rate"
                }
            ],
            "relevance_score": 0.85
        })

        # 2. Set up test data
        category = "market_dynamics"
        query = "cloud computing adoption trends"
        url = "https://example.com/cloud-report"
        content = "According to the report, cloud adoption saw a significant increase of 25% in 2023."
        title = "Cloud Computing Trends 2023"

        # 3. Get the extraction prompt for the category
        prompt = get_extraction_prompt(category, query, url, content)
        
        # 4. Assert that the prompt contains the necessary validation requirements
        assert "FORMAT YOUR RESPONSE AS JSON" in prompt
        assert "extracted_facts" in prompt
        
        # 5. Mock the _process_content function to return our mock response
        with patch("react_agent.utils.extraction._process_content") as mock_process:
            mock_process.return_value = {
                "extracted_facts": [
                    {
                        "fact": "Cloud adoption increased by 25% in 2023",
                        "source_text": "According to the report, cloud adoption saw a significant increase of 25% in 2023.",
                        "confidence": "high",
                        "data_type": "growth_rate"
                    }
                ],
                "relevance_score": 0.85
            }
            
            # 6. Call extract_category_information
            facts, relevance = await extract_category_information(
                content=content,
                url=url,
                title=title,
                category=category,
                original_query=query,
                prompt_template=prompt,
                extraction_model=mock_model
            )
            
            # 7. Assert that the extraction returned properly structured data
            assert isinstance(facts, list)
            assert len(facts) > 0
            assert facts[0]["type"] == "fact"
            assert "data" in facts[0]
            assert "text" in facts[0]["data"]
            assert relevance > 0

    def test_json_extraction_from_llm_response(self):
        """Test that JSON objects can be properly extracted from LLM responses."""
        # 1. Test with a clean JSON response
        clean_json = '{"key": "value", "nested": {"data": 123}}'
        extracted = find_json_object(clean_json)
        assert extracted == clean_json
        
        # 2. Test with JSON embedded in text
        text_with_json = 'Some text before {"key": "value", "nested": {"data": 123}} and text after'
        extracted = find_json_object(text_with_json)
        assert extracted == '{"key": "value", "nested": {"data": 123}}'
        
        # 3. Test with markdown code blocks
        markdown_json = '```json\n{"key": "value", "nested": {"data": 123}}\n```'
        extracted = find_json_object(markdown_json)
        assert extracted == '{"key": "value", "nested": {"data": 123}}'
        
        # 4. Test with single quotes instead of double quotes
        single_quotes = "{'key': 'value', 'nested': {'data': 123}}"
        extracted = find_json_object(single_quotes)
        assert extracted is not None
        parsed = json.loads(extracted.replace("'", '"'))
        assert parsed["key"] == "value"
        
        # 5. Test with trailing commas (common LLM mistake)
        trailing_comma = '{"key": "value", "nested": {"data": 123,},}'
        extracted = find_json_object(trailing_comma)
        assert extracted is not None
        
        # 6. Test safe_json_parse with category
        category = "market_dynamics"
        parsed = safe_json_parse(trailing_comma, category)
        assert isinstance(parsed, dict)
        assert "key" in parsed

    def test_content_chunking_and_merging(self):
        """Test that content can be chunked and results merged properly."""
        # 1. Create a long text that will be chunked
        long_text = "This is a test. " * 100
        
        # 2. Chunk the text
        chunks = chunk_text(long_text, chunk_size=200, overlap=50)
        
        # 3. Assert that chunking worked properly
        assert len(chunks) > 1
        assert len(chunks[0]) <= 200
        
        # 4. Create mock extraction results for each chunk
        results = []
        for i, chunk in enumerate(chunks):
            results.append({
                "extracted_facts": [
                    {
                        "fact": f"Fact {i}",
                        "confidence": 0.8,
                        "source_text": chunk[:50]
                    }
                ],
                "relevance_score": 0.7 + (i * 0.05)
            })
        
        # 5. Merge the results
        category = "market_dynamics"
        merged = merge_chunk_results(results, category)
        
        # 6. Assert that merging worked properly
        assert "extracted_facts" in merged
        assert len(merged["extracted_facts"]) == len(results)
        assert "relevance_score" in merged
        assert isinstance(merged["relevance_score"], float)

    @pytest.mark.asyncio
    async def test_full_extraction_pipeline(self, mocker):
        """Test the full extraction pipeline from prompt to structured data."""
        # 1. Mock the extraction model
        mock_model = MagicMock()
        mock_model.invoke.return_value = json.dumps({
            "extracted_facts": [
                {
                    "fact": "Market for cloud services reached $500B in 2023",
                    "source_text": "The global market for cloud services reached $500 billion in 2023.",
                    "confidence": "high",
                    "data_type": "market_size"
                },
                {
                    "fact": "AWS holds 32% market share",
                    "source_text": "Amazon Web Services (AWS) maintains a 32% market share.",
                    "confidence": "medium",
                    "data_type": "competitive"
                }
            ],
            "relevance_score": 0.9
        })
        
        # 2. Set up test data
        category = "market_dynamics"
        query = "cloud computing market size"
        url = "https://example.com/market-report"
        content = """
        The global market for cloud services reached $500 billion in 2023.
        This represents a 20% increase from the previous year.
        Amazon Web Services (AWS) maintains a 32% market share.
        Microsoft Azure follows with 22% market share.
        Google Cloud Platform has 10% of the market.
        """
        title = "Cloud Market Report 2023"
        
        # 3. Preprocess the content
        processed_content = preprocess_content(content, url)
        
        # 4. Get the extraction prompt
        prompt = get_extraction_prompt(category, query, url, processed_content)
        
        # 5. Mock the necessary functions to isolate the test
        with patch("react_agent.utils.extraction._validate_inputs", return_value=True), \
             patch("react_agent.utils.extraction._process_content") as mock_process:
            
            # Set up the mock to return our structured data
            mock_process.return_value = {
                "extracted_facts": [
                    {
                        "fact": "Market for cloud services reached $500B in 2023",
                        "source_text": "The global market for cloud services reached $500 billion in 2023.",
                        "confidence": "high",
                        "data_type": "market_size"
                    },
                    {
                        "fact": "AWS holds 32% market share",
                        "source_text": "Amazon Web Services (AWS) maintains a 32% market share.",
                        "confidence": "medium",
                        "data_type": "competitive"
                    }
                ],
                "relevance_score": 0.9
            }
            
            # 6. Call extract_category_information
            facts, relevance = await extract_category_information(
                content=processed_content,
                url=url,
                title=title,
                category=category,
                original_query=query,
                prompt_template=prompt,
                extraction_model=mock_model
            )
            
            # 7. Validate the results
            assert isinstance(facts, list)
            assert len(facts) == 2
            assert all(fact["type"] == "fact" for fact in facts)
            assert all("data" in fact for fact in facts)
            assert all("text" in fact["data"] for fact in facts)
            assert all("source_url" in fact["data"] for fact in facts)
            assert all("confidence_score" in fact["data"] for fact in facts)
            assert relevance > 0.8

    def test_prompt_validation_requirements(self):
        """Test that prompts include proper validation requirements for structured output."""
        # 1. Check that STRUCTURED_OUTPUT_VALIDATION contains necessary validation rules
        assert "valid JSON" in STRUCTURED_OUTPUT_VALIDATION
        assert "exact schema" in STRUCTURED_OUTPUT_VALIDATION
        assert "proper JSON syntax" in STRUCTURED_OUTPUT_VALIDATION
        
        # 2. Get an extraction prompt and check it includes validation requirements
        category = "market_dynamics"
        query = "test query"
        url = "https://example.com"
        content = "test content"
        
        prompt = get_extraction_prompt(category, query, url, content)
        
        # 3. Check that the prompt includes validation requirements
        assert "FORMAT YOUR RESPONSE AS JSON" in prompt
        assert "extracted_facts" in prompt
        assert "response must be" in prompt.lower()
