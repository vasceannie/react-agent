import json
import pytest
import sys
from react_agent.utils.extraction import safe_json_parse, find_json_object


def test_find_json_object_basic():
    """Test basic JSON object extraction."""
    text = "Some text before {\"key\": \"value\"} and after"
    result = find_json_object(text)
    assert result == "{\"key\": \"value\"}"
    
    # Test with JSON array
    text = "Some text before [1, 2, 3] and after"
    result = find_json_object(text)
    assert result == "[1, 2, 3]"


def test_find_json_object_nested():
    """Test nested JSON object extraction."""
    text = "Text {\"outer\": {\"inner\": \"value\"}} more"
    result = find_json_object(text)
    assert result == "{\"outer\": {\"inner\": \"value\"}}"
    
    # Test with nested arrays
    text = "Text {\"items\": [1, [2, 3], 4]} more"
    result = find_json_object(text)
    assert result == "{\"items\": [1, [2, 3], 4]}"


def test_find_json_object_multiple():
    """Test extraction with multiple JSON objects."""
    text = "{\"first\": 1} then {\"second\": 2}"
    result = find_json_object(text)
    # Should find the first complete object
    assert result == "{\"first\": 1}"


def test_find_json_object_unbalanced():
    """Test with unbalanced braces."""
    text = "Unbalanced {\"key\": \"value\"} and {\"broken\": \"value"
    result = find_json_object(text)
    assert result == "{\"key\": \"value\"}"


def test_safe_json_parse_direct():
    """Test direct JSON parsing."""
    json_str = json.dumps({"key": "value"})
    result = safe_json_parse(json_str, "test_category")
    assert result == {"key": "value"}


def test_safe_json_parse_embedded():
    """Test parsing JSON embedded in text."""
    text = "Some text before {\"key\": \"value\"} and after"
    result = safe_json_parse(text, "test_category")
    assert result == {"key": "value"}


def test_safe_json_parse_array():
    """Test parsing JSON array."""
    text = "Array: [1, 2, 3]"
    result = safe_json_parse(text, "test_category")
    assert result == [1, 2, 3]


def test_safe_json_parse_single_quotes():
    """Test parsing with single quotes."""
    text = "{'key': 'value'}"
    result = safe_json_parse(text, "test_category")
    assert result == {"key": "value"}


def test_safe_json_parse_trailing_comma():
    """Test parsing with trailing comma."""
    text = "{\"items\": [1, 2, 3,]}"
    result = safe_json_parse(text, "test_category")
    assert result == {"items": [1, 2, 3]}


def test_safe_json_parse_markdown():
    """Test parsing JSON in markdown code block."""
    text = "```json\n{\"key\": \"value\"}\n```"
    result = safe_json_parse(text, "test_category")
    assert result == {"key": "value"}



def test_safe_json_parse_unquoted_keys():
    """Test parsing with unquoted keys."""
    text = "{key: \"value\"}"
    result = safe_json_parse(text, "test_category")
    assert result == {"key": "value"}


def test_safe_json_parse_quoted_json_object():
    """Test parsing JSON object with quotes around the entire object."""
    text = "'{\"key\": \"value\"}'"
    result = safe_json_parse(text, "test_category")
    assert result == {"key": "value"}
    
    # Test with single quotes around the entire object
    text = "'{ \"key\": \"value\" }'"
    result = safe_json_parse(text, "test_category")
    assert result == {"key": "value"}
    
    # Test with double quotes around the entire object
    text = "\"{ \\\"key\\\": \\\"value\\\" }\""
    
    # Print the exact string for debugging
    print(f"\nExact string: {repr(text)}", file=sys.stderr)
    
    result = safe_json_parse(text, "test_category")
    assert result == {"key": "value"}, f"Expected {{'key': 'value'}} but got {repr(result)}"