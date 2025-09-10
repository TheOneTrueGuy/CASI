import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from CASI import safe_parse_json

def test_safe_parse_json_valid():
    """Test parsing valid JSON with response and suggestions."""
    raw = '{"response": "This is a test response", "suggestions": ["sug1", "sug2"]}'
    result, is_valid = safe_parse_json(raw)
    assert is_valid == True
    assert result['response'] == "This is a test response"
    assert result['suggestions'] == ["sug1", "sug2"]

def test_safe_parse_json_invalid():
    """Test parsing invalid JSON."""
    raw = "Not a JSON string"
    result, is_valid = safe_parse_json(raw)
    assert is_valid == False
    assert result['response'] == raw
    assert result['suggestions'] == []

def test_safe_parse_json_empty():
    """Test parsing empty string."""
    raw = ""
    result, is_valid = safe_parse_json(raw)
    assert is_valid == False
    assert result['response'] == "No content received from the model."
    assert result['suggestions'] == []

def test_safe_parse_json_missing_keys():
    """Test parsing JSON missing required keys."""
    raw = '{"other_key": "value"}'
    result, is_valid = safe_parse_json(raw)
    assert is_valid == True
    assert result['response'] == raw
    assert result['suggestions'] == []
    assert result['other_key'] == "value"
