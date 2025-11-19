import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path to import webCASI
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import webCASI

def test_safe_parse_json_simple():
    """Test parsing simple valid JSON."""
    raw = '{"response": "Test response", "suggestions": ["sug1"]}'
    result, is_valid = webCASI.safe_parse_json(raw)
    assert is_valid == True
    assert result['response'] == "Test response"
    assert result['suggestions'] == ["sug1"]

def test_safe_parse_json_markdown():
    """Test parsing JSON wrapped in markdown code blocks."""
    raw = 'Here is the json:\n```json\n{"response": "Markdown response", "suggestions": []}\n```'
    result, is_valid = webCASI.safe_parse_json(raw)
    assert is_valid == True
    assert result['response'] == "Markdown response"
    assert result['suggestions'] == []

def test_safe_parse_json_fallback():
    """Test fallback when JSON is invalid."""
    raw = "Just some text, not JSON."
    result, is_valid = webCASI.safe_parse_json(raw)
    assert is_valid == False
    assert result['response'] == raw
    assert result['suggestions'] == []

@patch('webCASI.generate_response')
def test_generator_function(mock_generate):
    """Test generator function with mocked backend."""
    # Mock the response from generate_response
    mock_generate.return_value = '{"response": "Generated content", "suggestions": ["Fix this"]}'
    
    response, suggestions = webCASI.generator(
        backend="openai",
        model="gpt-4",
        prompt="System prompt",
        user_input="User input",
        critic_feedback="None"
    )
    
    assert response == "Generated content"
    assert suggestions == ["Fix this"]
    mock_generate.assert_called_once()

@patch('webCASI.generate_response')
def test_critic_function(mock_generate):
    """Test critic function with mocked backend."""
    # Mock the response from generate_response
    # Critic prompt expects plain text response, but function splits by newlines for suggestions
    mock_generate.return_value = "Critique text.\n- Suggestion 1\n* Suggestion 2"
    
    response, suggestions = webCASI.critic(
        backend="openai",
        model="gpt-4",
        prompt="System prompt",
        generator_output="Gen output"
    )
    
    assert response == "Critique text.\n- Suggestion 1\n* Suggestion 2"
    assert "Suggestion 1" in suggestions
    assert "Suggestion 2" in suggestions

@patch('webCASI.ddgs')
def test_search_web_success(mock_ddgs):
    """Test web search wrapper when DDGS is available."""
    if mock_ddgs is None:
        pytest.skip("duckduckgo_search not installed")
        
    # Mock ddgs.text() to return an iterable
    mock_ddgs.text.return_value = [
        {"title": "Result 1", "href": "http://example.com", "body": "Content 1"}
    ]
    
    results = webCASI.search_web("test query")
    assert len(results) == 1
    assert results[0]['title'] == "Result 1"

@patch('webCASI.generate_response')
@patch('webCASI.search_web')
def test_agentic_step_no_search(mock_search, mock_generate):
    """Test agentic step when model decides no search is needed."""
    # Plan step returns NO_SEARCH_NEEDED
    mock_generate.return_value = "NO_SEARCH_NEEDED"
    
    final_prompt = webCASI.agentic_step(
        backend="openai",
        model="gpt-4",
        role="Generator",
        prompt="Task prompt",
        context="Context"
    )
    
    # Should return original prompt + newlines
    assert "Task prompt" in final_prompt
    assert "Thinking Process" not in final_prompt
    mock_search.assert_not_called()

@patch('webCASI.generate_response')
@patch('webCASI.search_web')
def test_agentic_step_with_search(mock_search, mock_generate):
    """Test agentic step when model decides search IS needed."""
    # Plan step returns a query
    mock_generate.return_value = "python testing best practices"
    
    # Search returns results
    mock_search.return_value = [
        {"title": "Pytest Guide", "href": "http://test.com", "body": "Use pytest fixtures"}
    ]
    
    final_prompt = webCASI.agentic_step(
        backend="openai",
        model="gpt-4",
        role="Generator",
        prompt="Task prompt",
        context="Context"
    )
    
    # Should include search results in the final prompt
    assert "Task prompt" in final_prompt
    assert "Thinking Process" in final_prompt
    assert "Pytest Guide" in final_prompt
    mock_search.assert_called()


