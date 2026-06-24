import os
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def mock_anthropic_api_key(monkeypatch):
    """Ensure ANTHROPIC_API_KEY is set for tests."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")

def test_get_llm_returns_chat_anthropic():
    from agent_graph.llm import get_llm
    from langchain_anthropic import ChatAnthropic
    llm = get_llm()
    assert isinstance(llm, ChatAnthropic)

def test_get_llm_uses_correct_model():
    from agent_graph.llm import get_llm
    llm = get_llm()
    assert llm.model == "claude-sonnet-4-6"

def test_get_llm_max_tokens():
    from agent_graph.llm import get_llm
    llm = get_llm()
    assert llm.max_tokens == 1000

def test_get_llm_temperature_respected():
    from agent_graph.llm import get_llm
    llm = get_llm(temperature=0.7)
    assert llm.temperature == 0.7

def test_get_llm_no_hardcoded_key():
    """API key must come from env, not be embedded in the factory."""
    import inspect
    import agent_graph.llm as llm_module
    src = inspect.getsource(llm_module)
    assert "sk-ant" not in src
