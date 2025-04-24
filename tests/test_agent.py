"""
Tests for Agent class.
"""

import pytest
from unittest.mock import MagicMock, patch

from lg_adk.agents.base import Agent
from lg_adk.config.settings import Settings


@pytest.fixture
def mock_settings():
    """Creates mock settings."""
    return Settings(
        default_llm="ollama/llama3"
    )


@pytest.fixture
def mock_model():
    """Creates a mock model."""
    model = MagicMock()
    model.invoke.return_value = "This is a test response"
    return model


@patch("lg_adk.agents.base.get_model")
def test_agent_creation(mock_get_model, mock_model):
    """Test that an agent can be created."""
    mock_get_model.return_value = mock_model
    
    agent = Agent(
        name="test_agent",
        llm="ollama/llama3",
        description="Test agent"
    )
    
    assert agent.name == "test_agent"
    assert agent.llm == "ollama/llama3"
    assert agent.description == "Test agent"
    assert agent.tools == []


@patch("lg_adk.agents.base.get_model")
def test_agent_run(mock_get_model, mock_model):
    """Test that an agent can run."""
    mock_get_model.return_value = mock_model
    
    agent = Agent(
        name="test_agent",
        llm="ollama/llama3",
        description="Test agent"
    )
    
    state = {"input": "Hello"}
    result = agent.run(state)
    
    assert result["output"] == "This is a test response"
    assert result["agent"] == "test_agent"
    mock_model.invoke.assert_called_once()


@patch("lg_adk.agents.base.get_model")
def test_agent_tools(mock_get_model, mock_model):
    """Test that tools can be added to an agent."""
    mock_get_model.return_value = mock_model
    
    agent = Agent(
        name="test_agent",
        llm="ollama/llama3",
        description="Test agent"
    )
    
    # Create a mock tool
    tool = MagicMock()
    tool.name = "test_tool"
    tool.description = "A test tool"
    
    agent.add_tool(tool)
    
    assert len(agent.tools) == 1
    assert agent.tools[0].name == "test_tool"
    
    # Test adding multiple tools
    tool2 = MagicMock()
    tool2.name = "test_tool2"
    
    agent.add_tools([tool2])
    
    assert len(agent.tools) == 2
    assert agent.tools[1].name == "test_tool2" 