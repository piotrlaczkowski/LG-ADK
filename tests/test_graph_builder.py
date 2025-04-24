"""
Tests for GraphBuilder class.
"""

import pytest
from unittest.mock import MagicMock, patch

from langgraph.graph import Graph
from lg_adk.agents.base import Agent
from lg_adk.builders.graph_builder import GraphBuilder
from lg_adk.memory.memory_manager import MemoryManager


@pytest.fixture
def mock_agent():
    """Creates a mock agent."""
    agent = MagicMock(spec=Agent)
    agent.name = "test_agent"
    agent.run.return_value = {"output": "Test response", "agent": "test_agent"}
    agent.__call__.return_value = {"output": "Test response", "agent": "test_agent"}
    return agent


@pytest.fixture
def mock_memory():
    """Creates a mock memory manager."""
    memory = MagicMock(spec=MemoryManager)
    return memory


def test_graph_builder_creation():
    """Test that a graph builder can be created."""
    builder = GraphBuilder()
    assert builder.agents == []
    assert builder.memory_manager is None
    assert builder.human_in_loop is False


def test_add_agent(mock_agent):
    """Test adding an agent to the graph builder."""
    builder = GraphBuilder()
    builder.add_agent(mock_agent)
    
    assert len(builder.agents) == 1
    assert builder.agents[0] == mock_agent


def test_add_memory(mock_memory):
    """Test adding memory to the graph builder."""
    builder = GraphBuilder()
    builder.add_memory(mock_memory)
    
    assert builder.memory_manager == mock_memory


def test_enable_human_in_loop():
    """Test enabling human-in-the-loop."""
    builder = GraphBuilder()
    builder.enable_human_in_loop()
    
    assert builder.human_in_loop is True


@patch("lg_adk.builders.graph_builder.StateGraph")
def test_build_single_agent(MockStateGraph, mock_agent):
    """Test building a graph with a single agent."""
    # Mock the StateGraph and its methods
    mock_graph = MagicMock()
    MockStateGraph.return_value = mock_graph
    mock_graph.compile.return_value = MagicMock(spec=Graph)
    
    # Create the builder with a single agent
    builder = GraphBuilder()
    builder.add_agent(mock_agent)
    
    # Build the graph
    graph = builder.build()
    
    # Assert things were called correctly
    MockStateGraph.assert_called_once()
    mock_graph.add_node.assert_called_once_with(mock_agent.name, mock_agent)
    mock_graph.add_edge.assert_any_call(None, mock_agent.name)
    mock_graph.compile.assert_called_once()
    assert isinstance(graph, MagicMock)  # Should return the compiled graph 