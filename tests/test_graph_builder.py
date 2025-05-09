"""Tests for GraphBuilder class."""

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lg_adk.agents.base import Agent
from lg_adk.builders.graph_builder import GraphBuilder
from lg_adk.memory.memory_manager import MemoryManager


@pytest.fixture
def mock_agent() -> Any:
    """Creates a mock agent."""
    agent = MagicMock(spec=Agent)
    agent.name = "test_agent"
    agent.run.return_value = {"output": "Test response", "agent": "test_agent", "memory": {}}
    agent.__call__ = MagicMock(return_value={"output": "Test response", "agent": "test_agent", "memory": {}})
    return agent


@pytest.fixture
def mock_memory() -> Any:
    """Creates a mock memory manager."""
    memory = MagicMock(spec=MemoryManager)
    return memory


class TestGraphBuilder(unittest.TestCase):
    """Tests for the GraphBuilder class."""

    def setUp(self) -> None:
        """Set up mock agent and memory instances for use in tests."""
        self.mock_agent = MagicMock(spec=Agent)
        self.mock_agent.name = "test_agent"
        self.mock_agent.run.return_value = {"output": "Test response", "agent": "test_agent", "memory": {}}
        self.mock_agent.__call__ = MagicMock(
            return_value={"output": "Test response", "agent": "test_agent", "memory": {}}
        )
        self.mock_memory = MagicMock(spec=MemoryManager)

    def test_graph_builder_creation(self) -> None:
        """Test that a graph builder can be created."""
        builder = GraphBuilder()
        self.assertEqual(builder.agents, [])
        self.assertIsNone(builder.memory_manager)
        self.assertFalse(builder.human_in_loop)

    def test_add_agent(self) -> None:
        """Test adding an agent to the graph builder."""
        builder = GraphBuilder()
        builder.add_agent(self.mock_agent)
        self.assertEqual(len(builder.agents), 1)
        self.assertEqual(builder.agents[0], self.mock_agent)

    @patch("lg_adk.builders.graph_builder.StateGraph")
    def test_build_single_agent(self, mock_state_graph) -> None:
        """Test building a graph with a single agent."""
        mock_graph = MagicMock()
        mock_state_graph.return_value = mock_graph
        mock_graph.compile.return_value = MagicMock()
        builder = GraphBuilder()
        builder.add_agent(self.mock_agent)
        graph = builder.build()
        mock_state_graph.assert_called_once()
        mock_graph.add_node.assert_called_once_with(self.mock_agent.name, self.mock_agent)
        mock_graph.add_edge.assert_any_call("__start__", self.mock_agent.name)
        mock_graph.compile.assert_called_once()
        self.assertIsInstance(graph, MagicMock)

    def test_add_memory(self) -> None:
        """Test adding memory to the graph builder."""
        builder = GraphBuilder()
        builder.add_memory(self.mock_memory)
        self.assertEqual(builder.memory_manager, self.mock_memory)

    def test_enable_human_in_loop(self) -> None:
        """Test enabling human-in-the-loop."""
        builder = GraphBuilder()
        builder.enable_human_in_loop()
        self.assertTrue(builder.human_in_loop)
