"""Tests for Agent class."""

import unittest
from unittest.mock import MagicMock, patch

from lg_adk.agents.base import Agent
from lg_adk.config.settings import Settings


class TestAgent(unittest.TestCase):
    """Tests for the Agent class."""

    def setUp(self) -> None:
        """Set up mock settings and model for each test."""
        self.mock_settings = Settings(default_llm="ollama/llama3")
        self.mock_model = MagicMock()
        self.mock_model.invoke.return_value = "This is a test response"

    @patch("lg_adk.agents.base.get_model")
    def test_agent_creation(self, mock_get_model) -> None:
        """Test that an agent can be created."""
        mock_get_model.return_value = self.mock_model
        agent = Agent(
            name="test_agent",
            llm="ollama/llama3",
            description="Test agent",
        )
        self.assertEqual(agent.name, "test_agent")
        self.assertEqual(agent.llm, "ollama/llama3")
        self.assertEqual(agent.description, "Test agent")
        self.assertEqual(agent.tools, [])

    @patch("lg_adk.agents.base.get_model")
    def test_agent_run(self, mock_get_model) -> None:
        """Test that an agent can run."""
        mock_get_model.return_value = self.mock_model
        agent = Agent(
            name="test_agent",
            llm="ollama/llama3",
            description="Test agent",
        )
        state = {"input": "Hello"}
        result = agent.run(state)
        self.assertEqual(result["output"], "This is a test response")
        self.assertEqual(result["agent"], "test_agent")
        self.mock_model.invoke.assert_called_once()

    @patch("lg_adk.agents.base.get_model")
    def test_agent_tools(self, mock_get_model) -> None:
        """Test that tools can be added to an agent."""
        mock_get_model.return_value = self.mock_model
        agent = Agent(
            name="test_agent",
            llm="ollama/llama3",
            description="Test agent",
        )
        tool = MagicMock()
        tool.name = "test_tool"
        tool.description = "A test tool"
        agent.add_tool(tool)
        self.assertEqual(len(agent.tools), 1)
        self.assertEqual(agent.tools[0].name, "test_tool")
        tool2 = MagicMock()
        tool2.name = "test_tool2"
        agent.add_tools([tool2])
        self.assertEqual(len(agent.tools), 2)
        self.assertEqual(agent.tools[1].name, "test_tool2")
