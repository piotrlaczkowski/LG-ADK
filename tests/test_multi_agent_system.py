from unittest.mock import MagicMock, patch

import pytest

from lg_adk import Agent, MultiAgentSystem
from lg_adk.agents.multi_agent import Conversation


@pytest.fixture
def mock_agent() -> Agent:
    agent = MagicMock(spec=Agent)
    agent.name = "mock_agent"
    agent.description = "A mock agent"
    agent.run.return_value = {"output": "Mock response", "agent": "mock_agent"}
    agent.__call__ = MagicMock(return_value={"output": "Mock response", "agent": "mock_agent"})
    return agent


def test_multi_agent_system_creation_and_run(mock_agent):
    """Test creating and running a MultiAgentSystem with mocked agents."""
    coordinator = MagicMock(spec=Agent)
    coordinator.name = "coordinator"
    coordinator.description = "Coordinates tasks"
    coordinator.run.return_value = {"output": "Coordinator response", "agent": "coordinator"}
    coordinator.__call__ = MagicMock(return_value={"output": "Coordinator response", "agent": "coordinator"})

    agent1 = mock_agent
    agent1.name = "agent1"
    agent1.description = "Handles research"
    agent2 = mock_agent
    agent2.name = "agent2"
    agent2.description = "Handles writing"

    system = MultiAgentSystem(
        name="test_system",
        coordinator=coordinator,
        agents=[agent1, agent2],
        description="A test multi-agent system",
    )

    # Add another agent
    agent3 = mock_agent
    agent3.name = "agent3"
    agent3.description = "Handles summaries"
    system.add_agent(agent3)
    assert len(system.agents) == 3

    # Add multiple agents
    agent4 = mock_agent
    agent4.name = "agent4"
    agent4.description = "Handles creative tasks"
    system.add_agents([agent4])
    assert len(system.agents) == 4

    # Patch _build_graph and _graph.invoke to avoid real graph execution
    system._build_graph = MagicMock()
    mock_graph = MagicMock()
    system._build_graph.return_value = mock_graph
    system._graph = mock_graph
    mock_graph.invoke.return_value = {"output": "Final output", "agent": "coordinator"}

    # Run the system
    result = system.run({"input": "Test input"})
    assert result["output"] == "Final output"
    assert "system_message" in system._graph.invoke.call_args[0][0]


def test_conversation_helper_class(mock_agent):
    """Test the Conversation helper for multi-agent systems."""
    coordinator = mock_agent
    coordinator.name = "coordinator"
    coordinator.description = "Coordinates tasks"
    system = MultiAgentSystem(
        name="test_system",
        coordinator=coordinator,
        agents=[mock_agent],
        description="A test multi-agent system",
    )
    # Patch run at the class level to return a predictable response
    with patch.object(MultiAgentSystem, "run", return_value={"output": "Hello, user!"}):
        conversation = Conversation(multi_agent_system=system)
        response = conversation.send_message("Hi!")
        assert response == "Hello, user!"
        assert conversation.conversation_history[0]["role"] == "user"
        assert conversation.conversation_history[1]["role"] == "system"
        assert conversation.conversation_history[1]["content"] == "Hello, user!"
