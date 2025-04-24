"""
Tests for session management in GraphBuilder.
"""

import pytest
from unittest.mock import MagicMock, patch
import uuid

from lg_adk.builders.graph_builder import GraphBuilder
from lg_adk.sessions.session_manager import SessionManager, Session
from lg_adk.memory.memory_manager import MemoryManager
from lg_adk.agents.base import Agent


@pytest.fixture
def mock_agent():
    """Creates a mock agent."""
    agent = MagicMock(spec=Agent)
    agent.name = "test_agent"
    agent.__call__.return_value = {"output": "Test response", "agent": "test_agent"}
    return agent


@pytest.fixture
def mock_session_manager():
    """Creates a mock session manager."""
    session_manager = MagicMock(spec=SessionManager)
    test_session_id = str(uuid.uuid4())
    
    # Setup mock methods
    session_manager.create_session.return_value = test_session_id
    session_manager.create_session_id.return_value = test_session_id
    session_manager.get_session.return_value = Session(
        id=test_session_id,
        user_id="test_user",
        metadata={"test": "metadata"}
    )
    
    return session_manager


@pytest.fixture
def mock_memory_manager():
    """Creates a mock memory manager."""
    memory_manager = MagicMock(spec=MemoryManager)
    memory_manager.get_session_messages.return_value = [
        {"role": "user", "content": "Hello"}
    ]
    return memory_manager


def test_configure_session_management(mock_session_manager):
    """Test configuring session management in GraphBuilder."""
    builder = GraphBuilder()
    builder.configure_session_management(mock_session_manager)
    
    assert builder.session_manager == mock_session_manager


def test_create_session(mock_session_manager):
    """Test creating a session through GraphBuilder."""
    builder = GraphBuilder()
    builder.configure_session_management(mock_session_manager)
    
    session_id = builder.create_session(user_id="test_user", metadata={"test": "metadata"})
    
    mock_session_manager.create_session.assert_called_once_with(
        user_id="test_user", 
        metadata={"test": "metadata"}, 
        timeout=3600
    )
    assert session_id == mock_session_manager.create_session.return_value
    assert session_id in builder.active_sessions


def test_get_session(mock_session_manager):
    """Test retrieving a session through GraphBuilder."""
    builder = GraphBuilder()
    builder.configure_session_management(mock_session_manager)
    
    session_id = "test-session-id"
    session = builder.get_session(session_id)
    
    mock_session_manager.get_session.assert_called_once_with(session_id)
    assert session == mock_session_manager.get_session.return_value


def test_update_session_metadata(mock_session_manager):
    """Test updating session metadata through GraphBuilder."""
    builder = GraphBuilder()
    builder.configure_session_management(mock_session_manager)
    
    session_id = "test-session-id"
    metadata = {"updated": "value"}
    builder.update_session_metadata(session_id, metadata)
    
    mock_session_manager.update_session_metadata.assert_called_once_with(
        session_id, metadata, merge=True
    )


def test_end_session(mock_session_manager, mock_memory_manager):
    """Test ending a session through GraphBuilder."""
    builder = GraphBuilder()
    builder.configure_session_management(mock_session_manager)
    builder.add_memory(mock_memory_manager)
    
    session_id = "test-session-id"
    builder.active_sessions.add(session_id)
    
    builder.end_session(session_id)
    
    mock_session_manager.remove_session.assert_called_once_with(session_id)
    mock_memory_manager.clear_session_memories.assert_called_once_with(session_id)
    assert session_id not in builder.active_sessions


def test_get_session_history(mock_session_manager, mock_memory_manager):
    """Test retrieving session history through GraphBuilder."""
    builder = GraphBuilder()
    builder.configure_session_management(mock_session_manager)
    builder.add_memory(mock_memory_manager)
    
    session_id = "test-session-id"
    history = builder.get_session_history(session_id)
    
    mock_memory_manager.get_session_messages.assert_called_once_with(session_id)
    assert history == mock_memory_manager.get_session_messages.return_value


@patch("lg_adk.builders.graph_builder.StateGraph")
def test_run_with_session_management(MockStateGraph, mock_agent, mock_session_manager):
    """Test running a graph with session management."""
    # Mock the StateGraph and graph
    mock_graph = MagicMock()
    MockStateGraph.return_value = mock_graph
    compiled_graph = MagicMock()
    mock_graph.compile.return_value = compiled_graph
    compiled_graph.invoke.return_value = {
        "input": "Hello",
        "output": "Test response",
        "agent": "test_agent",
        "session_id": "test-session-id",
        "metadata": {"test": "metadata"},
        "messages": [{"role": "user", "content": "Hello"}]
    }
    
    # Create builder with agent and session manager
    builder = GraphBuilder()
    builder.add_agent(mock_agent)
    builder.configure_session_management(mock_session_manager)
    
    # Run with session management
    result = builder.run(
        message="Hello",
        session_id="test-session-id",
        metadata={"test": "metadata"}
    )
    
    # Check that session was updated
    mock_session_manager.update_session.assert_called_once_with("test-session-id")
    
    # Check result has expected fields
    assert result["input"] == "Hello"
    assert result["output"] == "Test response"
    assert result["session_id"] == "test-session-id"
    assert result["metadata"] == {"test": "metadata"}
    assert "messages" in result


@patch("lg_adk.builders.graph_builder.StateGraph")
def test_configure_state_tracking(MockStateGraph, mock_agent):
    """Test configuring state tracking options."""
    # Mock the StateGraph and graph
    mock_graph = MagicMock()
    MockStateGraph.return_value = mock_graph
    mock_graph.compile.return_value = MagicMock()
    
    # Create builder and configure state tracking
    builder = GraphBuilder()
    builder.add_agent(mock_agent)
    builder.configure_state_tracking(
        include_session_id=True,
        include_metadata=False,
        include_messages=True
    )
    
    # Build the graph
    builder.build()
    
    # Verify state tracking configuration
    assert builder.state_tracking == {
        "include_session_id": True,
        "include_metadata": False,
        "include_messages": True
    } 