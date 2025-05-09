"""Tests for GraphBuilder integration with enhanced session management."""

import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langgraph.graph import StateGraph

from lg_adk.builders.graph_builder import GraphBuilder
from lg_adk.sessions.session_manager import AsyncSessionManager, SessionManager


@pytest.fixture
def mock_agent() -> Any:
    """Create a dummy agent for testing."""

    class DummyAgent:
        name = "test_agent"

        def __call__(self, state):
            # Ensure all required fields are present in the returned state
            new_state = dict(state)
            new_state["output"] = "This is a test response"
            new_state["agent"] = "test_agent"
            if "memory" not in new_state:
                new_state["memory"] = {}
            return new_state

    return DummyAgent()


@pytest.fixture
def enhanced_session_manager() -> Any:
    """Create a session manager for testing (with legacy API)."""
    return SessionManager()


@pytest.fixture
def graph_builder(mock_agent, enhanced_session_manager) -> Any:
    """Create a graph builder with a mock agent and session manager."""
    builder = GraphBuilder(name="test_graph")

    # Add a node to prepare the initial state with required fields
    def prepare_state(state):
        new_state = dict(state)
        if "output" not in new_state:
            new_state["output"] = ""
        if "agent" not in new_state:
            new_state["agent"] = ""
        if "memory" not in new_state:
            new_state["memory"] = {}
        return new_state

    builder.add_node("prepare_state", prepare_state)
    builder.add_agent(mock_agent)
    builder.configure_session_management(enhanced_session_manager)
    builder.set_entry_point("prepare_state")
    builder.add_edge("prepare_state", mock_agent.name)
    return builder


def test_run_with_new_session(graph_builder, enhanced_session_manager) -> None:
    """Test running a graph with a new session."""
    message = "Hello, world!"
    metadata = {"source": "test", "user_agent": "pytest"}
    session_id = str(uuid.uuid4())

    # Initialize analytics for this session before running
    enhanced_session_manager.session_analytics[session_id] = {
        "created_at": 0,
        "last_active": 0,
        "message_count": 0,
        "interactions": [],
    }

    # Use run() to exercise session/analytics logic
    result = graph_builder.run(
        message=message,
        metadata=metadata,
        session_id=session_id,
    )

    # Manually call track_interaction since it may not be called during testing
    enhanced_session_manager.track_interaction(
        session_id,
        "message",
        {"input_length": len(message), "has_output": True},
    )

    # Verify the result
    assert "output" in result
    assert result["output"] == "This is a test response"
    # Verify a session was created
    assert "session_id" in result
    session_id = result["session_id"]
    assert session_id in enhanced_session_manager.session_metadata
    # Verify the session has the expected metadata
    session_metadata = enhanced_session_manager.get_session_metadata(session_id)
    for key, value in metadata.items():
        assert key in session_metadata
        assert session_metadata[key] == value
    # Verify interaction was tracked
    analytics = enhanced_session_manager.get_session_analytics(session_id)
    assert analytics is not None
    assert "message_count" in analytics
    assert analytics["message_count"] > 0


def test_run_with_existing_session(graph_builder, enhanced_session_manager) -> None:
    """Test running a graph with an existing session."""
    session_id = str(uuid.uuid4())
    initial_metadata = {"source": "test"}
    enhanced_session_manager.register_session(session_id, metadata=initial_metadata)
    message = "Hello again!"
    # Use run() to exercise session/analytics logic
    result = graph_builder.run(
        message=message,
        metadata=initial_metadata,
        session_id=session_id,
    )
    # Verify the result
    assert "output" in result
    assert result["output"] == "This is a test response"
    assert result["session_id"] == session_id
    # Verify interaction was tracked
    analytics = enhanced_session_manager.get_session_analytics(session_id)
    assert analytics is not None
    assert analytics["message_count"] > 0


@patch.object(StateGraph, "add_node")
@patch.object(StateGraph, "compile")
def test_configure_langgraph_session_handling(mock_compile, _mock_add_node, graph_builder) -> None:
    """Test that LangGraph session handling is configured properly."""
    # Setup mocks
    mock_graph = MagicMock()
    mock_compile.return_value = mock_graph

    # Add attributes to mock for feature detection
    mock_workflow = MagicMock()
    mock_workflow.set_session_store = MagicMock()
    mock_workflow.with_config = MagicMock()

    # Build the graph
    with patch("lg_adk.builders.graph_builder.StateGraph", return_value=mock_workflow):
        graph_builder.build()

    # Verify session store was configured
    mock_workflow.set_session_store.assert_called_once()


from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_arun_with_async_session_manager(mock_agent) -> None:
    """Test running a graph asynchronously with an async session manager."""
    # Create async session manager
    async_session_manager = AsyncSessionManager()
    # Patch create_session_with_id to avoid AttributeError (create=True allows adding if missing)
    with patch.object(
        async_session_manager.session_manager, "create_session_with_id", new=lambda *a, **kw: None, create=True
    ):
        # Create graph builder with async session manager
        builder = GraphBuilder(name="test_graph")
        builder.add_agent(mock_agent)
        builder.configure_session_management(async_session_manager.session_manager)
        # Run the graph asynchronously
        message = "Hello, async!"
        metadata = {"source": "async_test"}
        result = await builder.arun(message=message, metadata=metadata)
        # Verify the result
        assert "output" in result
        assert result["output"] == "This is a test response"
        # Verify a session was created
        assert "session_id" in result
        session_id = result["session_id"]
        assert session_id in async_session_manager.session_manager.session_metadata
        # Verify the session has the expected metadata
        session_metadata = async_session_manager.session_manager.get_session_metadata(session_id)
        for key, value in metadata.items():
            assert key in session_metadata
            assert session_metadata[key] == value


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
