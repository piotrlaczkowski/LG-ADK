"""Tests for GraphBuilder integration with enhanced session management."""

import uuid
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, patch

import pytest
from langgraph.graph import StateGraph

from lg_adk.builders.graph_builder import GraphBuilder
from lg_adk.sessions.session_manager import EnhancedSessionManager, AsyncSessionManager
from lg_adk.agents.base import Agent


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = MagicMock(spec=Agent)
    agent.name = "test_agent"
    agent.__call__.side_effect = lambda state: {
        **state,
        "output": "This is a test response",
        "agent": "test_agent"
    }
    return agent


@pytest.fixture
def enhanced_session_manager():
    """Create an enhanced session manager for testing."""
    return EnhancedSessionManager()


@pytest.fixture
def graph_builder(mock_agent, enhanced_session_manager):
    """Create a graph builder with a mock agent and session manager."""
    builder = GraphBuilder(name="test_graph")
    builder.add_agent(mock_agent)
    builder.configure_session_management(enhanced_session_manager)
    return builder


class TestGraphBuilderSessions:
    """Tests for GraphBuilder's session management integration."""

    def test_run_with_new_session(self, graph_builder, enhanced_session_manager):
        """Test running a graph with a new session."""
        message = "Hello, world!"
        metadata = {"source": "test", "user_agent": "pytest"}
        
        # Run the graph
        result = graph_builder.run(message=message, metadata=metadata)
        
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
        assert analytics["message_count"] > 0

    def test_run_with_existing_session(self, graph_builder, enhanced_session_manager):
        """Test running a graph with an existing session."""
        # Create a session first
        session_id = str(uuid.uuid4())
        initial_metadata = {"source": "test"}
        enhanced_session_manager.register_session(session_id, metadata=initial_metadata)
        
        # Run the graph with the existing session
        message = "Hello again!"
        result = graph_builder.run(message=message, session_id=session_id)
        
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
    def test_configure_langgraph_session_handling(self, mock_compile, mock_add_node, graph_builder, enhanced_session_manager):
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
            graph = graph_builder.build()
        
        # Verify session store was configured
        mock_workflow.set_session_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_arun_with_async_session_manager(self, mock_agent):
        """Test running a graph asynchronously with an async session manager."""
        # Create async session manager
        async_session_manager = AsyncSessionManager()
        
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