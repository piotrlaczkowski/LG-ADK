"""Tests for session management in GraphBuilder."""

import unittest
import uuid
from unittest.mock import MagicMock, patch

from lg_adk.agents.base import Agent
from lg_adk.builders.graph_builder import GraphBuilder
from lg_adk.memory.memory_manager import MemoryManager
from lg_adk.sessions.session_manager import Session, SessionManager


class TestSessionManagement(unittest.TestCase):
    """Tests for session management in GraphBuilder."""

    def setUp(self) -> None:
        """Set up mock agent, session manager, and memory manager for each test."""
        self.mock_agent = MagicMock(spec=Agent)
        self.mock_agent.name = "test_agent"
        self.mock_agent.__call__.return_value = {"output": "Test response", "agent": "test_agent"}

        self.mock_session_manager = MagicMock(spec=SessionManager)
        test_session_id = str(uuid.uuid4())
        self.mock_session_manager.create_session.return_value = test_session_id
        self.mock_session_manager.create_session_id.return_value = test_session_id
        self.mock_session_manager.get_session.return_value = Session(
            id=test_session_id,
            user_id="test_user",
            metadata={"test": "metadata"},
        )

        self.mock_memory_manager = MagicMock(spec=MemoryManager)
        self.mock_memory_manager.get_session_messages.return_value = [
            {"role": "user", "content": "Hello"},
        ]

    def test_configure_session_management(self) -> None:
        """Test configuring session management in GraphBuilder."""
        builder = GraphBuilder()
        builder.configure_session_management(self.mock_session_manager)
        self.assertEqual(builder.session_manager, self.mock_session_manager)

    def test_create_session(self) -> None:
        """Test creating a session through GraphBuilder."""
        builder = GraphBuilder()
        builder.configure_session_management(self.mock_session_manager)
        session_id = builder.create_session(user_id="test_user", metadata={"test": "metadata"})
        self.mock_session_manager.create_session.assert_called_once_with(
            user_id="test_user",
            metadata={"test": "metadata"},
            timeout=3600,
        )
        self.assertEqual(session_id, self.mock_session_manager.create_session.return_value)
        self.assertIn(session_id, builder.active_sessions)

    def test_get_session(self) -> None:
        """Test retrieving a session through GraphBuilder."""
        builder = GraphBuilder()
        builder.configure_session_management(self.mock_session_manager)
        session_id = "test-session-id"
        self.mock_session_manager.get_session.return_value = "session_obj"
        session = builder.get_session(session_id)
        self.mock_session_manager.get_session.assert_called_once_with(session_id)
        self.assertEqual(session, self.mock_session_manager.get_session.return_value)

    def test_update_session_metadata(self) -> None:
        """Test updating session metadata through GraphBuilder."""
        builder = GraphBuilder()
        builder.configure_session_management(self.mock_session_manager)
        session_id = "test-session-id"
        metadata = {"updated": "value"}
        builder.update_session_metadata(session_id, metadata)
        self.mock_session_manager.update_session_metadata.assert_called_once_with(
            session_id,
            metadata,
            merge=True,
        )

    def test_end_session(self) -> None:
        """Test ending a session through GraphBuilder."""
        builder = GraphBuilder()
        builder.configure_session_management(self.mock_session_manager)
        builder.add_memory(self.mock_memory_manager)
        session_id = "test-session-id"
        builder.active_sessions.add(session_id)
        builder.end_session(session_id)
        self.mock_session_manager.remove_session.assert_called_once_with(session_id)
        self.mock_memory_manager.clear_session_memories.assert_called_once_with(session_id)
        self.assertNotIn(session_id, builder.active_sessions)

    def test_get_session_history(self) -> None:
        """Test retrieving session history through GraphBuilder."""
        builder = GraphBuilder()
        builder.configure_session_management(self.mock_session_manager)
        builder.add_memory(self.mock_memory_manager)
        session_id = "test-session-id"
        self.mock_memory_manager.get_session_messages.return_value = ["msg1", "msg2"]
        history = builder.get_session_history(session_id)
        self.mock_memory_manager.get_session_messages.assert_called_once_with(session_id)
        self.assertEqual(history, self.mock_memory_manager.get_session_messages.return_value)

    @patch("lg_adk.builders.graph_builder.StateGraph")
    def test_run_with_session_management(self, mock_state_graph) -> None:
        """Test running a graph with session management."""
        mock_graph = MagicMock()
        mock_state_graph.return_value = mock_graph
        compiled_graph = MagicMock()
        mock_graph.compile.return_value = compiled_graph
        compiled_graph.invoke.return_value = {
            "input": "Hello",
            "output": "Test response",
            "agent": "test_agent",
            "session_id": "test-session-id",
            "metadata": {"test": "metadata"},
            "messages": [{"role": "user", "content": "Hello"}],
        }
        builder = GraphBuilder()
        builder.add_agent(self.mock_agent)
        builder.configure_session_management(self.mock_session_manager)
        result = builder.run(
            message="Hello",
            session_id="test-session-id",
            metadata={"test": "metadata"},
        )
        self.mock_session_manager.update_session.assert_called_once_with("test-session-id")
        self.assertEqual(result["input"], "Hello")
        self.assertEqual(result["output"], "Test response")
        self.assertEqual(result["session_id"], "test-session-id")
        self.assertEqual(result["metadata"], {"test": "metadata"})
        self.assertIn("messages", result)

    @patch("lg_adk.builders.graph_builder.StateGraph")
    def test_configure_state_tracking(self, mock_state_graph) -> None:
        """Test configuring state tracking options in GraphBuilder."""
        mock_graph = MagicMock()
        mock_state_graph.return_value = mock_graph
        mock_graph.compile.return_value = MagicMock()
        builder = GraphBuilder()
        builder.add_agent(self.mock_agent)
        builder.configure_state_tracking(
            include_session_id=True,
            include_metadata=False,
            include_messages=True,
        )
        builder.build()
        self.assertEqual(
            builder.state_tracking,
            {
                "include_session_id": True,
                "include_metadata": False,
                "include_messages": True,
            },
        )
