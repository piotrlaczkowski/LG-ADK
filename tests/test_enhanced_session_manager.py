"""Tests for the EnhancedSessionManager class."""

import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set

import pytest
from unittest.mock import MagicMock, patch

from lg_adk.sessions.session_manager import (
    Session,
    EnhancedSessionManager,
    SessionManager,
    SynchronizedSessionManager,
    AsyncSessionManager
)


class TestEnhancedSessionManager:
    """Tests for the EnhancedSessionManager class."""

    def test_create_and_register_session(self):
        """Test registering a session with the manager."""
        manager = EnhancedSessionManager()
        session_id = str(uuid.uuid4())
        user_id = "test_user"
        metadata = {"source": "webapp", "device": "mobile"}

        # Register session
        manager.register_session(session_id, user_id, metadata)

        # Verify session was registered
        assert session_id in manager.session_metadata
        assert manager.session_metadata[session_id] == metadata
        assert session_id in manager.session_analytics
        assert user_id in manager.user_sessions
        assert session_id in manager.user_sessions[user_id]

    def test_track_interaction(self):
        """Test tracking interactions for a session."""
        manager = EnhancedSessionManager()
        session_id = str(uuid.uuid4())
        
        # Register session first
        manager.register_session(session_id)
        
        # Initial state
        initial_count = manager.session_analytics[session_id]["message_count"]
        
        # Track an interaction
        interaction_type = "message"
        details = {"input_length": 150, "has_response": True}
        manager.track_interaction(session_id, interaction_type, details)
        
        # Verify interaction was tracked
        assert manager.session_analytics[session_id]["message_count"] == initial_count + 1
        assert len(manager.session_analytics[session_id]["interaction_history"]) == 1
        assert manager.session_analytics[session_id]["interaction_history"][0]["type"] == interaction_type
        assert "details" in manager.session_analytics[session_id]["interaction_history"][0]

    def test_update_session_metadata(self):
        """Test updating session metadata."""
        manager = EnhancedSessionManager()
        session_id = str(uuid.uuid4())
        
        # Register session first
        initial_metadata = {"source": "test"}
        manager.register_session(session_id, metadata=initial_metadata)
        
        # Update metadata (merge)
        new_metadata = {"locale": "en-US"}
        result = manager.update_session_metadata(session_id, new_metadata, merge=True)
        
        # Verify metadata was updated and merged
        assert result is True
        assert manager.session_metadata[session_id]["source"] == "test"
        assert manager.session_metadata[session_id]["locale"] == "en-US"
        
        # Update metadata (replace)
        replacement_metadata = {"completely": "new"}
        result = manager.update_session_metadata(session_id, replacement_metadata, merge=False)
        
        # Verify metadata was replaced
        assert result is True
        assert manager.session_metadata[session_id] == replacement_metadata
        assert "source" not in manager.session_metadata[session_id]

    def test_get_user_sessions(self):
        """Test getting all sessions for a user."""
        manager = EnhancedSessionManager()
        user_id = "test_user"
        
        # Create multiple sessions for the same user
        session_id1 = str(uuid.uuid4())
        session_id2 = str(uuid.uuid4())
        session_id3 = str(uuid.uuid4())
        
        manager.register_session(session_id1, user_id)
        manager.register_session(session_id2, user_id)
        manager.register_session(session_id3, user_id)
        
        # Also create a session for a different user
        other_session = str(uuid.uuid4())
        manager.register_session(other_session, "other_user")
        
        # Get user sessions
        user_sessions = manager.get_user_sessions(user_id)
        
        # Verify user sessions
        assert len(user_sessions) == 3
        assert session_id1 in user_sessions
        assert session_id2 in user_sessions
        assert session_id3 in user_sessions
        assert other_session not in user_sessions

    def test_end_session(self):
        """Test ending a session."""
        manager = EnhancedSessionManager()
        session_id = str(uuid.uuid4())
        user_id = "test_user"
        
        # Register session
        manager.register_session(session_id, user_id)
        
        # End session
        result = manager.end_session(session_id)
        
        # Verify session was ended
        assert result is True
        assert session_id not in manager.session_metadata
        assert session_id not in manager.session_analytics
        assert session_id not in manager.user_sessions.get(user_id, set())

    def test_prepare_metadata(self):
        """Test preparing metadata for a session."""
        manager = EnhancedSessionManager(default_timeout=timedelta(hours=1))
        metadata = {"source": "test"}
        
        # Prepare metadata
        prepared = manager.prepare_metadata(metadata)
        
        # Verify metadata was prepared
        assert prepared["source"] == "test"
        assert prepared["_lg_adk_enhanced"] is True
        assert "_created_at" in prepared
        assert "_timeout" in prepared
        assert prepared["_timeout"] == 3600.0

    def test_langgraph_store_adapter(self):
        """Test the LangGraph store adapter."""
        manager = EnhancedSessionManager()
        
        # Create the adapter
        adapter = manager._as_langgraph_store()
        
        # Test session handling
        session_id = str(uuid.uuid4())
        session_data = {"metadata": {"source": "test"}}
        
        # Set session data
        adapter.set(session_id, session_data)
        
        # Verify session exists
        assert adapter.exists(session_id) is True
        
        # Get session data
        retrieved = adapter.get(session_id)
        assert retrieved is not None
        assert retrieved["id"] == session_id
        assert retrieved["metadata"]["source"] == "test"
        
        # Delete session
        adapter.delete(session_id)
        
        # Verify session no longer exists
        assert adapter.exists(session_id) is False


class TestSessionManagerCompatibility:
    """Tests for backward compatibility with SessionManager."""
    
    def test_backward_compatibility(self):
        """Test that SessionManager is backward compatible with old code."""
        # Create a session manager
        manager = SessionManager()
        
        # Use the old API
        session_id = manager.create_session(user_id="test_user", metadata={"source": "test"})
        
        # Get the session using the old API
        session = manager.get_session(session_id)
        
        # Verify session properties
        assert session.id == session_id
        assert session.user_id == "test_user"
        assert session.metadata["source"] == "test"
        
        # Verify enhanced features are also working
        assert session_id in manager.session_metadata
        assert "test_user" in manager.user_sessions


class TestAsyncSessionManager:
    """Tests for the AsyncSessionManager."""
    
    @pytest.mark.asyncio
    async def test_async_register_session(self):
        """Test registering a session asynchronously."""
        manager = AsyncSessionManager()
        session_id = str(uuid.uuid4())
        user_id = "test_user"
        metadata = {"source": "test"}
        
        # Register session asynchronously
        await manager.register_session_async(session_id, user_id, metadata)
        
        # Verify session was registered in the underlying manager
        assert session_id in manager.session_manager.session_metadata
        
    @pytest.mark.asyncio
    async def test_async_track_interaction(self):
        """Test tracking interactions asynchronously."""
        manager = AsyncSessionManager()
        session_id = str(uuid.uuid4())
        
        # Register session first
        await manager.register_session_async(session_id)
        
        # Track an interaction asynchronously
        await manager.track_interaction_async(session_id, "message", {"test": "data"})
        
        # Verify interaction was tracked
        assert manager.session_manager.session_analytics[session_id]["message_count"] == 1


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 