"""
Tests for database-backed session management functionality.

This module tests the DatabaseSessionManager class which provides
persistent session storage using a database backend.
"""

import pytest
import time
import uuid
from typing import Dict, Any
from unittest.mock import MagicMock, patch

from lg_adk.sessions.session_manager import (
    DatabaseSessionManager,
    Session,
)
from lg_adk.database.database_manager import DatabaseManager


class MockDatabaseManager(DatabaseManager):
    """Mock implementation of DatabaseManager for testing."""
    
    def __init__(self):
        """Initialize with an in-memory store."""
        self.store = {}
        self.namespaces = set()
    
    def put(self, namespace: str, key: str, value: Any) -> None:
        """Store a value in the mock database."""
        if namespace not in self.namespaces:
            self.namespaces.add(namespace)
            self.store[namespace] = {}
        self.store[namespace][key] = value
    
    def get(self, namespace: str, key: str) -> Any:
        """Retrieve a value from the mock database."""
        if namespace not in self.store or key not in self.store[namespace]:
            return None
        return self.store[namespace][key]
    
    def delete(self, namespace: str, key: str) -> None:
        """Delete a value from the mock database."""
        if namespace in self.store and key in self.store[namespace]:
            del self.store[namespace][key]
    
    def list_keys(self, namespace: str) -> list:
        """List all keys in a namespace."""
        if namespace not in self.store:
            return []
        return list(self.store[namespace].keys())
    
    def exists(self, namespace: str, key: str) -> bool:
        """Check if a key exists in the namespace."""
        return namespace in self.store and key in self.store[namespace]


@pytest.fixture
def mock_db():
    """Provide a mock database manager for testing."""
    return MockDatabaseManager()


@pytest.fixture
def db_session_manager(mock_db):
    """Return a database session manager for testing."""
    return DatabaseSessionManager(db_manager=mock_db)


class TestDatabaseSessionManager:
    """Test the DatabaseSessionManager class."""
    
    def test_basic_functionality(self, db_session_manager):
        """Test that basic session management works with database persistence."""
        # Create session
        session_id = db_session_manager.create_session()
        
        # Verify session was created
        session = db_session_manager.get_session(session_id)
        assert session.session_id == session_id
        
        # Verify session was stored in database
        assert db_session_manager.db_manager.exists('sessions', session_id)
        
        # Update session
        db_session_manager.update_session(session_id)
        
        # End session
        success = db_session_manager.end_session(session_id)
        assert success
        
        # Verify session was removed from database
        assert not db_session_manager.db_manager.exists('sessions', session_id)
    
    def test_session_persistence(self, mock_db):
        """Test that sessions persist across manager instances."""
        # Create a session with the first manager
        manager1 = DatabaseSessionManager(db_manager=mock_db)
        session_id = manager1.create_session(metadata={"persistent": True})
        
        # Create a new manager instance
        manager2 = DatabaseSessionManager(db_manager=mock_db)
        
        # Verify the second manager can retrieve the session
        session = manager2.get_session(session_id)
        assert session.session_id == session_id
        assert session.metadata.get("persistent") is True
    
    def test_metadata_management(self, db_session_manager):
        """Test metadata management with database persistence."""
        # Create session with initial metadata
        initial_metadata = {"user": "test_user", "topic": "test_topic"}
        session_id = db_session_manager.create_session(metadata=initial_metadata)
        
        # Verify initial metadata
        session = db_session_manager.get_session(session_id)
        assert session.metadata == initial_metadata
        
        # Update metadata
        update_metadata = {"status": "active", "priority": "high"}
        db_session_manager.update_session_metadata(session_id, update_metadata)
        
        # Verify updated metadata (should be merged)
        updated_session = db_session_manager.get_session(session_id)
        expected_metadata = {**initial_metadata, **update_metadata}
        assert updated_session.metadata == expected_metadata
        
        # Verify metadata was stored in database
        stored_session = db_session_manager.db_manager.get('sessions', session_id)
        assert stored_session.metadata == expected_metadata
    
    def test_session_timeout(self, db_session_manager):
        """Test that sessions expire after timeout."""
        # Create session with short timeout
        session_id = db_session_manager.create_session(timeout=0.1)
        
        # Verify session exists initially
        session = db_session_manager.get_session(session_id)
        assert session.session_id == session_id
        
        # Wait for session to expire
        time.sleep(0.2)
        
        # Clean up expired sessions
        cleaned_count = db_session_manager.cleanup_expired_sessions()
        assert cleaned_count == 1
        
        # Verify session was removed from both memory and database
        with pytest.raises(KeyError):
            db_session_manager.get_session(session_id)
        assert not db_session_manager.db_manager.exists('sessions', session_id)
    
    def test_session_with_user_id(self, db_session_manager):
        """Test creating sessions with user IDs."""
        # Create a session with user ID
        user_id = "test_user_456"
        session_id = db_session_manager.create_session(user_id=user_id)
        
        # Verify session was created with the correct user ID
        session = db_session_manager.get_session(session_id)
        assert session.user_id == user_id
        
        # Verify user ID is stored in the database
        stored_session = db_session_manager.db_manager.get('sessions', session_id)
        assert stored_session.user_id == user_id
    
    def test_tracking_interactions(self, db_session_manager):
        """Test tracking session interactions with database persistence."""
        # Create a session
        session_id = db_session_manager.create_session()
        
        # Track interactions
        db_session_manager.track_interaction(
            session_id,
            tokens_in=10,
            tokens_out=5,
            response_time=0.1
        )
        
        # Get updated session
        session = db_session_manager.get_session(session_id)
        
        # Verify interactions were tracked
        assert session.interactions == 1
        assert session.total_tokens_in == 10
        assert session.total_tokens_out == 5
        
        # Verify interaction data is stored in database
        stored_session = db_session_manager.db_manager.get('sessions', session_id)
        assert stored_session.interactions == 1
        assert stored_session.total_tokens_in == 10
        assert stored_session.total_tokens_out == 5
    
    def test_multiple_interactions(self, db_session_manager):
        """Test tracking multiple interactions with database persistence."""
        # Create a session
        session_id = db_session_manager.create_session()
        
        # Track multiple interactions
        num_interactions = 5
        total_tokens_in = 0
        total_tokens_out = 0
        
        for i in range(num_interactions):
            tokens_in = 10 * (i + 1)
            tokens_out = 5 * (i + 1)
            total_tokens_in += tokens_in
            total_tokens_out += tokens_out
            
            db_session_manager.track_interaction(
                session_id,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                response_time=0.1 * (i + 1)
            )
            
            # Add a small delay to ensure timestamps are different
            time.sleep(0.01)
        
        # Get session
        session = db_session_manager.get_session(session_id)
        
        # Verify interactions were tracked
        assert session.interactions == num_interactions
        assert session.total_tokens_in == total_tokens_in
        assert session.total_tokens_out == total_tokens_out
        
        # Verify timestamps
        assert session.created_at < session.last_interaction_at
        
        # Verify interaction data is stored in database
        stored_session = db_session_manager.db_manager.get('sessions', session_id)
        assert stored_session.interactions == num_interactions
        assert stored_session.total_tokens_in == total_tokens_in
        assert stored_session.total_tokens_out == total_tokens_out
    
    def test_end_nonexistent_session(self, db_session_manager):
        """Test ending a session that doesn't exist."""
        # Try to end a nonexistent session
        success = db_session_manager.end_session("nonexistent_session_id")
        
        # Should return False, not raise an exception
        assert not success
    
    def test_cleanup_with_no_expired_sessions(self, db_session_manager):
        """Test cleaning up when there are no expired sessions."""
        # Create a session with a long timeout
        session_id = db_session_manager.create_session(timeout=3600)
        
        # Clean up expired sessions
        cleaned_count = db_session_manager.cleanup_expired_sessions()
        
        # Should be 0 since the session hasn't expired
        assert cleaned_count == 0
        
        # Session should still exist in both memory and database
        session = db_session_manager.get_session(session_id)
        assert session.session_id == session_id
        assert db_session_manager.db_manager.exists('sessions', session_id)
    
    def test_serialization_and_deserialization(self, db_session_manager):
        """Test session serialization/deserialization for database storage."""
        # Create a complex session
        session_id = str(uuid.uuid4())
        user_id = "complex_user"
        metadata = {
            "tags": ["important", "customer"],
            "priority": 5,
            "nested": {
                "field1": "value1",
                "field2": 123
            }
        }
        
        # Create the session
        db_session_manager.create_session(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata
        )
        
        # Get the session from the database directly
        stored_session = db_session_manager.db_manager.get('sessions', session_id)
        
        # Verify all fields were properly serialized and deserialized
        assert stored_session.session_id == session_id
        assert stored_session.user_id == user_id
        assert stored_session.metadata == metadata
        assert "nested" in stored_session.metadata
        assert stored_session.metadata["nested"]["field1"] == "value1"
    
    def test_database_failure_handling(self, mock_db):
        """Test handling of database failures."""
        # Create a session manager with a mock database
        manager = DatabaseSessionManager(db_manager=mock_db)
        
        # Create a session
        session_id = manager.create_session()
        
        # Mock a database failure during get
        with patch.object(mock_db, 'get', side_effect=Exception("Database connection error")):
            # Should use in-memory cache
            session = manager.get_session(session_id)
            assert session.session_id == session_id
        
        # Mock a database failure during cleanup
        with patch.object(mock_db, 'delete', side_effect=Exception("Database delete error")):
            # Should not raise an exception
            cleaned = manager.cleanup_expired_sessions()
            assert cleaned == 0
    
    def test_multiple_cleanup_runs(self, db_session_manager):
        """Test multiple cleanup runs with varying expired sessions."""
        # Create sessions with different timeouts
        short_timeout_id = db_session_manager.create_session(timeout=0.1)
        medium_timeout_id = db_session_manager.create_session(timeout=0.3)
        long_timeout_id = db_session_manager.create_session(timeout=1.0)
        
        # Wait for short timeout to expire
        time.sleep(0.2)
        
        # First cleanup should remove only short timeout session
        cleaned = db_session_manager.cleanup_expired_sessions()
        assert cleaned == 1
        
        # Verify short timeout session is gone
        with pytest.raises(KeyError):
            db_session_manager.get_session(short_timeout_id)
        
        # Medium and long timeout sessions should still exist
        assert db_session_manager.db_manager.exists('sessions', medium_timeout_id)
        assert db_session_manager.db_manager.exists('sessions', long_timeout_id)
        
        # Wait for medium timeout to expire
        time.sleep(0.2)
        
        # Second cleanup should remove only medium timeout session
        cleaned = db_session_manager.cleanup_expired_sessions()
        assert cleaned == 1
        
        # Verify medium timeout session is gone
        with pytest.raises(KeyError):
            db_session_manager.get_session(medium_timeout_id)
        
        # Long timeout session should still exist
        assert db_session_manager.db_manager.exists('sessions', long_timeout_id) 