"""
Tests for SessionManager functionality.

This module tests the SessionManager classes and related functionality
in the lg_adk.sessions module.
"""

import pytest
from datetime import datetime, timedelta
import time
from typing import Dict, Any, Optional
import uuid

from lg_adk.sessions import (
    Session,
    SessionManager,
    SynchronizedSessionManager,
    DatabaseSessionManager,
    AsyncSessionManager,
)


@pytest.fixture
def basic_session_manager():
    """Return a basic session manager for testing."""
    return SessionManager()


@pytest.fixture
def session_manager_with_timeout():
    """Return a session manager with timeout for testing."""
    return SessionManager(default_timeout=timedelta(seconds=1))


@pytest.fixture
def mock_db_manager():
    """Mock database manager for testing."""
    class MockDBManager:
        def __init__(self):
            self.executed_queries = []
            self.tables = {}
            
        def execute(self, query, params=None):
            self.executed_queries.append((query, params))
            # Simple CREATE TABLE handling
            if query.strip().startswith("CREATE TABLE"):
                table_name = query.split("CREATE TABLE IF NOT EXISTS")[1].split("(")[0].strip()
                self.tables[table_name] = True
            return True
            
    return MockDBManager()


class TestSession:
    """Test the Session class."""
    
    def test_session_initialization(self):
        """Test session initialization."""
        # Test with default values
        session = Session()
        assert session.id is not None
        assert session.user_id is None
        assert isinstance(session.created_at, datetime)
        assert session.metadata == {}
        assert session.timeout is None
        
        # Test with provided values
        custom_id = "test-session-id"
        user_id = "test-user"
        metadata = {"key": "value"}
        timeout = timedelta(minutes=30)
        
        session = Session(
            session_id=custom_id,
            user_id=user_id,
            metadata=metadata,
            timeout=timeout
        )
        
        assert session.id == custom_id
        assert session.user_id == user_id
        assert session.metadata == metadata
        assert session.timeout == timeout
    
    def test_update_last_active(self):
        """Test updating last active timestamp."""
        session = Session()
        initial_timestamp = session.last_active
        time.sleep(0.01)  # Small delay to ensure timestamp difference
        session.update_last_active()
        assert session.last_active > initial_timestamp
    
    def test_is_expired(self):
        """Test session expiration check."""
        # Session with no timeout should never expire
        session = Session()
        assert not session.is_expired()
        
        # Session with future timeout should not be expired
        session = Session(timeout=timedelta(hours=1))
        assert not session.is_expired()
        
        # Session with past timeout should be expired
        session = Session(timeout=timedelta(microseconds=1))
        time.sleep(0.01)  # Small delay to ensure expiration
        assert session.is_expired()
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        original = Session(
            session_id="test-id",
            user_id="test-user",
            metadata={"test": "value"},
            timeout=timedelta(minutes=30)
        )
        
        # Convert to dict
        as_dict = original.to_dict()
        
        # Key fields should be present
        assert as_dict["id"] == "test-id"
        assert as_dict["user_id"] == "test-user"
        assert as_dict["metadata"] == {"test": "value"}
        assert as_dict["timeout"] == 30 * 60  # Minutes to seconds
        
        # Recreate from dict
        recreated = Session.from_dict(as_dict)
        
        # Check recreation worked
        assert recreated.id == original.id
        assert recreated.user_id == original.user_id
        assert recreated.metadata == original.metadata
        assert recreated.timeout.total_seconds() == original.timeout.total_seconds()


class TestSessionManager:
    """Test the SessionManager class."""
    
    def test_create_session(self, basic_session_manager):
        """Test session creation."""
        session_id = basic_session_manager.create_session()
        assert session_id is not None
        assert basic_session_manager.session_exists(session_id)
        
        # Test with custom parameters
        user_id = "test-user"
        metadata = {"test": "value"}
        custom_session_id = "custom-session-id"
        timeout = 3600
        
        session_id = basic_session_manager.create_session(
            user_id=user_id,
            metadata=metadata,
            session_id=custom_session_id,
            timeout=timeout
        )
        
        assert session_id == custom_session_id
        session = basic_session_manager.get_session(session_id)
        assert session.user_id == user_id
        assert session.metadata == metadata
        assert session.timeout.total_seconds() == timeout
    
    def test_create_session_with_id(self, basic_session_manager):
        """Test creating a session with a specific ID."""
        session_id = "test-specific-id"
        basic_session_manager.create_session_with_id(session_id)
        
        assert basic_session_manager.session_exists(session_id)
    
    def test_get_session(self, basic_session_manager):
        """Test retrieving a session."""
        # Non-existent session
        assert basic_session_manager.get_session("nonexistent") is None
        
        # Existing session
        session_id = basic_session_manager.create_session()
        session = basic_session_manager.get_session(session_id)
        assert session is not None
        assert session.id == session_id
    
    def test_update_session(self, basic_session_manager):
        """Test updating a session."""
        # Create a session
        session_id = basic_session_manager.create_session()
        original_session = basic_session_manager.get_session(session_id)
        
        # Wait briefly to ensure timestamp difference
        time.sleep(0.01)
        
        # Update the session (just timestamp)
        basic_session_manager.update_session(session_id)
        updated_session = basic_session_manager.get_session(session_id)
        
        assert updated_session.last_active > original_session.last_active
        
        # Create a new session object and use it to update
        new_session = Session(
            session_id=session_id,
            user_id="new-user",
            metadata={"updated": True}
        )
        
        basic_session_manager.update_session(session_id, new_session)
        replaced_session = basic_session_manager.get_session(session_id)
        
        assert replaced_session.user_id == "new-user"
        assert replaced_session.metadata == {"updated": True}
    
    def test_remove_session(self, basic_session_manager):
        """Test removing a session."""
        # Non-existent session
        assert not basic_session_manager.remove_session("nonexistent")
        
        # Existing session
        session_id = basic_session_manager.create_session()
        assert basic_session_manager.session_exists(session_id)
        
        assert basic_session_manager.remove_session(session_id)
        assert not basic_session_manager.session_exists(session_id)
    
    def test_get_all_sessions(self, basic_session_manager):
        """Test retrieving all sessions."""
        # Initially empty
        assert len(basic_session_manager.get_all_sessions()) == 0
        
        # Add some sessions
        basic_session_manager.create_session()
        basic_session_manager.create_session()
        basic_session_manager.create_session()
        
        # Should have 3 sessions
        assert len(basic_session_manager.get_all_sessions()) == 3
    
    def test_session_expiration(self, session_manager_with_timeout):
        """Test session expiration handling."""
        # Create a session with timeout
        session_id = session_manager_with_timeout.create_session()
        assert session_manager_with_timeout.session_exists(session_id)
        
        # Wait for expiration
        time.sleep(1.1)  # Just over the 1-second timeout
        
        # Session should be marked as expired and removed when accessed
        assert not session_manager_with_timeout.session_exists(session_id)
        assert session_manager_with_timeout.get_session(session_id) is None
    
    def test_cleanup_expired_sessions(self, session_manager_with_timeout):
        """Test cleanup of expired sessions."""
        # Create some sessions
        session_1 = session_manager_with_timeout.create_session()
        session_2 = session_manager_with_timeout.create_session()
        session_3 = session_manager_with_timeout.create_session()
        
        # Wait for expiration
        time.sleep(1.1)  # Just over the 1-second timeout
        
        # Clean up expired sessions
        removed_count = session_manager_with_timeout.cleanup_expired_sessions()
        assert removed_count == 3
        
        # All sessions should be gone
        assert len(session_manager_with_timeout.get_all_sessions()) == 0
    
    def test_user_sessions(self, basic_session_manager):
        """Test user session associations."""
        # Create sessions for user1
        user1_session1 = basic_session_manager.create_session(user_id="user1")
        user1_session2 = basic_session_manager.create_session(user_id="user1")
        
        # Create session for user2
        user2_session = basic_session_manager.create_session(user_id="user2")
        
        # Check user session associations
        user1_sessions = basic_session_manager.get_user_sessions("user1")
        assert len(user1_sessions) == 2
        assert user1_session1 in user1_sessions
        assert user1_session2 in user1_sessions
        
        user2_sessions = basic_session_manager.get_user_sessions("user2")
        assert len(user2_sessions) == 1
        assert user2_session in user2_sessions
    
    def test_session_analytics(self, basic_session_manager):
        """Test session analytics tracking."""
        # Create a session
        session_id = basic_session_manager.create_session()
        
        # Track some interactions
        basic_session_manager.track_interaction(
            session_id, 
            "message", 
            {"content": "Hello"}
        )
        
        basic_session_manager.track_interaction(
            session_id, 
            "tool_use", 
            {"tool": "calculator"}
        )
        
        # Get analytics
        analytics = basic_session_manager.get_session_analytics(session_id)
        assert analytics is not None
        assert analytics["message_count"] == 2
        assert len(analytics["interaction_history"]) == 2
        
        # Check interaction details
        interactions = analytics["interaction_history"]
        assert interactions[0]["type"] == "message"
        assert interactions[0]["details"]["content"] == "Hello"
        assert interactions[1]["type"] == "tool_use"
        assert interactions[1]["details"]["tool"] == "calculator"


class TestSynchronizedSessionManager:
    """Test the SynchronizedSessionManager class."""
    
    def test_basic_operations(self):
        """Test that basic operations work with synchronization."""
        manager = SynchronizedSessionManager()
        
        # Create a session
        session_id = manager.create_session()
        assert manager.session_exists(session_id)
        
        # Update a session
        manager.update_session(session_id)
        
        # Remove a session
        assert manager.remove_session(session_id)
        assert not manager.session_exists(session_id)


class TestDatabaseSessionManager:
    """Test the DatabaseSessionManager class."""
    
    def test_initialization(self, mock_db_manager):
        """Test that database manager initializes properly."""
        db_manager = DatabaseSessionManager(mock_db_manager)
        
        # Should have created tables
        assert "sessions" in mock_db_manager.tables
        assert "session_analytics" in mock_db_manager.tables
        
        # Verify that the execute method was called
        assert len(mock_db_manager.executed_queries) >= 2
    
    def test_register_session(self, mock_db_manager):
        """Test registering a session in the database."""
        db_manager = DatabaseSessionManager(mock_db_manager)
        
        # Clear queries from initialization
        mock_db_manager.executed_queries = []
        
        # Register a session
        session_id = "test-db-session"
        db_manager.register_session(session_id, "test-user", {"test": "value"})
        
        # Verify SQL was executed
        assert len(mock_db_manager.executed_queries) >= 1
        
        # Check that the session exists
        assert session_id in db_manager.session_analytics


class TestAsyncSessionManager:
    """Test the AsyncSessionManager class."""
    
    @pytest.mark.asyncio
    async def test_create_session(self):
        """Test creating a session asynchronously."""
        manager = AsyncSessionManager()
        
        session_id = await manager.create_session(
            user_id="async-user",
            metadata={"async": True}
        )
        
        # Verify session was created in underlying manager
        assert session_id in manager.session_manager.session_metadata 