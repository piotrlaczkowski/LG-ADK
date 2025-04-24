"""Tests for the SessionManager class."""

import time
import uuid
from datetime import datetime, timedelta
import threading
import unittest
from unittest.mock import MagicMock, patch

import pytest

from lg_adk.sessions.session_manager import (
    Session,
    SessionManager,
    SynchronizedSessionManager,
    DatabaseSessionManager,
    AsyncSessionManager,
)


class TestSession:
    """Tests for the Session class."""

    def test_session_creation(self):
        """Test that a session can be created with the expected attributes."""
        session_id = str(uuid.uuid4())
        user_id = "user123"
        metadata = {"source": "web"}
        
        session = Session(id=session_id, user_id=user_id, metadata=metadata)
        
        assert session.id == session_id
        assert session.user_id == user_id
        assert session.metadata == metadata
        assert session.created_at is not None
        assert session.last_active is not None
        assert session.timeout == 3600  # Default timeout

    def test_session_is_expired(self):
        """Test that a session correctly reports when it is expired."""
        session = Session(id="test", timeout=10)
        
        # New session should not be expired
        assert not session.is_expired()
        
        # Set last_active to 11 seconds ago
        session.last_active = datetime.now() - timedelta(seconds=11)
        assert session.is_expired()
        
        # Set last_active to 9 seconds ago
        session.last_active = datetime.now() - timedelta(seconds=9)
        assert not session.is_expired()
        
        # Session with no timeout should not expire
        session.timeout = None
        session.last_active = datetime.now() - timedelta(days=30)
        assert not session.is_expired()


class TestSessionManager:
    """Tests for the SessionManager class."""

    def test_create_session(self):
        """Test that a session can be created and retrieved."""
        manager = SessionManager()
        
        # Create a session
        session_id = manager.create_session()
        
        # Verify session exists
        assert manager.session_exists(session_id)
        
        # Retrieve and check session
        session = manager.get_session(session_id)
        assert session.id == session_id
        assert session.user_id is None
        assert isinstance(session.metadata, dict)

    def test_create_session_with_user_id(self):
        """Test that a session can be created with a user ID."""
        manager = SessionManager()
        user_id = "user123"
        
        # Create a session with user ID
        session_id = manager.create_session(user_id=user_id)
        
        # Retrieve and check session
        session = manager.get_session(session_id)
        assert session.user_id == user_id

    def test_create_session_with_metadata(self):
        """Test that a session can be created with metadata."""
        manager = SessionManager()
        metadata = {"source": "web", "browser": "chrome"}
        
        # Create a session with metadata
        session_id = manager.create_session(metadata=metadata)
        
        # Retrieve and check session
        session = manager.get_session(session_id)
        assert session.metadata == metadata

    def test_get_nonexistent_session(self):
        """Test that getting a nonexistent session returns None."""
        manager = SessionManager()
        assert manager.get_session("nonexistent") is None

    def test_update_session(self):
        """Test that a session can be updated."""
        manager = SessionManager()
        
        # Create a session
        session_id = manager.create_session()
        
        # Update the session
        original_last_active = manager.get_session(session_id).last_active
        time.sleep(0.1)  # Ensure last_active will be different
        
        manager.update_session(session_id)
        
        # Verify session was updated
        updated_session = manager.get_session(session_id)
        assert updated_session.last_active > original_last_active

    def test_update_session_metadata(self):
        """Test that session metadata can be updated."""
        manager = SessionManager()
        
        # Create a session with initial metadata
        session_id = manager.create_session(metadata={"source": "web"})
        
        # Update metadata
        manager.update_session_metadata(session_id, {"page": "home"})
        
        # Verify metadata was updated and merged
        session = manager.get_session(session_id)
        assert session.metadata == {"source": "web", "page": "home"}
        
        # Replace metadata
        manager.update_session_metadata(session_id, {"theme": "dark"}, merge=False)
        
        # Verify metadata was replaced
        session = manager.get_session(session_id)
        assert session.metadata == {"theme": "dark"}

    def test_remove_session(self):
        """Test that a session can be removed."""
        manager = SessionManager()
        
        # Create a session
        session_id = manager.create_session()
        
        # Verify session exists
        assert manager.session_exists(session_id)
        
        # Remove the session
        manager.remove_session(session_id)
        
        # Verify session no longer exists
        assert not manager.session_exists(session_id)
        assert manager.get_session(session_id) is None

    def test_clear_expired_sessions(self):
        """Test that expired sessions are cleared."""
        manager = SessionManager()
        
        # Create sessions with short timeout
        session1 = manager.create_session(timeout=1)
        session2 = manager.create_session(timeout=60)
        
        # Wait for first session to expire
        time.sleep(1.1)
        
        # Clear expired sessions
        expired = manager.clear_expired_sessions()
        
        # Verify only first session was cleared
        assert len(expired) == 1
        assert expired[0] == session1
        assert not manager.session_exists(session1)
        assert manager.session_exists(session2)

    def test_get_all_sessions(self):
        """Test that all sessions can be retrieved."""
        manager = SessionManager()
        
        # Create sessions
        session1 = manager.create_session()
        session2 = manager.create_session()
        
        # Get all sessions
        sessions = manager.get_all_sessions()
        
        # Verify both sessions are retrieved
        assert len(sessions) == 2
        session_ids = [s.id for s in sessions]
        assert session1 in session_ids
        assert session2 in session_ids

    def test_get_user_sessions(self):
        """Test that user sessions can be retrieved."""
        manager = SessionManager()
        
        # Create sessions for different users
        user1_session1 = manager.create_session(user_id="user1")
        user1_session2 = manager.create_session(user_id="user1")
        user2_session = manager.create_session(user_id="user2")
        
        # Get user1 sessions
        user1_sessions = manager.get_user_sessions("user1")
        
        # Verify user1 sessions are retrieved
        assert len(user1_sessions) == 2
        session_ids = [s.id for s in user1_sessions]
        assert user1_session1 in session_ids
        assert user1_session2 in session_ids
        assert user2_session not in session_ids


class TestSynchronizedSessionManager:
    """Tests for the SynchronizedSessionManager class."""

    def test_thread_safety(self):
        """Test that the synchronized manager is thread-safe."""
        manager = SynchronizedSessionManager()
        
        # Number of sessions to create per thread
        sessions_per_thread = 50
        num_threads = 5
        
        # List to store all created session IDs
        all_session_ids = []
        
        # Create a lock to protect the list
        lock = threading.Lock()
        
        def create_sessions():
            """Create sessions in a thread."""
            thread_session_ids = []
            for _ in range(sessions_per_thread):
                session_id = manager.create_session()
                thread_session_ids.append(session_id)
            
            # Add thread's session IDs to the main list
            with lock:
                all_session_ids.extend(thread_session_ids)
        
        # Create and start threads
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=create_sessions)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify the correct number of sessions were created
        assert len(all_session_ids) == sessions_per_thread * num_threads
        
        # Verify each session ID is unique
        assert len(set(all_session_ids)) == len(all_session_ids)
        
        # Verify all sessions exist in the manager
        for session_id in all_session_ids:
            assert manager.session_exists(session_id)


@pytest.mark.asyncio
class TestAsyncSessionManager:
    """Tests for the AsyncSessionManager class."""

    async def test_create_and_get_session(self):
        """Test that a session can be created and retrieved asynchronously."""
        manager = AsyncSessionManager()
        
        # Create a session
        session_id = await manager.create_session()
        
        # Verify session exists
        assert await manager.session_exists(session_id)
        
        # Retrieve and check session
        session = await manager.get_session(session_id)
        assert session.id == session_id
        assert session.user_id is None
        assert isinstance(session.metadata, dict)

    async def test_update_and_remove_session(self):
        """Test that a session can be updated and removed asynchronously."""
        manager = AsyncSessionManager()
        
        # Create a session
        session_id = await manager.create_session()
        
        # Update the session
        original_session = await manager.get_session(session_id)
        original_last_active = original_session.last_active
        
        await manager.update_session(session_id)
        
        # Verify session was updated
        updated_session = await manager.get_session(session_id)
        assert updated_session.last_active > original_last_active
        
        # Remove the session
        await manager.remove_session(session_id)
        
        # Verify session no longer exists
        assert not await manager.session_exists(session_id)
        assert await manager.get_session(session_id) is None


class TestDatabaseSessionManager:
    """Tests for the DatabaseSessionManager class."""

    @patch("lg_adk.sessions.session_manager.DatabaseManager")
    def test_database_integration(self, mock_db_manager):
        """Test that the database manager correctly interacts with a database."""
        # Create a mock database manager
        mock_db = MagicMock()
        mock_db_manager.return_value = mock_db
        
        # Set up mock session data
        test_session = Session(id="test-session", user_id="test-user")
        mock_db.retrieve.return_value = {
            "id": test_session.id,
            "user_id": test_session.user_id,
            "created_at": test_session.created_at.isoformat(),
            "last_active": test_session.last_active.isoformat(),
            "metadata": test_session.metadata,
            "timeout": test_session.timeout,
        }
        
        # Create database session manager
        manager = DatabaseSessionManager(db_url="sqlite:///:memory:")
        
        # Test session creation
        session_id = manager.create_session(user_id="test-user")
        mock_db.store.assert_called_once()
        
        # Test session retrieval
        session = manager.get_session(session_id)
        mock_db.retrieve.assert_called_once()
        assert session.id == test_session.id
        assert session.user_id == test_session.user_id
        
        # Test session update
        manager.update_session(session_id)
        assert mock_db.update.call_count == 1
        
        # Test session removal
        manager.remove_session(session_id)
        assert mock_db.delete.call_count == 1


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 