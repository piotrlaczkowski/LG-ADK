"""Tests for the SessionManager class."""

import threading
import time
import unittest
import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from lg_adk.sessions.session_manager import (
    AsyncSessionManager,
    DatabaseSessionManager,
    Session,
    SessionManager,
    SynchronizedSessionManager,
)


def test_session_creation() -> None:
    """Test that a session can be created with the expected attributes."""
    session_id = str(uuid.uuid4())
    user_id = "user123"
    metadata = {"source": "web"}
    session = Session(session_id=session_id, user_id=user_id, metadata=metadata)
    assert session.id == session_id
    assert session.user_id == user_id
    assert session.metadata == metadata
    assert session.created_at is not None
    assert session.last_active is not None
    assert session.timeout is None


def test_session_is_expired() -> None:
    """Test that a session correctly reports when it is expired."""
    session = Session(session_id="test", timeout=timedelta(seconds=10))
    assert not session.is_expired()
    session.last_active = datetime.now() - timedelta(seconds=11)
    assert session.is_expired()
    session.last_active = datetime.now() - timedelta(seconds=9)
    assert not session.is_expired()
    session.timeout = None
    session.last_active = datetime.now() - timedelta(days=30)
    assert not session.is_expired()


def test_session_to_dict_and_from_dict() -> None:
    """Test converting a session to and from a dictionary."""
    original_session = Session(session_id="test-id", user_id="test-user", metadata={"test": "value"})
    session_dict = original_session.to_dict()
    recreated_session = Session.from_dict(session_dict)
    assert recreated_session.id == original_session.id
    assert recreated_session.user_id == original_session.user_id
    assert recreated_session.metadata == original_session.metadata


def test_create_session() -> None:
    """Test that a session can be created and retrieved."""
    manager = SessionManager()
    session_id = manager.create_session()
    assert manager.session_exists(session_id)
    session = manager.get_session(session_id)
    assert session.id == session_id
    assert session.user_id is None
    assert isinstance(session.metadata, dict)


def test_create_session_with_id() -> None:
    """Test that a session can be created with a specific ID."""
    manager = SessionManager()
    custom_id = str(uuid.uuid4())
    manager.create_session_with_id(custom_id)
    assert manager.session_exists(custom_id)
    session = manager.get_session(custom_id)
    assert session.id == custom_id


def test_create_session_with_user_id() -> None:
    """Test that a session can be created with a user ID."""
    manager = SessionManager()
    user_id = "user123"
    session_id = manager.create_session(user_id=user_id)
    session = manager.get_session(session_id)
    assert session.user_id == user_id


def test_create_session_with_metadata() -> None:
    """Test that a session can be created with metadata."""
    manager = SessionManager()
    metadata = {"source": "web", "browser": "chrome"}
    session_id = manager.create_session(metadata=metadata)
    session = manager.get_session(session_id)
    assert session.metadata == metadata


def test_get_session() -> None:
    """Test getting a session by ID."""
    manager = SessionManager()
    session_id = manager.create_session()
    session = manager.get_session(session_id)
    assert session.id == session_id


def test_get_nonexistent_session() -> None:
    """Test that getting a nonexistent session returns None."""
    manager = SessionManager()
    assert manager.get_session("nonexistent") is None


def test_update_session() -> None:
    """Test that a session can be updated."""
    manager = SessionManager()
    session_id = manager.create_session()
    original_last_active = manager.get_session(session_id).last_active
    time.sleep(0.1)
    manager.update_session(session_id)
    updated_session = manager.get_session(session_id)
    assert updated_session.last_active > original_last_active


def test_remove_session() -> None:
    """Test that a session can be removed."""
    manager = SessionManager()
    session_id = manager.create_session()
    assert manager.session_exists(session_id)
    manager.remove_session(session_id)
    assert not manager.session_exists(session_id)
    assert manager.get_session(session_id) is None


@pytest.mark.xfail(reason="merge=False parameter not correctly implemented")
def test_update_session_metadata() -> None:
    """Test that session metadata can be updated."""
    manager = SessionManager()
    session_id = manager.create_session(metadata={"source": "web"})

    # Test merging behavior (default)
    manager.update_session_metadata(session_id, {"page": "home"})
    session = manager.get_session(session_id)
    assert session.metadata == {"source": "web", "page": "home"}

    # Test that merge=False parameter should replace instead of merge
    # but current implementation doesn't respect this parameter
    manager.update_session_metadata(session_id, {"theme": "dark"}, merge=False)
    session = manager.get_session(session_id)

    # This assertion would pass if merge=False worked correctly
    assert session.metadata == {"theme": "dark"}


@pytest.mark.xfail(reason="Timeout and expiration needs more setup")
def test_clear_expired_sessions() -> None:
    """Test that expired sessions are cleared."""
    manager = SessionManager()
    session1 = manager.create_session(timeout=1)
    session2 = manager.create_session(timeout=60)
    time.sleep(1.1)

    # Force expiration for test purposes
    session = manager.get_session(session1)
    if session:
        session.last_active = datetime.now() - timedelta(seconds=10)

    # Now run cleanup
    expired = manager.cleanup_expired_sessions()
    assert expired >= 1
    assert not manager.session_exists(session1)
    assert manager.session_exists(session2)


def test_get_all_sessions() -> None:
    """Test that all sessions can be retrieved."""
    manager = SessionManager()
    session1 = manager.create_session()
    session2 = manager.create_session()
    sessions = manager.get_all_sessions()
    assert len(sessions) == 2
    session_ids = [s.id for s in sessions]
    assert session1 in session_ids
    assert session2 in session_ids


def test_get_user_sessions() -> None:
    """Test that user sessions can be retrieved."""
    manager = SessionManager()
    user1_session1 = manager.create_session(user_id="user1")
    user1_session2 = manager.create_session(user_id="user1")
    user2_session = manager.create_session(user_id="user2")
    # The get_user_sessions method returns session IDs, not Session objects
    user1_sessions = manager.get_user_sessions("user1")
    assert len(user1_sessions) == 2
    assert user1_session1 in user1_sessions
    assert user1_session2 in user1_sessions
    assert user2_session not in user1_sessions


def test_session_analytics() -> None:
    """Test that session analytics are tracked."""
    manager = SessionManager()
    session_id = manager.create_session()

    # Track an interaction
    manager.track_interaction(session_id, "test_interaction", {"detail": "value"})

    # Get analytics
    analytics = manager.get_session_analytics(session_id)
    assert analytics is not None
    assert "interaction_history" in analytics


def test_synchronized_basic_functionality() -> None:
    """Test that the synchronized manager has basic functionality."""
    manager = SynchronizedSessionManager()
    session_id = manager.create_session()
    assert manager.session_exists(session_id)


def test_synchronized_thread_safety() -> None:
    """Test that the synchronized manager is thread-safe."""
    manager = SynchronizedSessionManager()
    sessions_per_thread = 50
    num_threads = 5
    all_session_ids = []
    lock = threading.Lock()

    def create_sessions() -> None:
        """Create sessions in a thread."""
        thread_session_ids = []
        for _ in range(sessions_per_thread):
            session_id = manager.create_session()
            thread_session_ids.append(session_id)
        with lock:
            all_session_ids.extend(thread_session_ids)

    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=create_sessions)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    assert len(all_session_ids) == sessions_per_thread * num_threads
    assert len(set(all_session_ids)) == len(all_session_ids)
    for session_id in all_session_ids:
        assert manager.session_exists(session_id)


@pytest.mark.asyncio
@pytest.mark.skip(reason="Async tests are not implemented")
class TestAsyncSessionManager:
    """Tests for the AsyncSessionManager class."""

    async def test_create_session(self) -> None:
        """Test creating a session asynchronously."""
        manager = AsyncSessionManager()
        session_id = await manager.create_session()
        assert session_id is not None


@pytest.mark.skip(reason="Mock needs to be updated")
def test_database_initialization() -> None:
    """Test database session manager initialization."""
    with patch("lg_adk.database.database_manager.DatabaseManager") as mock_db_manager:
        mock_db = MagicMock()
        manager = DatabaseSessionManager(database_manager=mock_db)
        assert manager.db == mock_db


@pytest.mark.skip(reason="Mock needs to be updated")
def test_database_register_session() -> None:
    """Test registering a session with the database."""
    with patch("lg_adk.database.database_manager.DatabaseManager") as mock_db_manager:
        mock_db = MagicMock()
        manager = DatabaseSessionManager(database_manager=mock_db)
        session_id = "test-session"
        user_id = "test-user"
        metadata = {"test": "value"}
        manager.register_session(session_id, user_id=user_id, metadata=metadata)
        mock_db.put.assert_called()


if __name__ == "__main__":
    pytest.main()
