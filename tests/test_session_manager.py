"""Tests for the SessionManager class."""

import threading
import time
import unittest
import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from lg_adk.sessions.session_manager import (
    AsyncSessionManager,
    DatabaseSessionManager,
    Session,
    SessionManager,
    SynchronizedSessionManager,
)


class TestSession(unittest.TestCase):
    """Tests for the Session class."""

    def test_session_creation(self) -> None:
        """Test that a session can be created with the expected attributes."""
        session_id = str(uuid.uuid4())
        user_id = "user123"
        metadata = {"source": "web"}
        session = Session(id=session_id, user_id=user_id, metadata=metadata)
        self.assertEqual(session.id, session_id)
        self.assertEqual(session.user_id, user_id)
        self.assertEqual(session.metadata, metadata)
        self.assertIsNotNone(session.created_at)
        self.assertIsNotNone(session.last_active)
        self.assertEqual(session.timeout, 3600)

    def test_session_is_expired(self) -> None:
        """Test that a session correctly reports when it is expired."""
        session = Session(id="test", timeout=10)
        self.assertFalse(session.is_expired())
        session.last_active = datetime.now() - timedelta(seconds=11)
        self.assertTrue(session.is_expired())
        session.last_active = datetime.now() - timedelta(seconds=9)
        self.assertFalse(session.is_expired())
        session.timeout = None
        session.last_active = datetime.now() - timedelta(days=30)
        self.assertFalse(session.is_expired())


class TestSessionManager(unittest.TestCase):
    """Tests for the SessionManager class."""

    def test_create_session(self) -> None:
        """Test that a session can be created and retrieved."""
        manager = SessionManager()
        session_id = manager.create_session()
        self.assertTrue(manager.session_exists(session_id))
        session = manager.get_session(session_id)
        self.assertEqual(session.id, session_id)
        self.assertIsNone(session.user_id)
        self.assertIsInstance(session.metadata, dict)

    def test_create_session_with_user_id(self) -> None:
        """Test that a session can be created with a user ID."""
        manager = SessionManager()
        user_id = "user123"
        session_id = manager.create_session(user_id=user_id)
        session = manager.get_session(session_id)
        self.assertEqual(session.user_id, user_id)

    def test_create_session_with_metadata(self) -> None:
        """Test that a session can be created with metadata."""
        manager = SessionManager()
        metadata = {"source": "web", "browser": "chrome"}
        session_id = manager.create_session(metadata=metadata)
        session = manager.get_session(session_id)
        self.assertEqual(session.metadata, metadata)

    def test_get_nonexistent_session(self) -> None:
        """Test that getting a nonexistent session returns None."""
        manager = SessionManager()
        self.assertIsNone(manager.get_session("nonexistent"))

    def test_update_session(self) -> None:
        """Test that a session can be updated."""
        manager = SessionManager()
        session_id = manager.create_session()
        original_last_active = manager.get_session(session_id).last_active
        time.sleep(0.1)
        manager.update_session(session_id)
        updated_session = manager.get_session(session_id)
        self.assertGreater(updated_session.last_active, original_last_active)

    def test_update_session_metadata(self) -> None:
        """Test that session metadata can be updated."""
        manager = SessionManager()
        session_id = manager.create_session(metadata={"source": "web"})
        manager.update_session_metadata(session_id, {"page": "home"})
        session = manager.get_session(session_id)
        self.assertEqual(session.metadata, {"source": "web", "page": "home"})
        manager.update_session_metadata(session_id, {"theme": "dark"}, merge=False)
        session = manager.get_session(session_id)
        self.assertEqual(session.metadata, {"theme": "dark"})

    def test_remove_session(self) -> None:
        """Test that a session can be removed."""
        manager = SessionManager()
        session_id = manager.create_session()
        self.assertTrue(manager.session_exists(session_id))
        manager.remove_session(session_id)
        self.assertFalse(manager.session_exists(session_id))
        self.assertIsNone(manager.get_session(session_id))

    def test_clear_expired_sessions(self) -> None:
        """Test that expired sessions are cleared."""
        manager = SessionManager()
        session1 = manager.create_session(timeout=1)
        session2 = manager.create_session(timeout=60)
        time.sleep(1.1)
        expired = manager.clear_expired_sessions()
        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0], session1)
        self.assertFalse(manager.session_exists(session1))
        self.assertTrue(manager.session_exists(session2))

    def test_get_all_sessions(self) -> None:
        """Test that all sessions can be retrieved."""
        manager = SessionManager()
        session1 = manager.create_session()
        session2 = manager.create_session()
        sessions = manager.get_all_sessions()
        self.assertEqual(len(sessions), 2)
        session_ids = [s.id for s in sessions]
        self.assertIn(session1, session_ids)
        self.assertIn(session2, session_ids)

    def test_get_user_sessions(self) -> None:
        """Test that user sessions can be retrieved."""
        manager = SessionManager()
        user1_session1 = manager.create_session(user_id="user1")
        user1_session2 = manager.create_session(user_id="user1")
        user2_session = manager.create_session(user_id="user2")
        user1_sessions = manager.get_user_sessions("user1")
        self.assertEqual(len(user1_sessions), 2)
        session_ids = [s.id for s in user1_sessions]
        self.assertIn(user1_session1, session_ids)
        self.assertIn(user1_session2, session_ids)
        self.assertNotIn(user2_session, session_ids)


class TestSynchronizedSessionManager(unittest.TestCase):
    """Tests for the SynchronizedSessionManager class."""

    def test_thread_safety(self) -> None:
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
        self.assertEqual(len(all_session_ids), sessions_per_thread * num_threads)
        self.assertEqual(len(set(all_session_ids)), len(all_session_ids))
        for session_id in all_session_ids:
            self.assertTrue(manager.session_exists(session_id))


@unittest.skip("Async tests are not implemented")
class TestAsyncSessionManager(unittest.TestCase):
    """Tests for the AsyncSessionManager class."""

    async def test_create_and_get_session(self) -> None:
        """Test that a session can be created and retrieved asynchronously."""
        manager = AsyncSessionManager()
        session_id = await manager.create_session()
        self.assertTrue(await manager.session_exists(session_id))
        session = await manager.get_session(session_id)
        self.assertEqual(session.id, session_id)
        self.assertIsNone(session.user_id)
        self.assertIsInstance(session.metadata, dict)

    async def test_update_and_remove_session(self) -> None:
        """Test that a session can be updated and removed asynchronously."""
        manager = AsyncSessionManager()
        session_id = await manager.create_session()
        original_session = await manager.get_session(session_id)
        original_last_active = original_session.last_active
        await manager.update_session(session_id)
        updated_session = await manager.get_session(session_id)
        self.assertGreater(updated_session.last_active, original_last_active)
        await manager.remove_session(session_id)
        self.assertFalse(await manager.session_exists(session_id))
        self.assertIsNone(await manager.get_session(session_id))


class TestDatabaseSessionManager(unittest.TestCase):
    """Tests for the DatabaseSessionManager class."""

    @patch("lg_adk.sessions.session_manager.DatabaseManager")
    def test_database_integration(self, mock_db_manager) -> None:
        """Test that the database manager correctly interacts with a database."""
        mock_db = MagicMock()
        mock_db_manager.return_value = mock_db
        test_session = Session(id="test-session", user_id="test-user")
        mock_db.retrieve.return_value = {
            "id": test_session.id,
            "user_id": test_session.user_id,
            "created_at": test_session.created_at.isoformat(),
            "last_active": test_session.last_active.isoformat(),
            "metadata": test_session.metadata,
            "timeout": test_session.timeout,
        }
        manager = DatabaseSessionManager(db_url="sqlite:///:memory:")
        session_id = manager.create_session(user_id="test-user")
        mock_db.store.assert_called_once()
        session = manager.get_session(session_id)
        mock_db.retrieve.assert_called_once()
        self.assertEqual(session.id, test_session.id)
        self.assertEqual(session.user_id, test_session.user_id)
        manager.update_session(session_id)
        self.assertEqual(mock_db.update.call_count, 1)
        manager.remove_session(session_id)
        self.assertEqual(mock_db.delete.call_count, 1)


if __name__ == "__main__":
    unittest.main()
