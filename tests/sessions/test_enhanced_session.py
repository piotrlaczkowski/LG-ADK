"""Tests for enhanced session management functionality.

This module tests the EnhancedSessionManager and related functionality
in the lg_adk.sessions module.
"""

import time
import unittest

import pytest

from lg_adk.sessions.session_manager import EnhancedSessionManager


@pytest.fixture
def enhanced_session_manager() -> EnhancedSessionManager:
    """Return an enhanced session manager for testing."""
    return EnhancedSessionManager()


class TestEnhancedSessionManager(unittest.TestCase):
    """Test the EnhancedSessionManager class."""

    def test_create_session_with_user_tracking(self, enhanced_session_manager) -> None:
        """Test session creation with user tracking."""
        user_id = "test-user"
        session_id = enhanced_session_manager.create_session(user_id=user_id)
        self.assertTrue(enhanced_session_manager.session_exists(session_id))
        user_sessions = enhanced_session_manager.get_user_sessions(user_id)
        self.assertEqual(len(user_sessions), 1)
        self.assertIn(session_id, user_sessions)
        session_id2 = enhanced_session_manager.create_session(user_id=user_id)
        user_sessions = enhanced_session_manager.get_user_sessions(user_id)
        self.assertEqual(len(user_sessions), 2)
        self.assertIn(session_id, user_sessions)
        self.assertIn(session_id2, user_sessions)

    def test_session_metadata_management(self, enhanced_session_manager) -> None:
        """Test session metadata management."""
        initial_metadata = {"user_name": "Test User", "preferences": {"theme": "dark"}}
        session_id = enhanced_session_manager.create_session(metadata=initial_metadata)
        session = enhanced_session_manager.get_session(session_id)
        self.assertEqual(session.metadata, initial_metadata)
        updated_metadata = {"user_name": "Test User", "preferences": {"theme": "light"}}
        enhanced_session_manager.update_session_metadata(session_id, updated_metadata)
        session = enhanced_session_manager.get_session(session_id)
        self.assertEqual(session.metadata, updated_metadata)
        enhanced_session_manager.update_session_metadata(
            session_id,
            {"status": "active"},
        )
        session = enhanced_session_manager.get_session(session_id)
        expected = {
            "user_name": "Test User",
            "preferences": {"theme": "light"},
            "status": "active",
        }
        self.assertEqual(session.metadata, expected)

    def test_analytics_tracking(self, enhanced_session_manager) -> None:
        """Test analytics tracking for sessions."""
        session_id = enhanced_session_manager.create_session()
        enhanced_session_manager.track_interaction(
            session_id,
            "message",
            {"role": "user", "content": "Hello"},
        )
        enhanced_session_manager.track_interaction(
            session_id,
            "response",
            {"role": "assistant", "content": "Hi there!"},
        )
        enhanced_session_manager.track_interaction(
            session_id,
            "tool_use",
            {"tool": "weather", "params": {"location": "New York"}},
        )
        analytics = enhanced_session_manager.get_session_analytics(session_id)
        self.assertEqual(analytics["message_count"], 3)
        self.assertEqual(len(analytics["interaction_history"]), 3)
        interactions = analytics["interaction_history"]
        self.assertEqual(interactions[0]["type"], "message")
        self.assertEqual(interactions[1]["type"], "response")
        self.assertEqual(interactions[2]["type"], "tool_use")
        self.assertEqual(interactions[0]["details"]["role"], "user")
        self.assertEqual(interactions[1]["details"]["role"], "assistant")
        self.assertEqual(interactions[2]["details"]["tool"], "weather")

    def test_session_duration(self, enhanced_session_manager) -> None:
        """Test session duration tracking."""
        session_id = enhanced_session_manager.create_session()
        time.sleep(0.1)
        enhanced_session_manager.update_session(session_id)
        analytics = enhanced_session_manager.get_session_analytics(session_id)
        duration = analytics.get("duration_seconds", 0)
        self.assertGreater(duration, 0)
        self.assertLess(duration, 1)

    def test_response_metrics(self, enhanced_session_manager) -> None:
        """Test response time metrics tracking."""
        session_id = enhanced_session_manager.create_session()
        enhanced_session_manager.track_interaction(
            session_id,
            "message",
            {"role": "user", "content": "What's the weather?"},
        )
        time.sleep(0.1)
        enhanced_session_manager.track_interaction(
            session_id,
            "response",
            {
                "role": "assistant",
                "content": "It's sunny!",
                "metrics": {
                    "tokens": 10,
                    "processing_time": 0.05,
                },
            },
        )
        analytics = enhanced_session_manager.get_session_analytics(session_id)
        self.assertIn("response_metrics", analytics)
        metrics = analytics["response_metrics"]
        self.assertIn("avg_response_time", metrics)
        self.assertGreater(metrics["avg_response_time"], 0)
        self.assertIn("total_tokens", metrics)
        self.assertEqual(metrics["total_tokens"], 10)

    def test_clear_expired_sessions(self, enhanced_session_manager) -> None:
        """Test clearing expired sessions."""
        session_1 = enhanced_session_manager.create_session(
            timeout=0.1,  # 100ms timeout
        )
        session_2 = enhanced_session_manager.create_session(
            timeout=0.1,  # 100ms timeout
        )
        session_3 = enhanced_session_manager.create_session(
            timeout=10,  # 10s timeout
        )
        time.sleep(0.2)
        num_cleared = enhanced_session_manager.cleanup_expired_sessions()
        self.assertEqual(num_cleared, 2)
        self.assertFalse(enhanced_session_manager.session_exists(session_1))
        self.assertFalse(enhanced_session_manager.session_exists(session_2))
        self.assertTrue(enhanced_session_manager.session_exists(session_3))

    def test_end_session(self, enhanced_session_manager) -> None:
        """Test ending a session."""
        session_id = enhanced_session_manager.create_session()
        self.assertTrue(enhanced_session_manager.session_exists(session_id))
