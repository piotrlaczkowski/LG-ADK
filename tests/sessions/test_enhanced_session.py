"""Tests for enhanced session management functionality.

This module tests the EnhancedSessionManager and related functionality
in the lg_adk.sessions module.
"""

import time
from datetime import datetime

import pytest

from lg_adk.sessions.session_manager import SessionManager


@pytest.fixture
def enhanced_session_manager() -> SessionManager:
    """Return a session manager for testing."""
    return SessionManager()


class TestEnhancedSessionManager:
    """Test the EnhancedSessionManager class."""

    def test_create_session_with_user_tracking(self, enhanced_session_manager) -> None:
        """Test session creation with user tracking."""
        user_id = "test-user"
        session_id = enhanced_session_manager.create_session(user_id=user_id)
        assert enhanced_session_manager.session_exists(session_id)
        user_sessions = enhanced_session_manager.get_user_sessions(user_id)
        assert len(user_sessions) == 1
        assert session_id in user_sessions
        session_id2 = enhanced_session_manager.create_session(user_id=user_id)
        user_sessions = enhanced_session_manager.get_user_sessions(user_id)
        assert len(user_sessions) == 2
        assert session_id in user_sessions
        assert session_id2 in user_sessions

    def test_session_metadata_management(self, enhanced_session_manager) -> None:
        """Test session metadata management."""
        initial_metadata = {"user_name": "Test User", "preferences": {"theme": "dark"}}
        session_id = enhanced_session_manager.create_session(metadata=initial_metadata)
        session = enhanced_session_manager.get_session(session_id)
        assert session.metadata == initial_metadata
        updated_metadata = {"user_name": "Test User", "preferences": {"theme": "light"}}
        enhanced_session_manager.update_session_metadata(session_id, updated_metadata)
        session = enhanced_session_manager.get_session(session_id)
        assert session.metadata == updated_metadata
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
        assert session.metadata == expected

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
        assert analytics["message_count"] == 3
        assert len(analytics["interactions"]) == 3
        interactions = analytics["interactions"]
        assert interactions[0]["type"] == "message"
        assert interactions[1]["type"] == "response"
        assert interactions[2]["type"] == "tool_use"
        assert interactions[0]["details"]["role"] == "user"
        assert interactions[1]["details"]["role"] == "assistant"
        assert interactions[2]["details"]["tool"] == "weather"

    def test_session_duration(self, enhanced_session_manager) -> None:
        """Test session duration tracking."""
        session_id = enhanced_session_manager.create_session()
        time.sleep(0.1)
        enhanced_session_manager.update_session(session_id)
        analytics = enhanced_session_manager.get_session_analytics(session_id)
        assert "last_active" in analytics
        assert isinstance(analytics["last_active"], int | float | datetime)

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
        interactions = analytics.get("interactions", [])
        assert len(interactions) == 2
        response_interaction = interactions[1]
        assert response_interaction["type"] == "response"
        assert "metrics" in response_interaction["details"]
        assert response_interaction["details"]["metrics"]["tokens"] == 10
        assert response_interaction["details"]["metrics"]["processing_time"] == 0.05

    def test_clear_expired_sessions(self, enhanced_session_manager) -> None:
        """Test clearing expired sessions."""
        session_id = enhanced_session_manager.create_session()
        original_is_session_expired = enhanced_session_manager.is_session_expired

        def mock_is_session_expired(session_id_param):
            if session_id_param == session_id:
                return True
            return original_is_session_expired(session_id_param)

        enhanced_session_manager.is_session_expired = mock_is_session_expired

        expired_sessions = enhanced_session_manager.clear_expired_sessions()

        assert session_id in expired_sessions
        assert not enhanced_session_manager.session_exists(session_id)

        enhanced_session_manager.is_session_expired = original_is_session_expired

    def test_end_session(self, enhanced_session_manager) -> None:
        """Test ending a session."""
        session_id = enhanced_session_manager.create_session()
        assert enhanced_session_manager.session_exists(session_id)
