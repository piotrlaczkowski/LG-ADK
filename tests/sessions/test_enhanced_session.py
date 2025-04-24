"""
Tests for enhanced session management functionality.

This module tests the EnhancedSessionManager and related functionality
in the lg_adk.sessions module.
"""

import pytest
from datetime import datetime, timedelta
import time
from typing import Dict, Any, Optional

from lg_adk.sessions.session_manager import (
    EnhancedSessionManager,
    Session,
)


@pytest.fixture
def enhanced_session_manager():
    """Return an enhanced session manager for testing."""
    return EnhancedSessionManager()


class TestEnhancedSessionManager:
    """Test the EnhancedSessionManager class."""
    
    def test_create_session_with_user_tracking(self, enhanced_session_manager):
        """Test session creation with user tracking."""
        # Create session for user
        user_id = "test-user"
        session_id = enhanced_session_manager.create_session(user_id=user_id)
        
        # Verify session was created
        assert enhanced_session_manager.session_exists(session_id)
        
        # Verify user is tracked
        user_sessions = enhanced_session_manager.get_user_sessions(user_id)
        assert len(user_sessions) == 1
        assert session_id in user_sessions
        
        # Create another session for same user
        session_id2 = enhanced_session_manager.create_session(user_id=user_id)
        
        # Verify both sessions are tracked
        user_sessions = enhanced_session_manager.get_user_sessions(user_id)
        assert len(user_sessions) == 2
        assert session_id in user_sessions
        assert session_id2 in user_sessions
    
    def test_session_metadata_management(self, enhanced_session_manager):
        """Test session metadata management."""
        # Create session with metadata
        initial_metadata = {"user_name": "Test User", "preferences": {"theme": "dark"}}
        session_id = enhanced_session_manager.create_session(metadata=initial_metadata)
        
        # Verify metadata was stored
        session = enhanced_session_manager.get_session(session_id)
        assert session.metadata == initial_metadata
        
        # Update metadata
        updated_metadata = {"user_name": "Test User", "preferences": {"theme": "light"}}
        enhanced_session_manager.update_session_metadata(session_id, updated_metadata)
        
        # Verify metadata was updated
        session = enhanced_session_manager.get_session(session_id)
        assert session.metadata == updated_metadata
        
        # Update partial metadata
        enhanced_session_manager.update_session_metadata(
            session_id, 
            {"status": "active"}
        )
        
        # Verify metadata was merged
        session = enhanced_session_manager.get_session(session_id)
        expected = {
            "user_name": "Test User",
            "preferences": {"theme": "light"},
            "status": "active"
        }
        assert session.metadata == expected
    
    def test_analytics_tracking(self, enhanced_session_manager):
        """Test analytics tracking for sessions."""
        # Create session
        session_id = enhanced_session_manager.create_session()
        
        # Track interactions
        enhanced_session_manager.track_interaction(
            session_id,
            "message",
            {"role": "user", "content": "Hello"}
        )
        
        enhanced_session_manager.track_interaction(
            session_id,
            "response",
            {"role": "assistant", "content": "Hi there!"}
        )
        
        enhanced_session_manager.track_interaction(
            session_id,
            "tool_use",
            {"tool": "weather", "params": {"location": "New York"}}
        )
        
        # Get analytics
        analytics = enhanced_session_manager.get_session_analytics(session_id)
        
        # Check basic counts
        assert analytics["message_count"] == 3
        assert len(analytics["interaction_history"]) == 3
        
        # Check interaction types
        interactions = analytics["interaction_history"]
        assert interactions[0]["type"] == "message"
        assert interactions[1]["type"] == "response"
        assert interactions[2]["type"] == "tool_use"
        
        # Check interaction details
        assert interactions[0]["details"]["role"] == "user"
        assert interactions[1]["details"]["role"] == "assistant"
        assert interactions[2]["details"]["tool"] == "weather"
    
    def test_session_duration(self, enhanced_session_manager):
        """Test session duration tracking."""
        # Create session
        session_id = enhanced_session_manager.create_session()
        
        # Wait a bit
        time.sleep(0.1)
        
        # Update session to simulate activity
        enhanced_session_manager.update_session(session_id)
        
        # Get duration
        analytics = enhanced_session_manager.get_session_analytics(session_id)
        duration = analytics.get("duration_seconds", 0)
        
        # Duration should be positive but less than 1 second
        assert duration > 0
        assert duration < 1
    
    def test_response_metrics(self, enhanced_session_manager):
        """Test response time metrics tracking."""
        # Create session
        session_id = enhanced_session_manager.create_session()
        
        # Track user message with timestamp
        enhanced_session_manager.track_interaction(
            session_id,
            "message",
            {"role": "user", "content": "What's the weather?"}
        )
        
        # Wait to simulate processing time
        time.sleep(0.1)
        
        # Track response with metrics
        enhanced_session_manager.track_interaction(
            session_id,
            "response",
            {
                "role": "assistant", 
                "content": "It's sunny!",
                "metrics": {
                    "tokens": 10,
                    "processing_time": 0.05
                }
            }
        )
        
        # Get analytics
        analytics = enhanced_session_manager.get_session_analytics(session_id)
        
        # Check response metrics
        assert "response_metrics" in analytics
        metrics = analytics["response_metrics"]
        
        # There should be average metrics
        assert "avg_response_time" in metrics
        assert metrics["avg_response_time"] > 0
        
        assert "total_tokens" in metrics
        assert metrics["total_tokens"] == 10
    
    def test_clear_expired_sessions(self, enhanced_session_manager):
        """Test clearing expired sessions."""
        # Create sessions with short timeout
        session_1 = enhanced_session_manager.create_session(
            timeout=0.1  # 100ms timeout
        )
        
        session_2 = enhanced_session_manager.create_session(
            timeout=0.1  # 100ms timeout
        )
        
        # Create session with longer timeout
        session_3 = enhanced_session_manager.create_session(
            timeout=10  # 10s timeout
        )
        
        # Wait for first two sessions to expire
        time.sleep(0.2)
        
        # Clear expired sessions
        num_cleared = enhanced_session_manager.cleanup_expired_sessions()
        
        # Should have cleared 2 sessions
        assert num_cleared == 2
        
        # First two sessions should be gone, third should remain
        assert not enhanced_session_manager.session_exists(session_1)
        assert not enhanced_session_manager.session_exists(session_2)
        assert enhanced_session_manager.session_exists(session_3)
    
    def test_end_session(self, enhanced_session_manager):
        """Test ending a session."""
        # Create session
        session_id = enhanced_session_manager.create_session()
        
        # Confirm session exists
        assert enhanced_session_manager.session_exists(session_id)
        
        # Track some interactions
        enhanced_session_manager.track_interaction(
            session_id,
            "message",
            {"content": "Test message"}
        )
        
        # End session
        success = enhanced_session_manager.end_session(session_id)
        assert success
        
        # Session should be removed
        assert not enhanced_session_manager.session_exists(session_id)
        
        # Analytics should still be available
        analytics = enhanced_session_manager.get_session_analytics(session_id)
        assert analytics is not None
        assert "end_time" in analytics
        assert analytics["status"] == "ended" 