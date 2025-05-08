"""Tests for asynchronous session management functionality.

This module tests the AsyncSessionManager class which provides
asynchronous session management capabilities.
"""

import asyncio
import unittest
from typing import Any

import pytest

from lg_adk.sessions.session_manager import AsyncSessionManager


@pytest.fixture
def async_session_manager() -> AsyncSessionManager:
    """Return an async session manager for testing."""
    return AsyncSessionManager()


# Helper for running async tests
async def create_session_async(manager: AsyncSessionManager, metadata=None) -> str:
    """Create a session asynchronously."""
    return await manager.acreate_session(metadata=metadata)


async def get_session_async(manager: AsyncSessionManager, session_id: str) -> Any:
    """Get a session asynchronously."""
    return await manager.aget_session(session_id)


async def update_session_async(manager: AsyncSessionManager, session_id: str) -> None:
    """Update a session asynchronously."""
    return await manager.aupdate_session(session_id)


async def update_metadata_async(manager: AsyncSessionManager, session_id: str, metadata: dict) -> None:
    """Update session metadata asynchronously."""
    return await manager.aupdate_session_metadata(session_id, metadata)


async def end_session_async(manager: AsyncSessionManager, session_id: str) -> bool:
    """End a session asynchronously."""
    return await manager.aend_session(session_id)


async def cleanup_expired_sessions_async(manager: AsyncSessionManager) -> int:
    """Clean up expired sessions asynchronously."""
    return await manager.acleanup_expired_sessions()


class TestAsyncSessionManager(unittest.TestCase):
    """Test the AsyncSessionManager class."""

    @pytest.mark.asyncio
    async def test_basic_functionality(self, async_session_manager) -> None:
        """Test that basic async session management works."""
        # Create session
        session_id = await create_session_async(async_session_manager)

        # Verify session was created
        session = await get_session_async(async_session_manager, session_id)
        self.assertEqual(session.session_id, session_id)

        # Update session
        await update_session_async(async_session_manager, session_id)

        # End session
        success = await end_session_async(async_session_manager, session_id)
        self.assertTrue(success)

        # Verify session was removed
        with self.assertRaises(KeyError):
            await get_session_async(async_session_manager, session_id)

    @pytest.mark.asyncio
    async def test_metadata_management(self, async_session_manager) -> None:
        """Test async metadata management."""
        # Create session with initial metadata
        initial_metadata = {"user": "test_user", "topic": "test_topic"}
        session_id = await create_session_async(
            async_session_manager,
            metadata=initial_metadata,
        )

        # Verify initial metadata
        session = await get_session_async(async_session_manager, session_id)
        self.assertEqual(session.metadata, initial_metadata)

        # Update metadata
        update_metadata = {"status": "active", "priority": "high"}
        await update_metadata_async(
            async_session_manager,
            session_id,
            update_metadata,
        )

        # Verify updated metadata (should be merged)
        updated_session = await get_session_async(async_session_manager, session_id)
        expected_metadata = {**initial_metadata, **update_metadata}
        self.assertEqual(updated_session.metadata, expected_metadata)

    @pytest.mark.asyncio
    async def test_session_timeout(self, async_session_manager) -> None:
        """Test that sessions expire after timeout."""
        # Create session with short timeout
        session_id = await async_session_manager.acreate_session(timeout=0.1)

        # Verify session exists initially
        session = await get_session_async(async_session_manager, session_id)
        self.assertEqual(session.session_id, session_id)

        # Wait for session to expire
        await asyncio.sleep(0.2)

        # Clean up expired sessions
        cleaned_count = await cleanup_expired_sessions_async(async_session_manager)
        self.assertEqual(cleaned_count, 1)

        # Verify session was removed
        with pytest.raises(KeyError):
            await get_session_async(async_session_manager, session_id)

    @pytest.mark.asyncio
    async def test_concurrent_session_operations(self, async_session_manager) -> None:
        """Test performing multiple async operations concurrently."""
        # Create multiple sessions concurrently
        num_sessions = 10
        create_tasks = [create_session_async(async_session_manager) for _ in range(num_sessions)]
        session_ids = await asyncio.gather(*create_tasks)

        # Verify all session IDs are unique
        self.assertEqual(len(set(session_ids)), num_sessions)

        # Update all sessions concurrently
        update_tasks = [update_session_async(async_session_manager, session_id) for session_id in session_ids]
        await asyncio.gather(*update_tasks)

        # Get all sessions concurrently
        get_tasks = [get_session_async(async_session_manager, session_id) for session_id in session_ids]
        sessions = await asyncio.gather(*get_tasks)

        # Verify all sessions were retrieved
        self.assertEqual(len(sessions), num_sessions)
        for i, session in enumerate(sessions):
            self.assertEqual(session.session_id, session_ids[i])

        # End all sessions concurrently
        end_tasks = [end_session_async(async_session_manager, session_id) for session_id in session_ids]
        end_results = await asyncio.gather(*end_tasks)

        # Verify all sessions were ended
        self.assertTrue(all(end_results))

    @pytest.mark.asyncio
    async def test_async_metadata_updates(self, async_session_manager) -> None:
        """Test async metadata updates with sequential increments."""
        # Create session with initial counter
        session_id = await create_session_async(
            async_session_manager,
            metadata={"counter": 0},
        )

        # Update counter multiple times
        num_updates = 10
        for _i in range(1, num_updates + 1):
            # Get current session
            session = await get_session_async(async_session_manager, session_id)
            current_counter = session.metadata.get("counter", 0)

            # Update counter
            await update_metadata_async(
                async_session_manager,
                session_id,
                {"counter": current_counter + 1},
            )

        # Verify final counter value
        final_session = await get_session_async(async_session_manager, session_id)
        self.assertEqual(final_session.metadata["counter"], num_updates)

    @pytest.mark.asyncio
    async def test_parallel_metadata_updates(self, async_session_manager) -> None:
        """Test parallel async metadata updates."""
        # Create session with initial metadata
        session_id = await create_session_async(
            async_session_manager,
            metadata={"tags": []},
        )

        # Update tags concurrently
        num_updates = 10

        async def add_tag(tag_num) -> None:
            # Get current session
            session = await get_session_async(async_session_manager, session_id)
            current_tags = session.metadata.get("tags", [])

            # Add a new tag
            new_tag = f"tag_{tag_num}"
            new_tags = current_tags + [new_tag]

            # Update metadata
            await update_metadata_async(
                async_session_manager,
                session_id,
                {"tags": new_tags},
            )

            return new_tag

        # Run updates in parallel
        update_tasks = [add_tag(i) for i in range(num_updates)]
        added_tags = await asyncio.gather(*update_tasks)

        # Verify final tags
        final_session = await get_session_async(async_session_manager, session_id)
        self.assertEqual(len(final_session.metadata["tags"]), num_updates)

        # All tags should be in the final list (order may vary)
        for tag in added_tags:
            self.assertIn(tag, final_session.metadata["tags"])

    @pytest.mark.asyncio
    async def test_session_creation_with_user_id(self, async_session_manager) -> None:
        """Test creating sessions with user IDs."""
        # Create multiple sessions for the same user
        user_id = "test_user_123"
        num_sessions = 5

        session_ids = []
        for i in range(num_sessions):
            session_id = await async_session_manager.acreate_session(
                user_id=user_id,
                metadata={"session_number": i},
            )
            session_ids.append(session_id)

        # Verify all sessions were created
        for session_id in session_ids:
            session = await get_session_async(async_session_manager, session_id)
            self.assertEqual(session.user_id, user_id)

    @pytest.mark.asyncio
    async def test_async_session_interactions(self, async_session_manager) -> None:
        """Test tracking async session interactions."""
        # Create a session
        session_id = await create_session_async(async_session_manager)

        # Track some interactions
        num_interactions = 5
        for i in range(num_interactions):
            await async_session_manager.atrack_interaction(
                session_id,
                tokens_in=10 * (i + 1),
                tokens_out=5 * (i + 1),
                response_time=0.1 * (i + 1),
            )
            # Add a small delay to ensure timestamps are different
            await asyncio.sleep(0.01)

        # Get session
        session = await get_session_async(async_session_manager, session_id)

        # Verify interactions were tracked
        self.assertEqual(session.interactions, num_interactions)
        self.assertEqual(session.total_tokens_in, sum(10 * (i + 1) for i in range(num_interactions)))
        self.assertEqual(session.total_tokens_out, sum(5 * (i + 1) for i in range(num_interactions)))

        # Verify timestamps
        self.assertTrue(session.created_at < session.last_interaction_at)

    @pytest.mark.asyncio
    async def test_end_nonexistent_session(self, async_session_manager) -> None:
        """Test ending a session that doesn't exist."""
        # Try to end a nonexistent session
        success = await end_session_async(async_session_manager, "nonexistent_session_id")

        # Should return False, not raise an exception
        self.assertFalse(success)

    @pytest.mark.asyncio
    async def test_cleanup_with_no_expired_sessions(self, async_session_manager) -> None:
        """Test cleaning up when there are no expired sessions."""
        # Create a session with a long timeout
        session_id = await create_session_async(async_session_manager, metadata={"timeout": 3600})

        # Clean up expired sessions
        cleaned_count = await cleanup_expired_sessions_async(async_session_manager)

        # Should be 0 since the session hasn't expired
        self.assertEqual(cleaned_count, 0)

        # Session should still exist
        session = await get_session_async(async_session_manager, session_id)
        self.assertEqual(session.session_id, session_id)
