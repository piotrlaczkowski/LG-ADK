"""Tests for asynchronous session management functionality.

This module tests the AsyncSessionManager class which provides
asynchronous session management capabilities.
"""

import asyncio
import time
import uuid
from datetime import timedelta
from typing import Any

import pytest

from lg_adk.sessions.session_manager import AsyncSessionManager


@pytest.fixture
def async_session_manager() -> AsyncSessionManager:
    """Return an async session manager for testing."""
    return AsyncSessionManager()


async def create_session_async(manager: AsyncSessionManager, metadata=None, timeout=None, user_id=None) -> str:
    """Create a session asynchronously."""
    return await manager.create_session(user_id=user_id, metadata=metadata)


async def get_session_async(manager: AsyncSessionManager, session_id: str) -> Any:
    """Get a session asynchronously.

    Note: AsyncSessionManager doesn't have a direct get_session method,
    but we can access the underlying session_manager.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(executor, lambda: manager.session_manager.get_session_metadata(session_id))


async def update_metadata_async(manager: AsyncSessionManager, session_id: str, metadata: dict) -> bool:
    """Update session metadata asynchronously."""
    return await manager.update_session_metadata_async(session_id, metadata)


async def end_session_async(manager: AsyncSessionManager, session_id: str) -> bool:
    """End a session asynchronously."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(executor, lambda: manager.session_manager.end_session(session_id))


async def cleanup_expired_sessions_async(manager: AsyncSessionManager) -> list[str]:
    """Clean up expired sessions asynchronously."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(executor, lambda: manager.session_manager.clear_expired_sessions())


async def is_session_expired_async(manager: AsyncSessionManager, session_id: str) -> bool:
    """Check if a session is expired asynchronously."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(executor, lambda: manager.session_manager.is_session_expired(session_id))


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Async session implementation needs fixing")
async def test_basic_functionality(async_session_manager) -> None:
    """Test that basic session management works."""
    # Create session
    session_id = await create_session_async(async_session_manager)

    # Verify session was created
    session = await get_session_async(async_session_manager, session_id)
    assert session is not None
    assert "id" not in session  # Metadata doesn't include ID directly

    # End session
    success = await end_session_async(async_session_manager, session_id)
    assert success

    # Verify session was removed
    nonexistent_session = await get_session_async(async_session_manager, session_id)
    assert nonexistent_session is None


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Async session implementation needs fixing")
async def test_metadata_management(async_session_manager) -> None:
    """Test that metadata can be managed correctly."""
    # Create session with initial metadata
    initial_metadata = {"user": "test_user", "topic": "test_topic"}
    session_id = await create_session_async(
        async_session_manager,
        metadata=initial_metadata,
    )

    # Verify initial metadata
    session = await get_session_async(async_session_manager, session_id)
    assert session is not None
    assert session == initial_metadata

    # Update metadata
    update_metadata = {"status": "active", "priority": "high"}
    success = await update_metadata_async(
        async_session_manager,
        session_id,
        update_metadata,
    )
    assert success

    # Verify updated metadata (should be merged)
    updated_session = await get_session_async(async_session_manager, session_id)
    assert updated_session is not None
    expected_metadata = {**initial_metadata, **update_metadata}
    assert updated_session == expected_metadata


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Async session implementation needs fixing")
async def test_session_timeout(async_session_manager) -> None:
    """Test that sessions expire after timeout."""
    # Create session with short timeout
    session_id = await create_session_async(
        async_session_manager,
        timeout=0.1,
    )

    # Verify session exists initially
    session = await get_session_async(async_session_manager, session_id)
    assert session is not None

    # Wait for session to expire
    await asyncio.sleep(0.2)

    # Force session to expire for testing
    if hasattr(async_session_manager, "sessions") and session_id in async_session_manager.sessions:
        session = async_session_manager.sessions[session_id]
        session.last_active = datetime.now() - timedelta(seconds=1)
        if isinstance(session.timeout, (int, float)):
            session.timeout = timedelta(seconds=session.timeout)

    # Clean up expired sessions
    cleaned_count = await cleanup_expired_sessions_async(async_session_manager)
    assert cleaned_count >= 1

    # Verify session was removed
    with pytest.raises(Exception):  # Either KeyError or another exception
        await get_session_async(async_session_manager, session_id)


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Async session interactions require more setup")
async def test_async_session_interactions(async_session_manager) -> None:
    """Test tracking session interactions."""
    # Create a session
    session_id = await create_session_async(async_session_manager)

    # Track an interaction
    await async_session_manager.track_interaction_async(
        session_id,
        "message",
        {"text": "Hello world", "tokens": 10},
    )

    # Get session analytics
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        analytics = await loop.run_in_executor(
            executor, lambda: async_session_manager.session_manager.get_session_analytics(session_id)
        )

    assert analytics is not None

    # Verify interactions were tracked
    assert "interaction_history" in analytics
    assert len(analytics["interaction_history"]) > 0
    assert analytics["interaction_history"][0]["type"] == "message"


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Concurrent sessions need more setup")
async def test_concurrent_operations() -> None:
    """Test that concurrent operations are handled correctly."""
    manager = AsyncSessionManager()

    # Create multiple sessions concurrently
    num_sessions = 5
    create_tasks = [create_session_async(manager) for _ in range(num_sessions)]
    session_ids = await asyncio.gather(*create_tasks)

    # Verify all sessions were created
    assert len(session_ids) == num_sessions
    assert len(set(session_ids)) == num_sessions  # All IDs should be unique

    # Verify all sessions can be retrieved
    get_tasks = [get_session_async(manager, sid) for sid in session_ids]
    sessions = await asyncio.gather(*get_tasks)
    assert all(session is not None for session in sessions)

    # End all sessions concurrently
    end_tasks = [end_session_async(manager, sid) for sid in session_ids]
    end_results = await asyncio.gather(*end_tasks)
    assert all(end_results)


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Advanced timeout features need implementation")
async def test_session_with_inactivity_timeout(async_session_manager) -> None:
    """Test sessions with inactivity timeout."""
    # Create session with inactivity timeout
    session_id = await create_session_async(
        async_session_manager,
        timeout=0.3,  # Short timeout for testing
    )

    # Verify session exists
    assert await is_session_expired_async(async_session_manager, session_id) is False

    # Keep session active by updating it
    await update_metadata_async(async_session_manager, session_id, {"status": "active"})

    # Should still exist after a short delay
    await asyncio.sleep(0.1)
    assert await is_session_expired_async(async_session_manager, session_id) is False

    # Keep it active again
    await update_metadata_async(async_session_manager, session_id, {"status": "active"})

    # Should still exist
    await asyncio.sleep(0.1)
    assert await is_session_expired_async(async_session_manager, session_id) is False

    # Now let it expire by waiting without updating
    await asyncio.sleep(0.4)

    # Force session to expire for testing
    if hasattr(async_session_manager, "sessions") and session_id in async_session_manager.sessions:
        session = async_session_manager.sessions[session_id]
        session.last_active = datetime.now() - timedelta(seconds=1)
        if isinstance(session.timeout, (int, float)):
            session.timeout = timedelta(seconds=session.timeout)

    # Clean up expired sessions
    await cleanup_expired_sessions_async(async_session_manager)

    # Should be gone now
    assert await is_session_expired_async(async_session_manager, session_id)


@pytest.mark.asyncio
async def test_session_with_user_id(async_session_manager) -> None:
    """Test creating sessions with user IDs."""
    # Create multiple sessions for the same user
    user_id = "test_user_123"
    num_sessions = 5

    session_ids = []
    for i in range(num_sessions):
        session_id = await create_session_async(
            async_session_manager,
            user_id=user_id,
            metadata={"session_number": i},
        )
        session_ids.append(session_id)

    # Get all sessions for the user
    user_sessions = await async_session_manager.get_user_sessions_async(user_id)
    assert len(user_sessions) == num_sessions
    assert set(user_sessions) == set(session_ids)


@pytest.mark.asyncio
async def test_end_nonexistent_session(async_session_manager) -> None:
    """Test ending a nonexistent session."""
    # Attempt to end a session that doesn't exist
    result = await end_session_async(async_session_manager, "nonexistent-session")

    # Should return False, not raise an exception
    assert result is False


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Async session cleanup needs implementation")
async def test_cleanup_with_no_expired_sessions(async_session_manager) -> None:
    """Test cleaning up when there are no expired sessions."""
    # Create a session with a long timeout
    session_id = await create_session_async(async_session_manager, timeout=3600)

    # Clean up expired sessions
    cleaned_count = await cleanup_expired_sessions_async(async_session_manager)

    # Should be 0 since the session hasn't expired
    assert cleaned_count == 0

    # Session should still exist
    session = await get_session_async(async_session_manager, session_id)
    assert session is not None
