"""
Tests for thread-safe session management functionality.

This module tests the SynchronizedSessionManager class which provides
thread-safe session management capabilities.
"""

import pytest
import time
import threading
import concurrent.futures
from typing import Dict, Any, List, Optional

from lg_adk.sessions.session_manager import (
    SynchronizedSessionManager,
    Session,
)


@pytest.fixture
def sync_session_manager():
    """Return a synchronized session manager for testing."""
    return SynchronizedSessionManager()


class TestSynchronizedSessionManager:
    """Test the SynchronizedSessionManager class."""
    
    def test_basic_functionality(self, sync_session_manager):
        """Test that basic session management works with thread safety."""
        # Create session
        session_id = sync_session_manager.create_session()
        
        # Verify session was created
        session = sync_session_manager.get_session(session_id)
        assert session.session_id == session_id
        
        # Update session
        sync_session_manager.update_session(session_id)
        
        # End session
        success = sync_session_manager.end_session(session_id)
        assert success
        
        # Verify session was removed
        with pytest.raises(KeyError):
            sync_session_manager.get_session(session_id)
    
    def test_metadata_management(self, sync_session_manager):
        """Test metadata management with thread safety."""
        # Create session with initial metadata
        initial_metadata = {"user": "test_user", "topic": "test_topic"}
        session_id = sync_session_manager.create_session(metadata=initial_metadata)
        
        # Verify initial metadata
        session = sync_session_manager.get_session(session_id)
        assert session.metadata == initial_metadata
        
        # Update metadata
        update_metadata = {"status": "active", "priority": "high"}
        sync_session_manager.update_session_metadata(session_id, update_metadata)
        
        # Verify updated metadata (should be merged)
        updated_session = sync_session_manager.get_session(session_id)
        expected_metadata = {**initial_metadata, **update_metadata}
        assert updated_session.metadata == expected_metadata
    
    def test_concurrent_session_creation(self, sync_session_manager):
        """Test concurrent session creation."""
        session_ids = []
        num_sessions = 20
        
        def create_session():
            session_id = sync_session_manager.create_session()
            session_ids.append(session_id)
            return session_id
        
        # Create sessions concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_session) for _ in range(num_sessions)]
            concurrent.futures.wait(futures)
        
        # Verify all sessions were created
        assert len(session_ids) == num_sessions
        
        # Verify each session has a unique ID
        assert len(set(session_ids)) == num_sessions
        
        # Verify all sessions can be retrieved
        for session_id in session_ids:
            session = sync_session_manager.get_session(session_id)
            assert session.session_id == session_id
    
    def test_concurrent_metadata_updates(self, sync_session_manager):
        """Test concurrent metadata updates."""
        # Create a session
        session_id = sync_session_manager.create_session()
        
        # Define metadata updates
        metadata_updates = [
            {"key" + str(i): "value" + str(i)} for i in range(20)
        ]
        
        results = []
        
        def update_metadata(metadata):
            sync_session_manager.update_session_metadata(session_id, metadata)
            return True
        
        # Update metadata concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(update_metadata, md) for md in metadata_updates]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        
        # Verify all updates were processed
        assert len(results) == len(metadata_updates)
        assert all(results)
        
        # Verify final session metadata contains all updates
        session = sync_session_manager.get_session(session_id)
        for metadata in metadata_updates:
            for key, value in metadata.items():
                assert session.metadata.get(key) == value
    
    def test_concurrent_session_access(self, sync_session_manager):
        """Test concurrent session access and updates."""
        # Create a session
        session_id = sync_session_manager.create_session()
        
        access_count = 0
        lock = threading.Lock()
        
        def access_session():
            nonlocal access_count
            # Get the session
            session = sync_session_manager.get_session(session_id)
            assert session.session_id == session_id
            
            # Simulate work
            time.sleep(0.01)
            
            # Track access
            with lock:
                access_count += 1
                
            return session
        
        # Access session concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(access_session) for _ in range(50)]
            concurrent.futures.wait(futures)
        
        # Verify all accesses were processed
        assert access_count == 50
    
    def test_concurrent_session_tracking(self, sync_session_manager):
        """Test concurrent session interaction tracking."""
        # Create a session
        session_id = sync_session_manager.create_session()
        
        def track_interaction(i):
            sync_session_manager.track_interaction(
                session_id,
                tokens_in=10 * i,
                tokens_out=5 * i,
                response_time=0.1 * i
            )
            return True
        
        # Track interactions concurrently
        num_interactions = 10
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(track_interaction, i + 1) for i in range(num_interactions)]
            concurrent.futures.wait(futures)
        
        # Get session
        session = sync_session_manager.get_session(session_id)
        
        # Verify interactions were tracked
        assert session.interactions == num_interactions
        
        # Verify token counting is accurate
        # The sum of tokens_in should be 10*(1+2+...+10) = 10*55 = 550
        expected_tokens_in = 10 * sum(range(1, num_interactions + 1))
        expected_tokens_out = 5 * sum(range(1, num_interactions + 1))
        
        assert session.total_tokens_in == expected_tokens_in
        assert session.total_tokens_out == expected_tokens_out
    
    def test_concurrent_session_cleanup(self, sync_session_manager):
        """Test concurrent session creation and cleanup."""
        # Create sessions with short timeout
        num_sessions = 20
        session_ids = []
        
        for _ in range(num_sessions):
            session_id = sync_session_manager.create_session(timeout=0.1)
            session_ids.append(session_id)
        
        # Wait for sessions to expire
        time.sleep(0.2)
        
        # Clean up sessions concurrently with new session creation
        def cleanup_sessions():
            return sync_session_manager.cleanup_expired_sessions()
            
        def create_new_session():
            return sync_session_manager.create_session(timeout=3600)
        
        cleanup_results = []
        new_session_ids = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit cleanup tasks
            cleanup_futures = [executor.submit(cleanup_sessions) for _ in range(5)]
            
            # Submit new session creation tasks
            creation_futures = [executor.submit(create_new_session) for _ in range(10)]
            
            # Get cleanup results
            for future in concurrent.futures.as_completed(cleanup_futures):
                cleanup_results.append(future.result())
                
            # Get new session IDs
            for future in concurrent.futures.as_completed(creation_futures):
                new_session_ids.append(future.result())
        
        # Verify all expired sessions were cleaned up
        # The sum of all cleaned sessions across all cleanup calls should be num_sessions
        assert sum(cleanup_results) == num_sessions
        
        # Verify all old sessions are gone
        for session_id in session_ids:
            with pytest.raises(KeyError):
                sync_session_manager.get_session(session_id)
                
        # Verify all new sessions exist
        assert len(new_session_ids) == 10
        for session_id in new_session_ids:
            session = sync_session_manager.get_session(session_id)
            assert session.session_id == session_id
    
    def test_race_condition_session_update(self, sync_session_manager):
        """Test for race conditions in session updates."""
        # Create a session
        session_id = sync_session_manager.create_session()
        
        # Define a function that simulates a slow update
        def slow_update():
            # Get the session
            session = sync_session_manager.get_session(session_id)
            
            # Simulate slow processing
            time.sleep(0.05)
            
            # Update session
            sync_session_manager.update_session(session_id)
            
            # Simulate more work
            time.sleep(0.05)
            
            # Track interaction
            sync_session_manager.track_interaction(
                session_id,
                tokens_in=10,
                tokens_out=5,
                response_time=0.1
            )
            
            return True
        
        # Run multiple updates concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(slow_update) for _ in range(10)]
            concurrent.futures.wait(futures)
        
        # Get the final session state
        session = sync_session_manager.get_session(session_id)
        
        # Verify session was updated correctly
        assert session.interactions == 10
        assert session.total_tokens_in == 100
        assert session.total_tokens_out == 50
    
    def test_concurrent_session_end(self, sync_session_manager):
        """Test concurrent session ending."""
        # Create sessions
        num_sessions = 20
        session_ids = [sync_session_manager.create_session() for _ in range(num_sessions)]
        
        # End sessions concurrently
        results = []
        
        def end_session(session_id):
            result = sync_session_manager.end_session(session_id)
            return result
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(end_session, sid) for sid in session_ids]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        
        # Verify all sessions were ended
        assert len(results) == num_sessions
        assert all(results)
        
        # Verify all sessions are gone
        for session_id in session_ids:
            with pytest.raises(KeyError):
                sync_session_manager.get_session(session_id)
    
    def test_session_creation_during_cleanup(self, sync_session_manager):
        """Test session creation during cleanup."""
        # Create sessions with short timeout
        num_sessions = 10
        for _ in range(num_sessions):
            sync_session_manager.create_session(timeout=0.1)
        
        # Wait for sessions to expire
        time.sleep(0.2)
        
        # Define functions for concurrent operations
        def do_cleanup():
            return sync_session_manager.cleanup_expired_sessions()
            
        def create_session():
            return sync_session_manager.create_session()
        
        # Run cleanup and creation concurrently
        cleanup_result = None
        new_session_id = None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            cleanup_future = executor.submit(do_cleanup)
            create_future = executor.submit(create_session)
            
            cleanup_result = cleanup_future.result()
            new_session_id = create_future.result()
        
        # Verify cleanup was successful
        assert cleanup_result == num_sessions
        
        # Verify new session was created
        assert new_session_id is not None
        session = sync_session_manager.get_session(new_session_id)
        assert session.session_id == new_session_id
    
    def test_large_number_of_concurrent_operations(self, sync_session_manager):
        """Test many concurrent operations on the session manager."""
        # Create a large number of sessions
        num_sessions = 50
        session_ids = [sync_session_manager.create_session() for _ in range(num_sessions)]
        
        # Define operation types
        operations = []
        
        # Add get operations
        for session_id in session_ids:
            operations.append(("get", session_id))
        
        # Add update operations
        for session_id in session_ids[:25]:
            operations.append(("update", session_id))
        
        # Add metadata operations
        for i, session_id in enumerate(session_ids[:20]):
            operations.append(("metadata", session_id, {"key" + str(i): "value" + str(i)}))
        
        # Add track operations
        for i, session_id in enumerate(session_ids[:30]):
            operations.append(("track", session_id, i))
        
        # Add end operations for some sessions
        for session_id in session_ids[40:]:
            operations.append(("end", session_id))
        
        # Shuffle operations to simulate random access
        import random
        random.shuffle(operations)
        
        # Execute operations concurrently
        results = []
        
        def execute_operation(op):
            op_type = op[0]
            session_id = op[1]
            
            try:
                if op_type == "get":
                    session = sync_session_manager.get_session(session_id)
                    return ("get", session_id, session.session_id == session_id)
                    
                elif op_type == "update":
                    sync_session_manager.update_session(session_id)
                    return ("update", session_id, True)
                    
                elif op_type == "metadata":
                    metadata = op[2]
                    sync_session_manager.update_session_metadata(session_id, metadata)
                    return ("metadata", session_id, True)
                    
                elif op_type == "track":
                    tokens = op[2]
                    sync_session_manager.track_interaction(
                        session_id,
                        tokens_in=tokens * 10,
                        tokens_out=tokens * 5,
                        response_time=0.1
                    )
                    return ("track", session_id, True)
                    
                elif op_type == "end":
                    success = sync_session_manager.end_session(session_id)
                    return ("end", session_id, success)
                    
                return (op_type, session_id, False)
                
            except Exception as e:
                return (op_type, session_id, str(e))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(execute_operation, op) for op in operations]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        
        # Analyze results
        successes = [r for r in results if r[2] is True]
        failures = [r for r in results if r[2] is not True]
        
        # Some operations might fail due to sessions being ended by other operations,
        # but we should have a high success rate
        assert len(successes) > len(operations) * 0.8
        
        # Get remaining sessions
        remaining_session_ids = []
        for session_id in session_ids:
            try:
                sync_session_manager.get_session(session_id)
                remaining_session_ids.append(session_id)
            except KeyError:
                pass
        
        # Ended sessions should be gone
        ended_session_ids = [op[1] for op in operations if op[0] == "end"]
        for session_id in ended_session_ids:
            assert session_id not in remaining_session_ids 