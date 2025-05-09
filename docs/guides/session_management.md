---
title: Session Management in LG-ADK
---

# 🗂️ Session Management in LG-ADK

Session management is a critical aspect of building conversational applications with LangGraph. LG-ADK provides a powerful enhanced session management system that builds on top of LangGraph's native session capabilities while adding valuable features.

## 🤔 Why Use Session Management?

Sessions let you track conversations, users, and context over time! 🗂️

## 🧩 Key Concepts

- 🆔 **Session ID**: Unique identifier for each session
- 👤 **User ID**: Track sessions per user
- 📝 **Metadata**: Store extra info (source, device, etc.)
- ⏳ **Timeouts**: Auto-expire inactive sessions

## 🚦 Quick Example

!!! tip "Add session management to your graph"
    ```python
    from lg_adk.sessions import SessionManager
    builder.add_session_manager(SessionManager())
    ```

## 🛠️ How Session Management Works

- Sessions are managed by `SessionManager` or `EnhancedSessionManager`
- You can register, update, and end sessions
- Metadata and analytics are tracked for each session

## 🚨 Common Pitfalls

!!! warning "Session expiration"
    Make sure to configure timeouts and cleanup for long-running apps.

## 🌟 Next Steps

- [Memory Management](memory_management.md) ��
- [Building Graphs](building_graphs.md) 🏗️
- [Examples](../examples/) 💡

## Understanding Sessions and Their Importance

Sessions are crucial for:

1. **Conversation Persistence**: Maintaining the state and history of a conversation across multiple interactions
2. **Context Preservation**: Keeping context and memory accessible throughout a conversation
3. **Memory Scoping**: Ensuring memories and state are isolated between different conversations
4. **Multi-tenant Applications**: Supporting multiple concurrent users with separate contexts

## Building on LangGraph's Native Sessions

LG-ADK's session management enhances LangGraph's built-in session capabilities:

- Uses LangGraph's native session management for core functionality
- Adds advanced features like user association, rich metadata, and analytics
- Provides a clean API that works with both LangGraph's sessions and our enhancements
- Transparently adapts to the available LangGraph features

## Using Session IDs with LG-ADK Graphs

When interacting with LG-ADK graphs, the session ID is a key parameter:

```python
from lg_adk.builders.graph_builder import GraphBuilder
from lg_adk.sessions.session_manager import SynchronizedSessionManager

# Create a session manager
session_manager = SynchronizedSessionManager()

# Create and configure a graph builder
builder = GraphBuilder(name="my_graph")
builder.add_agent(my_agent)
builder.configure_session_management(session_manager)

# Build the graph
graph = builder.build()

# Run without session ID (new session will be created)
response = builder.run(message="Hello!", metadata={"user_id": "alice"})
session_id = response["session_id"]

# Continue the conversation with the same session
follow_up = builder.run(message="Tell me more", session_id=session_id)
```

## Enhanced Session Manager Types

LG-ADK provides several session manager implementations:

- `SessionManager`: Base implementation with user tracking, metadata, and analytics
- `SynchronizedSessionManager`: Thread-safe implementation for production use
- `DatabaseSessionManager`: Persistent implementation that stores sessions in a database
- `AsyncSessionManager`: Asynchronous implementation for async/await code

## Tracking Users and Sessions

A key feature of LG-ADK's session management is user tracking:

```python
# Create a session with user association
session_id = session_manager.create_session(user_id="alice")

# Or add user information in metadata
response = builder.run(
    message="Hello!",
    metadata={"user_id": "alice", "device": "mobile"}
)

# Get all sessions for a user
user_sessions = session_manager.get_user_sessions("alice")
```

## Session Metadata Management

LG-ADK provides rich metadata management for sessions:

```python
# Add metadata when creating a session
session_id = session_manager.create_session(metadata={"source": "web", "locale": "en-US"})

# Update session metadata
session_manager.update_session_metadata(
    session_id,
    {"last_page": "checkout"},
    merge=True  # Merge with existing metadata (default)
)

# Get session metadata
metadata = session_manager.get_session_metadata(session_id)
```

## Session Analytics and Tracking

Track and analyze session usage with built-in analytics:

```python
# Get session object with all tracking information
session = session_manager.get_session(session_id)

# Access session statistics
interaction_count = session.interactions
total_tokens_in = session.total_tokens_in
total_tokens_out = session.total_tokens_out
response_time = session.total_response_time
last_active = session.last_active

# Track an interaction manually (usually done automatically by GraphBuilder)
session_manager.track_interaction(
    session_id,
    tokens_in=15,     # Input token count
    tokens_out=25,    # Output token count
    response_time=1.2 # Response time in seconds
)
```

## Session Lifecycle Management

Manage the complete lifecycle of sessions:

```python
# Create a session
session_id = session_manager.create_session()

# Check if a session exists
if session_manager.session_exists(session_id):
    # Use the session
    pass

# End a session when it's no longer needed
session_manager.end_session(session_id)

# Clear expired sessions
expired_count = session_manager.cleanup_expired_sessions()
```

## Asynchronous Session Management

LG-ADK provides full async support for session management:

```python
from lg_adk.sessions.session_manager import AsyncSessionManager

# Create an async session manager
async_manager = AsyncSessionManager()

# Register a session asynchronously
session_id = await async_manager.acreate_session(user_id="alice")

# Track interaction asynchronously
await async_manager.atrack_interaction(
    session_id,
    tokens_in=10,
    tokens_out=20,
    response_time=0.5
)

# Use with graph builder's async run method
result = await builder.arun(message="Hello!", session_id=session_id)
```

## Thread Safety with SynchronizedSessionManager

For production applications, use the thread-safe session manager:

```python
from lg_adk.sessions.session_manager import SynchronizedSessionManager

# Create a thread-safe session manager
session_manager = SynchronizedSessionManager()

# All operations are now thread-safe and can be called from multiple threads
session_id = session_manager.create_session()
```

## Persistent Sessions with DatabaseSessionManager

For long-running applications, use database-backed sessions:

```python
from lg_adk.sessions.session_manager import DatabaseSessionManager
from lg_adk.database.database_manager import DatabaseManager

# Create a database manager
db_manager = DatabaseManager()

# Create a database-backed session manager
session_manager = DatabaseSessionManager(db_manager=db_manager)

# Sessions will now persist across application restarts
session_id = session_manager.create_session()
```

## Session Timeout and Expiration

Configure session timeouts to automatically clean up inactive sessions:

```python
# Create a session with a 30-minute timeout
session_id = session_manager.create_session(timeout=1800)  # 30 minutes in seconds

# Or with a timedelta
from datetime import timedelta
session_id = session_manager.create_session(timeout=timedelta(hours=1))

# Clean up expired sessions periodically
expired_count = session_manager.cleanup_expired_sessions()
```

## Complete Example

Here's a complete example demonstrating session management with LG-ADK:

```python
from lg_adk.builders.graph_builder import GraphBuilder
from lg_adk.agents.base import Agent
from lg_adk.sessions.session_manager import SynchronizedSessionManager
from lg_adk.models import get_model

# Create thread-safe session manager for production use
session_manager = SynchronizedSessionManager()

# Create an agent
agent = Agent(
    name="assistant",
    model=get_model("openai/gpt-3.5-turbo")
)

# Create a graph builder
builder = GraphBuilder(name="my_app")
builder.add_agent(agent)
builder.configure_session_management(session_manager)
graph = builder.build()

# User 1: Start a new conversation
alice_metadata = {"user_id": "alice", "device": "mobile"}
alice_response = builder.run(
    message="Hello! Can you help me with my project?",
    metadata=alice_metadata
)
alice_session_id = alice_response["session_id"]

# User 2: Start a different conversation
bob_metadata = {"user_id": "bob", "device": "web"}
bob_response = builder.run(
    message="What's the weather like today?",
    metadata=bob_metadata
)
bob_session_id = bob_response["session_id"]

# Continue Alice's conversation
alice_followup = builder.run(
    message="I need help with Python.",
    session_id=alice_session_id
)

# Get analytics for both users
alice_session = session_manager.get_session(alice_session_id)
bob_session = session_manager.get_session(bob_session_id)

print(f"Alice's interactions: {alice_session.interactions}")
print(f"Bob's interactions: {bob_session.interactions}")

# End sessions when done
session_manager.end_session(alice_session_id)
session_manager.end_session(bob_session_id)
```

## Best Practices

1. **Use SynchronizedSessionManager for Production**: The thread-safe implementation prevents race conditions in multi-threaded environments.

2. **Associate Users with Sessions**: Always associate sessions with users when possible for better tracking and analytics.

3. **Clean Up Sessions**: Call `end_session()` when a conversation is complete to free resources.

4. **Use Timeouts**: Configure appropriate timeouts to automatically clean up inactive sessions.

5. **Track Interactions**: Use the `track_interaction()` method to record token usage and response times.

6. **Store Minimal Metadata**: Only store necessary information in session metadata to avoid bloat.

7. **Leverage Analytics**: Use the built-in analytics to understand user behavior and optimize your application.

8. **Use Async When Appropriate**: For async applications, use the AsyncSessionManager for better performance.

By leveraging LG-ADK's enhanced session management, you can build sophisticated conversational applications that maintain context, track users, and provide rich analytics while seamlessly integrating with LangGraph's native capabilities.
