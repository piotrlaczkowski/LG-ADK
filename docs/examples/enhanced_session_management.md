# 🧑‍💻 Enhanced Session Management Example with LG-ADK

This example demonstrates how to use the enhanced session management features of LG-ADK to build a multi-user conversational application with rich analytics and user tracking.

---

!!! tip "You can copy and run this example as a script."

```python
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from lg_adk.agents.base import Agent
from lg_adk.builders.graph_builder import GraphBuilder
from lg_adk.models import get_model
from lg_adk.sessions.session_manager import SynchronizedSessionManager

class SimpleAgent(Agent):
    """A simple agent that processes user queries and generates responses."""

    def __init__(self, name: str, model_name: str = "gpt-3.5-turbo"):
        super().__init__(name=name)
        self.model = ChatOpenAI(model=model_name, temperature=0.7)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant that provides concise, accurate responses."),
                ("human", "{input}"),
                ("ai", "{agent_scratchpad}"),
            ]
        )

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the user input and generate a response."""
        # Format messages for the model
        messages = self.prompt.format_messages(input=state["input"], agent_scratchpad=state.get("agent_scratchpad", ""))

        # Generate response
        response = self.model.invoke(messages).content

        # Update state with user input and assistant response
        state["agent_scratchpad"] = response
        state["output"] = response
        state["conversation_history"] = state.get("conversation_history", []) + [
            {"role": "user", "content": state["input"]},
            {"role": "assistant", "content": response},
        ]

        return state


def get_session_summary(session_manager, session_id):
    """Get a summary of session information."""
    try:
        session = session_manager.get_session(session_id)
        metadata = session_manager.get_session_metadata(session_id)

        # Calculate duration
        created_at = metadata.get("created_at", datetime.now())
        duration = (session.last_active - created_at).total_seconds()

        # Calculate average response time
        avg_response_time = 0
        if session.interactions > 0:
            avg_response_time = session.total_response_time / session.interactions

        return {
            "user_id": metadata.get("user_id", "unknown"),
            "interactions": session.interactions,
            "total_tokens_in": session.total_tokens_in,
            "total_tokens_out": session.total_tokens_out,
            "avg_response_time": avg_response_time,
            "duration": duration,
            "last_active": session.last_active,
            "created_at": created_at,
            "metadata": metadata,
        }
    except KeyError:
        return None


def main():
    """Run the enhanced session management example."""
    # Set up OpenAI API key (replace with your key or use environment variable)
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-api-key")

    # Create a thread-safe enhanced session manager
    session_manager = SynchronizedSessionManager()

    # Create a simple agent
    agent = SimpleAgent(name="assistant", model_name="gpt-3.5-turbo")

    # Create a graph builder with session management
    builder = GraphBuilder(name="conversational_assistant")
    builder.add_agent(agent)
    builder.configure_session_management(session_manager)

    # Build the graph
    graph = builder.build()

    print("Enhanced Session Management Example")
    print("-----------------------------------")

    # User 1: Alice's conversation
    print("\n🧑‍💻 Alice starts a conversation:")
    alice_metadata = {"user_id": "alice", "device": "laptop", "location": "New York", "language": "en-US"}

    # Start Alice's session with metadata
    alice_session = session_manager.create_session(user_id="alice", metadata=alice_metadata)
    print(f"  Session created for Alice: {alice_session}")

    # Alice sends a message
    start_time = time.time()
    alice_message = "Hello! Can you tell me about machine learning?"
    alice_response = builder.run(message=alice_message, session_id=alice_session)
    response_time = time.time() - start_time

    # Track interaction metrics
    session_manager.track_interaction(
        alice_session,
        tokens_in=len(alice_message) // 4,  # Rough estimate
        tokens_out=len(alice_response["output"]) // 4,  # Rough estimate
        response_time=response_time,
    )

    print(f"  Alice: {alice_message}")
    print(f"  Assistant: {alice_response['output']}")
    print(f"  Response time: {response_time:.2f}s")

    # User 2: Bob's conversation
    print("\n🧑‍💻 Bob starts a conversation:")
    bob_metadata = {"user_id": "bob", "device": "mobile", "location": "London", "language": "en-GB"}

    # Start Bob's session with run (automatic session creation)
    bob_message = "Hi there! What's the weather like today?"
    start_time = time.time()
    bob_response = builder.run(message=bob_message, metadata=bob_metadata)
    response_time = time.time() - start_time

    bob_session = bob_response["session_id"]

    # Track interaction metrics
    session_manager.track_interaction(
        bob_session,
        tokens_in=len(bob_message) // 4,  # Rough estimate
        tokens_out=len(bob_response["output"]) // 4,  # Rough estimate
        response_time=response_time,
    )

    print(f"  Session created for Bob: {bob_session}")
    print(f"  Bob: {bob_message}")
    print(f"  Assistant: {bob_response['output']}")
    print(f"  Response time: {response_time:.2f}s")

    # Alice continues her conversation
    print("\n🧑‍💻 Alice continues her conversation:")
    alice_followup = "Can you explain neural networks specifically?"
    start_time = time.time()
    alice_response = builder.run(message=alice_followup, session_id=alice_session)
    response_time = time.time() - start_time

    # Track interaction metrics
    session_manager.track_interaction(
        alice_session,
        tokens_in=len(alice_followup) // 4,
        tokens_out=len(alice_response["output"]) // 4,
        response_time=response_time,
    )

    print(f"  Alice: {alice_followup}")
    print(f"  Assistant: {alice_response['output']}")
    print(f"  Response time: {response_time:.2f}s")

    # Bob continues his conversation
    print("\n🧑‍💻 Bob continues his conversation:")
    bob_followup = "And what about tomorrow's forecast?"
    start_time = time.time()
    bob_response = builder.run(message=bob_followup, session_id=bob_session)
    response_time = time.time() - start_time

    # Track interaction metrics
    session_manager.track_interaction(
        bob_session,
        tokens_in=len(bob_followup) // 4,
        tokens_out=len(bob_response["output"]) // 4,
        response_time=response_time,
    )

    print(f"  Bob: {bob_followup}")
    print(f"  Assistant: {bob_response['output']}")
    print(f"  Response time: {response_time:.2f}s")

    # Demonstrate session tracking features
    print("\n📊 Session Analytics and Tracking:")

    # Get all of Alice's sessions
    alice_sessions = session_manager.get_user_sessions("alice")
    print(f"  Alice's active sessions: {len(alice_sessions)}")

    # Get Alice's session statistics
    alice_summary = get_session_summary(session_manager, alice_session)
    if alice_summary:
        print(f"  Alice's session statistics:")
        print(f"    - Interactions: {alice_summary['interactions']}")
        print(f"    - Total tokens in: {alice_summary['total_tokens_in']}")
        print(f"    - Total tokens out: {alice_summary['total_tokens_out']}")
        print(f"    - Avg response time: {alice_summary['avg_response_time']:.2f}s")
        print(f"    - Session duration: {alice_summary['duration']:.2f}s")

    # Update session metadata
    session_manager.update_session_metadata(
        alice_session, {"subscription": "premium", "last_topic": "neural networks"}, merge=True
    )

    # Get updated metadata
    alice_metadata = session_manager.get_session_metadata(alice_session)
    print(f"  Alice's updated metadata: {alice_metadata}")

    # End sessions
    print("\n🔚 Ending sessions:")
    session_manager.end_session(alice_session)
    session_manager.end_session(bob_session)
    print(f"  Ended Alice's session: {alice_session}")
    print(f"  Ended Bob's session: {bob_session}")

    # Verify sessions are gone
    print(f"  Alice's session exists: {session_manager.session_exists(alice_session)}")
    print(f"  Bob's session exists: {session_manager.session_exists(bob_session)}")


if __name__ == "__main__":
    main()
```
