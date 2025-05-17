"""
Example demonstrating enhanced session management with LangGraph.

This example shows how to use the LG-ADK's enhanced session management capabilities
with LangGraph, including user tracking, rich metadata, and analytics features.
"""

import os
import uuid
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from lg_adk.agents.base import Agent
from lg_adk.builders.graph_builder import GraphBuilder
from lg_adk.sessions.session_manager import SynchronizedSessionManager


# Set up a simple agent for this example
class SimpleAgent(Agent):
    """A simple agent that responds to user queries."""

    def __init__(self, name: str = "simple_agent"):
        """Initialize the agent."""
        super().__init__(name=name)

        # Set up the model
        model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.7)

    @property
    def prompt(self):
        return ChatPromptTemplate.from_template("You are a helpful AI assistant. Answer the user's questions.")

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the state and generate a response."""
        # Extract input and context from state
        user_input = getattr(state, "input", "")
        messages = getattr(state, "messages", [])

        # Format messages for context
        formatted_messages = "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in messages])

        # Generate response
        chain = self.prompt | self.llm
        response = chain.invoke({"input": user_input, "messages": formatted_messages})

        # Build new state dict
        new_state = dict(state.__dict__)
        new_state["output"] = response.content
        new_state["agent"] = self.name

        # Add the new message to the history
        if "messages" not in new_state:
            new_state["messages"] = []

        # Add user message if not already present
        if not any(msg.get("role") == "user" and msg.get("content") == user_input for msg in new_state["messages"]):
            new_state["messages"].append({"role": "user", "content": user_input})

        # Add assistant response
        new_state["messages"].append({"role": "assistant", "content": response.content})

        return new_state


def main():
    """Run the session management example."""
    # Create a synchronized session manager (thread-safe for production use)
    session_manager = SynchronizedSessionManager()

    # Create a simple agent
    agent = SimpleAgent()

    # Create a graph builder with the agent and session manager
    builder = GraphBuilder(name="session_example")
    builder.add_agent(agent)
    builder.configure_session_management(session_manager)

    # Build the graph
    graph = builder.build()

    print("\n🔄 Starting new session for user 'alice'...\n")

    # Start a new conversation (simulate a user session)
    alice_metadata = {"user_id": "alice", "device": "iPhone", "locale": "en-US", "source": "mobile_app"}

    # First message in the conversation
    response1 = builder.run(
        message="Hello! Can you tell me about session management in LangGraph?", metadata=alice_metadata
    )

    # Get the session ID that was created
    alice_session_id = response1["session_id"]
    print(f"Created session: {alice_session_id}")
    print(f"Response: {response1['output']}\n")

    # Continue the conversation using the same session ID
    response2 = builder.run(message="How does it help with multi-user applications?", session_id=alice_session_id)
    print(f"Response: {response2['output']}\n")

    # Start a new conversation for a different user
    print("\n🔄 Starting new session for user 'bob'...\n")

    bob_metadata = {"user_id": "bob", "device": "Android", "locale": "en-GB", "source": "web_browser"}

    # First message in the second conversation
    response3 = builder.run(
        message="What's the difference between StateGraph and Graph in LangGraph?", metadata=bob_metadata
    )

    # Get the session ID for Bob
    bob_session_id = response3["session_id"]
    print(f"Created session: {bob_session_id}")
    print(f"Response: {response3['output']}\n")

    # Now demonstrate the enhanced features
    print("\n📊 Enhanced Session Features:\n")

    # 1. Get all sessions for a user
    alice_sessions = session_manager.get_user_sessions("alice")
    print(f"Alice's sessions: {alice_sessions}")

    # 2. Get session statistics and analytics
    alice_analytics = session_manager.get_session_analytics(alice_session_id)
    print(f"Alice's session statistics:")
    if alice_analytics:
        print(f"  - Interactions: {alice_analytics.get('message_count', 0)}")
        print(f"  - Total tokens in: {alice_analytics.get('total_tokens_in', 0)}")
        print(f"  - Total tokens out: {alice_analytics.get('total_tokens_out', 0)}")
        avg_response_time = alice_analytics.get("total_response_time", 0) / max(
            1, alice_analytics.get("message_count", 1)
        )
        print(f"  - Average response time: {avg_response_time:.2f}s")
    else:
        print("  - No analytics available.")

    # 3. Get session metadata
    alice_metadata = session_manager.get_session_metadata(alice_session_id)
    print(f"Alice's metadata: {alice_metadata}")

    # 4. Update session metadata
    session_manager.update_session_metadata(
        alice_session_id, {"subscription_tier": "premium", "last_topic": "session_management"}, merge=True
    )

    # Get updated metadata
    updated_metadata = session_manager.get_session_metadata(alice_session_id)
    print(f"Alice's updated metadata: {updated_metadata}")

    # Calculate session duration
    metadata = session_manager.get_session_metadata(alice_session_id)
    created_at = metadata.get("created_at", datetime.now())
    duration = (datetime.now() - created_at).total_seconds()
    print(f"Alice's session duration: {duration:.2f} seconds")

    # Cleanup
    print("\n🧹 Cleaning up sessions...")
    session_manager.end_session(alice_session_id)
    session_manager.end_session(bob_session_id)
    print("Sessions ended.")


if __name__ == "__main__":
    main()
