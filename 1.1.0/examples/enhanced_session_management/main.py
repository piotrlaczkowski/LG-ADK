"""
Enhanced Session Management Example for LG-ADK

This example demonstrates advanced session management capabilities with
customized session analytics and multi-user support.
"""
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

# Import LangGraph and other dependencies
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from lg_adk.agents import Agent

# Import LG-ADK components
from lg_adk.builders import GraphBuilder
from lg_adk.memory import MemoryManager
from lg_adk.models import get_model
from lg_adk.sessions import SynchronizedSessionManager


def build_graph():
    """Build an example graph with enhanced session management."""
    # Initialize components
    model = get_model("openai/gpt-3.5-turbo")

    # Create agent
    agent = Agent(
        model=model,
        system_prompt="""You are a helpful assistant with enhanced memory capabilities.
        You can remember previous conversations within the same session.
        Be concise and helpful in your responses.""",
    )

    # Create session manager
    session_manager = SynchronizedSessionManager()

    # Build graph
    builder = GraphBuilder(name="session_example")
    builder.add_agent(agent)
    builder.configure_session_management(session_manager)
    graph = builder.build()

    return graph, session_manager


def calculate_session_duration(session_manager, session_id):
    """Calculate the duration of a session in seconds."""
    session = session_manager.get_session(session_id)
    metadata = session_manager.get_session_metadata(session_id)
    if not metadata or "created_at" not in metadata:
        return 0

    # Calculate duration from creation to last active time
    created_at = metadata["created_at"]
    return (session.last_active - created_at).total_seconds()


def update_response_metrics(session_manager, session_id, response_time):
    """Update response time metrics for a session."""
    session_manager.track_interaction(
        session_id,
        tokens_in=0,  # Not tracking tokens here
        tokens_out=0,
        response_time=response_time,
    )


def get_session_analytics(session_manager):
    """Return analytics data for all sessions."""
    analytics = {}
    for session_id in session_manager.active_sessions:
        session = session_manager.get_session(session_id)
        metadata = session_manager.get_session_metadata(session_id)
        user_id = metadata.get("user_id", "unknown")

        duration = calculate_session_duration(session_manager, session_id)

        analytics[session_id] = {
            "user_id": user_id,
            "total_messages": session.interactions,
            "avg_response_time": session.total_response_time / max(1, session.interactions),
            "session_duration": round(duration, 2),
            "last_active": session.last_active.strftime("%Y-%m-%d %H:%M:%S"),
            "tokens_in": session.total_tokens_in,
            "tokens_out": session.total_tokens_out,
        }
    return analytics


def interactive_cli(graph, session_manager):
    """Interactive CLI to demonstrate session management."""
    active_sessions = {}

    def print_menu():
        print("\n=== Enhanced Session Management Demo ===")
        print("1. Start new session as Alice")
        print("2. Start new session as Bob")
        print("3. Resume existing session")
        print("4. List active sessions")
        print("5. View session analytics")
        print("6. Clear expired sessions")
        print("7. End specific session")
        print("8. Quit")
        print("======================================")

    while True:
        print_menu()
        choice = input("Enter your choice (1-8): ")

        if choice == "1":
            user_id = "alice"
            session_id = session_manager.create_session(
                user_id=user_id, metadata={"user_id": user_id, "device": "mobile", "location": "New York"}
            )
            active_sessions[session_id] = user_id
            print(f"Started new session for Alice (ID: {session_id})")
            chat_session(graph, session_manager, session_id)

        elif choice == "2":
            user_id = "bob"
            session_id = session_manager.create_session(
                user_id=user_id, metadata={"user_id": user_id, "device": "laptop", "location": "San Francisco"}
            )
            active_sessions[session_id] = user_id
            print(f"Started new session for Bob (ID: {session_id})")
            chat_session(graph, session_manager, session_id)

        elif choice == "3":
            if not active_sessions:
                print("No active sessions to resume!")
                continue

            print("\nAvailable sessions:")
            for idx, (sid, uid) in enumerate(active_sessions.items(), 1):
                print(f"{idx}. User: {uid}, Session ID: {sid}")

            try:
                idx = int(input("\nSelect session to resume (number): ")) - 1
                session_ids = list(active_sessions.keys())
                if 0 <= idx < len(session_ids):
                    session_id = session_ids[idx]
                    chat_session(graph, session_manager, session_id)
                else:
                    print("Invalid selection!")
            except ValueError:
                print("Please enter a valid number")

        elif choice == "4":
            print("\nActive Sessions:")
            for session_id in session_manager.active_sessions:
                try:
                    session = session_manager.get_session(session_id)
                    metadata = session_manager.get_session_metadata(session_id)
                    user_id = metadata.get("user_id", "unknown")
                    created_at = metadata.get("created_at", datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
                    last_active = session.last_active.strftime("%Y-%m-%d %H:%M:%S")

                    print(f"Session ID: {session_id}")
                    print(f"  User: {user_id}")
                    print(f"  Created: {created_at}")
                    print(f"  Last Activity: {last_active}")
                    print(f"  Messages: {session.interactions}")
                    print(f"  Tokens (in/out): {session.total_tokens_in}/{session.total_tokens_out}")
                    print(f"  Device: {metadata.get('device', 'unknown')}")
                    print(f"  Location: {metadata.get('location', 'unknown')}")
                    print("---")
                except KeyError:
                    continue

        elif choice == "5":
            analytics = get_session_analytics(session_manager)
            print("\nSession Analytics:")
            for session_id, stats in analytics.items():
                print(f"Session {session_id} (User: {stats['user_id']}):")
                print(f"  Messages: {stats['total_messages']}")
                print(f"  Avg Response Time: {stats['avg_response_time']:.2f}s")
                print(f"  Duration: {stats['session_duration']}s")
                print(f"  Last Active: {stats['last_active']}")
                print(f"  Tokens (in/out): {stats['tokens_in']}/{stats['tokens_out']}")
                print("---")

        elif choice == "6":
            expired_count = session_manager.cleanup_expired_sessions()
            print(f"Cleared {expired_count} expired sessions")
            # Also clean up our local tracking
            active_sessions = {k: v for k, v in active_sessions.items() if session_manager.session_exists(k)}

        elif choice == "7":
            if not active_sessions:
                print("No active sessions to end!")
                continue

            print("\nAvailable sessions:")
            for idx, (sid, uid) in enumerate(active_sessions.items(), 1):
                print(f"{idx}. User: {uid}, Session ID: {sid}")

            try:
                idx = int(input("\nSelect session to end (number): ")) - 1
                session_ids = list(active_sessions.keys())
                if 0 <= idx < len(session_ids):
                    session_id = session_ids[idx]
                    success = session_manager.end_session(session_id)
                    if success:
                        print(f"Session {session_id} ended successfully")
                        del active_sessions[session_id]
                    else:
                        print(f"Failed to end session {session_id}")
                else:
                    print("Invalid selection!")
            except ValueError:
                print("Please enter a valid number")

        elif choice == "8":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")


def chat_session(graph, session_manager, session_id):
    """Run an interactive chat session with the given session ID."""
    print(f"\n=== Chat Session: {session_id} ===")
    print("(Type 'exit' to end session, 'quit' to exit program)")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            print("Ending session and returning to menu...")
            break
        elif user_input.lower() == "quit":
            print("Exiting program...")
            exit(0)

        # Track response time
        start_time = time.time()

        # Track input tokens (rough estimate: 4 chars ≈ 1 token)
        approx_input_tokens = len(user_input) // 4

        # Run the message through the graph
        response = graph.run(message=user_input, session_id=session_id)

        # Calculate response time
        end_time = time.time()
        response_time = end_time - start_time

        # Track output tokens (rough estimate: 4 chars ≈ 1 token)
        output = response.get("output", "")
        approx_output_tokens = len(output) // 4

        # Update session stats with the interaction
        session_manager.track_interaction(
            session_id, tokens_in=approx_input_tokens, tokens_out=approx_output_tokens, response_time=response_time
        )

        print(f"\nAssistant: {output}")
        print(f"(Response time: {response_time:.2f}s)")


if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Either set it with 'export OPENAI_API_KEY=your_key' or enter it now:")
        api_key = input("OPENAI_API_KEY: ")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            print("No API key provided. Exiting.")
            exit(1)

    # Build the graph and start the CLI
    graph, session_manager = build_graph()
    interactive_cli(graph, session_manager)
