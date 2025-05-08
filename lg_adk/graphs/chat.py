"""Simple chat graph implementation for LangGraph CLI.

This module implements a basic chat graph that can be discovered
and executed by the LangGraph CLI.
"""

from typing import Any, TypedDict

from langgraph.graph import Graph
from pydantic import BaseModel, Field

from lg_adk.agents import Agent
from lg_adk.builders import GraphBuilder
from lg_adk.database import DatabaseManager
from lg_adk.memory import MemoryManager
from lg_adk.models import get_model


# Define message type
class Message(BaseModel):
    """Represents a chat message."""

    role: str = Field(..., description="Role of the message sender (system, user, assistant)")
    content: str = Field(..., description="Content of the message")


# Define state type
class ChatState(TypedDict):
    """Represents the state of a chat conversation."""

    messages: list[Message]
    session_id: str
    metadata: dict[str, Any]


def create_chat_graph() -> Graph:
    """Create a simple chat graph with proper session handling.

    This function builds a chat graph compatible with the LangGraph CLI.
    It properly handles session state and context across multiple turns.

    Returns:
        Graph: A LangGraph graph instance for chat functionality.
    """
    # Create agent
    chat_agent = Agent(
        name="chat_assistant",
        model=get_model("openai/gpt-4"),
        system_prompt="You are a helpful, friendly assistant. Answer user questions concisely and accurately.",
    )

    # Create memory manager
    db_manager = DatabaseManager(connection_string="sqlite:///chat_memory.db")
    memory_manager = MemoryManager(database_manager=db_manager)

    # Create graph builder
    builder = GraphBuilder(name="chat")
    builder.add_agent(chat_agent)
    builder.add_memory(memory_manager)

    # Configure state tracking
    builder.configure_state_tracking(
        include_session_id=True,
        include_metadata=True,
    )

    # Process user input
    def process_user_input(state: ChatState, user_input: str) -> ChatState:
        """Add user input to state."""
        messages = state.get("messages", [])
        user_message = Message(role="user", content=user_input)
        return {
            **state,
            "messages": messages + [user_message],
        }

    # Generate assistant response
    def generate_response(state: ChatState) -> ChatState:
        """Generate assistant response based on conversation history."""
        messages = state.get("messages", [])
        session_id = state.get("session_id")

        # Configure for proper checkpointing
        config = {
            "configurable": {
                "thread_id": session_id,
            },
        }

        # Prepare messages for the model
        model_messages = []
        for msg in messages:
            model_messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                },
            )

        # Generate response from the model
        response = chat_agent.model.generate(
            messages=model_messages,
            config=config,
        )

        # Add assistant response to messages
        assistant_message = Message(role="assistant", content=response)
        return {
            **state,
            "messages": messages + [assistant_message],
        }

    # Wire up the graph
    builder.add_node("process_input", process_user_input)
    builder.add_node("generate_response", generate_response)

    builder.add_edge("process_input", "generate_response")
    builder.add_edge("generate_response", "process_input")

    # Set entry and exit points
    builder.set_entry_point("process_input")
    builder.set_exit_point("generate_response")

    # Build the graph with proper typing
    typed_graph: Graph[ChatState] = builder.build()
    return typed_graph


# Export the graph for the LangGraph CLI to discover
graph = create_chat_graph()

# Example of how to use the graph directly
if __name__ == "__main__":
    import uuid

    # Create a session ID
    session_id = str(uuid.uuid4())

    # Configure for using with checkpointer
    config = {
        "configurable": {
            "thread_id": session_id,
        },
    }

    # Initialize state
    initial_state = {
        "messages": [],
        "session_id": session_id,
        "metadata": {"started_at": str(uuid.uuid4())},
    }

    # Process messages
    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break

        # Process input through the graph
        result = graph.invoke(
            {"user_input": user_input},
            config=config,
            state=initial_state,
        )

        # Update state for next iteration
        initial_state = result

        # Extract and print the last message (assistant's response)
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            if last_message.role == "assistant":
                pass
