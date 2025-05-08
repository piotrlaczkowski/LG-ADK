"""
Simple chat example using LG-ADK.
"""

import os

from lg_adk import Agent, GraphBuilder, MemoryManager, SessionManager
from lg_adk.config.settings import Settings
from lg_adk.tools.web_search import WebSearchTool


def main():
    """Run the simple chat example."""
    # Load settings from environment variables
    settings = Settings.from_env()

    # Create an agent
    assistant = Agent(
        name="assistant",
        llm=settings.default_llm,
        description="A helpful AI assistant that answers user questions",
    )

    # Add a tool to the agent
    assistant.add_tool(WebSearchTool())

    # Create memory and session managers
    memory_manager = MemoryManager()
    session_manager = SessionManager(memory_manager=memory_manager)

    # Create a new session
    session = session_manager.create_session()

    # Create a graph with the agent
    builder = GraphBuilder()
    builder.add_agent(assistant)

    # Build the graph
    graph = builder.build()

    # Simple chat loop
    print("Simple Chat Example (type 'exit' to quit)")
    print("----------------------------------------")

    while True:
        # Get user input
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            break

        # Process the input with the graph
        state = {"input": user_input}
        state = session_manager.process_with_session(session.id, state)
        result = graph.invoke(state)

        # Display the result
        print(f"\nAssistant: {result.get('output', 'No response')}")

    print("\nChat session ended.")


if __name__ == "__main__":
    main()
