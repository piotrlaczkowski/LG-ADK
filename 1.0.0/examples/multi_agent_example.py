"""
Multi-Agent System Example with LG-ADK

This example demonstrates how to create a multi-agent system using LG-ADK,
with a coordinator agent that delegates tasks to specialized agents.
"""

from lg_adk import Agent, MultiAgentSystem, get_model

# Create a coordinator agent
coordinator = Agent(name="coordinator", llm="ollama/llama3", description="Coordinates tasks between specialized agents")

# Create specialized agents
researcher = Agent(
    name="researcher", llm="ollama/llama3", description="Researches information and provides detailed answers"
)

summarizer = Agent(name="summarizer", llm="ollama/llama3", description="Summarizes information concisely")

creative_writer = Agent(
    name="creative_writer", llm="ollama/llama3", description="Creates engaging and creative content"
)

# Create the multi-agent system
multi_agent_system = MultiAgentSystem(
    name="research_and_writing_team",
    coordinator=coordinator,
    agents=[researcher, summarizer, creative_writer],
    description="A team that researches topics and creates summaries or creative content",
)

# Run the multi-agent system interactively
if __name__ == "__main__":
    print("Multi-Agent System Example")
    print("==========================")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break

        # Run the multi-agent system
        result = multi_agent_system.run({"input": user_input})

        # Print the response
        print(f"\nSystem: {result.get('output', '')}\n")
