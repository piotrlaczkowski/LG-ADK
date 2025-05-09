"""
Graph Builder Example with LG-ADK

This example demonstrates how to create a graph using the GraphBuilder,
with two agents that process the input sequentially.
"""

from lg_adk import Agent, GraphBuilder
from lg_adk.memory import MemoryManager

# Create agents for different stages of processing
analyzer = Agent(
    name="analyzer", llm="ollama/llama3", description="Analyzes the input and breaks it down into components"
)

responder = Agent(name="responder", llm="ollama/llama3", description="Generates a final response based on the analysis")

# Create a graph builder
builder = GraphBuilder()

# Add the agents to the graph
builder.add_agent(analyzer)
builder.add_agent(responder)

# Add memory manager for keeping track of conversation history
memory_manager = MemoryManager()
builder.add_memory(memory_manager)

# Enable human-in-the-loop for interactive correction if needed
builder.enable_human_in_loop()

# Build the graph with a specific flow: analyzer -> responder
graph = builder.build(
    flow=[
        ("analyzer", "responder"),  # analyzer output goes to responder
        ("responder", None),  # responder output is the final output
    ]
)

# Run the graph interactively
if __name__ == "__main__":
    print("Graph Example")
    print("=============")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break

        # Run the graph
        result = graph.invoke({"input": user_input})

        # Print the response
        print(f"\nSystem: {result.get('output', '')}\n")
