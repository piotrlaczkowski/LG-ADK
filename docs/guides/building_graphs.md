# Building Graphs with LG-ADK

This guide covers how to build agent graphs using the LangGraph Agent Development Kit (LG-ADK).

## Understanding Graph Architecture

In LG-ADK, a graph is a collection of connected agents and components that work together to accomplish complex tasks. Graphs allow you to:

1. Connect multiple agents in a workflow
2. Define how information flows between agents
3. Create conditional logic between agent interactions
4. Maintain state throughout the entire process

## Getting Started with GraphBuilder

The `GraphBuilder` class is your main tool for constructing agent graphs:

```python
from lg_adk.builders import GraphBuilder
from lg_adk.agents import Agent
from lg_adk.models import get_model

# Create a new graph builder
builder = GraphBuilder(name="simple_graph")
```

## Adding Agents to a Graph

You can add pre-configured agents to your graph:

```python
# Create some agents
researcher = Agent(
    name="researcher",
    model=get_model("openai/gpt-4"),
    system_prompt="You are a research agent who finds information on topics."
)

writer = Agent(
    name="writer",
    model=get_model("openai/gpt-4"),
    system_prompt="You are a writer agent who creates content based on research."
)

# Add agents to the graph
builder.add_agent(researcher)
builder.add_agent(writer)
```

## Defining Node Connections

Connect nodes to establish the flow of information:

```python
# Connect the researcher to the writer
builder.connect(
    source="researcher",
    target="writer"
)

# Connect the writer back to the user (end of the graph)
builder.connect(
    source="writer",
    target="__end__"  # Special node that represents the end of the graph
)
```

## Conditional Routing

You can create more complex flows with conditional routing:

```python
# Define a routing function
def route_based_on_complexity(state):
    """Route to different agents based on query complexity"""
    complexity = state.get("complexity", "simple")
    if complexity == "complex":
        return "deep_researcher"
    else:
        return "basic_researcher"

# Add a conditional branch
builder.add_conditional_edge(
    source="user_input",
    condition_function=route_based_on_complexity,
    targets=["basic_researcher", "deep_researcher"]
)
```

## Adding Memory to a Graph

Memory allows your graph to maintain context across interactions:

```python
from lg_adk.memory import MemoryManager
from lg_adk.database import DatabaseManager

# Create memory manager
memory_manager = MemoryManager(
    database_manager=DatabaseManager(connection_string="sqlite:///graph_memory.db")
)

# Add memory to the graph
builder.add_memory(memory_manager)
```

## Human-in-the-Loop Integration

For tasks that require human oversight:

```python
# Enable human-in-the-loop for the entire graph
builder.enable_human_in_the_loop()

# Or for specific transitions
builder.enable_human_in_the_loop(
    source="critical_decision",
    target="high_impact_action"
)
```

## Building and Running the Graph

Once you've configured your graph, build and run it:

```python
# Build the graph
graph = builder.build()

# Run the graph with an initial input
result = graph.run("Research the latest advancements in quantum computing and write a summary.")
print(result)
```

## Streaming Graph Output

For long-running processes, you might want to stream the results:

```python
# Enable streaming for the graph
graph = builder.build(stream=True)

# Stream the results
for chunk in graph.stream("Tell me about artificial intelligence."):
    print(chunk, end="", flush=True)
```

## Asynchronous Graph Execution

For non-blocking operation:

```python
import asyncio

async def main():
    # Build the graph with async support
    graph = builder.build()

    # Run the graph asynchronously
    result = await graph.arun("Analyze the current trends in renewable energy.")
    print(result)

    # Stream results asynchronously
    async for chunk in graph.astream("Explain machine learning concepts."):
        print(chunk, end="", flush=True)

# Run the async function
asyncio.run(main())
```

## Advanced Graph Patterns

### Parallel Processing

Execute multiple agents in parallel:

```python
# Create a parallel processing section
builder.add_parallel_nodes(
    ["market_researcher", "technical_researcher", "social_researcher"]
)

# Add a node to combine the parallel results
builder.add_agent(combiner)

# Connect the parallel nodes to the combiner
for node in ["market_researcher", "technical_researcher", "social_researcher"]:
    builder.connect(source=node, target="combiner")
```

### Iterative Refinement

Create loops for iterative improvement:

```python
# Create a drafting loop
builder.connect(source="writer", target="editor")
builder.connect(source="editor", target="quality_check")

# Add a conditional to either continue refining or finish
def is_quality_sufficient(state):
    quality_score = state.get("quality_score", 0)
    if quality_score >= 8:
        return "complete"
    else:
        return "writer"  # Back to writer for another draft

builder.add_conditional_edge(
    source="quality_check",
    condition_function=is_quality_sufficient,
    targets=["writer", "complete"]
)
```

## Example: Complete Research Assistant Graph

Here's a complete example of a research assistant graph:

```python
from lg_adk.builders import GraphBuilder
from lg_adk.agents import Agent
from lg_adk.models import get_model
from lg_adk.tools import WebSearchTool
from lg_adk.memory import MemoryManager
from lg_adk.database import DatabaseManager

# Setup agents
researcher = Agent(
    name="researcher",
    model=get_model("openai/gpt-4"),
    system_prompt="""You are a research agent. Your job is to find information about a given topic.
    Use your web search tool to gather relevant information.""",
    tools=[WebSearchTool()]
)

analyzer = Agent(
    name="analyzer",
    model=get_model("openai/gpt-4"),
    system_prompt="""You are an analysis agent. Your job is to analyze the research provided
    and extract key insights. Focus on what's most important and relevant."""
)

writer = Agent(
    name="writer",
    model=get_model("openai/gpt-4"),
    system_prompt="""You are a content writing agent. Your job is to take analyzed research
    and create a well-structured, informative article about the topic."""
)

# Setup memory
memory_manager = MemoryManager(
    database_manager=DatabaseManager(connection_string="sqlite:///research_graph.db")
)

# Create graph builder
builder = GraphBuilder(name="research_assistant")

# Add agents
builder.add_agent(researcher)
builder.add_agent(analyzer)
builder.add_agent(writer)

# Add memory
builder.add_memory(memory_manager)

# Connect agents
builder.connect(source="__start__", target="researcher")  # Start with researcher
builder.connect(source="researcher", target="analyzer")   # Send research to analyzer
builder.connect(source="analyzer", target="writer")       # Send analysis to writer
builder.connect(source="writer", target="__end__")        # End with writer's output

# Build the graph
graph = builder.build()

# Run the graph
result = graph.run("What are the latest advancements in fusion energy research?")
print(result)
```

## Best Practices

1. **Plan Your Graph**: Sketch out the flow before implementation to visualize node relationships.

2. **Name Agents Clearly**: Use descriptive names that indicate function.

3. **Keep Prompts Focused**: Each agent should have a clear, specific role.

4. **Test Incremental Builds**: Add and test a few nodes at a time rather than building the entire graph at once.

5. **Use Conditional Routing**: Leverage conditional logic to create more adaptive workflows.

6. **Manage State Carefully**: Be mindful of what information needs to be passed between nodes.

7. **Include Error Handling**: Add error handling nodes or conditional paths for unexpected scenarios.

8. **Monitor Performance**: Track execution time and success rates to optimize your graph.

With these techniques, you can build powerful, multi-agent systems that can tackle complex, multi-stage tasks with LG-ADK. For more details on specific components, refer to the [Creating Agents](creating_agents.md), [Tool Integration](tool_integration.md), and [Memory Management](memory_management.md) guides.
