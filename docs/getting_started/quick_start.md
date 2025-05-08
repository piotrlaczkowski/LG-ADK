# Quick Start Guide

This guide will help you get started with LG-ADK and build your first agent in minutes.

## Installation

First, install LG-ADK:

```bash
pip install lg-adk
```

Or with Poetry:

```bash
poetry add lg-adk
```

## Creating a Simple Agent

Let's create a simple assistant agent:

```python
from lg_adk import Agent

# Create a simple agent
agent = Agent(
    name="assistant",
    llm="ollama/llama3",  # You can use "gemini/gemini-pro" or other models
    description="A helpful assistant that answers questions"
)

# Run the agent with a user query
result = agent.run({"input": "What is artificial intelligence?"})
print(result["output"])
```

## Building an Agent with a Graph

For more complex workflows, you can use a graph builder:

```python
from lg_adk import Agent, GraphBuilder

# Create an agent
agent = Agent(
    name="assistant",
    llm="ollama/llama3",
    description="A helpful assistant that answers questions"
)

# Create a graph builder
builder = GraphBuilder()
builder.add_agent(agent)

# Build the graph
graph = builder.build()

# Run the graph
result = graph.invoke({"input": "What is machine learning?"})
print(result["output"])
```

## Creating a Multi-Agent System

For complex tasks, you can create a multi-agent system:

```python
from lg_adk import Agent, MultiAgentSystem

# Create a coordinator agent
coordinator = Agent(
    name="coordinator",
    llm="ollama/llama3",
    description="Coordinates tasks between specialized agents"
)

# Create specialized agents
researcher = Agent(
    name="researcher",
    llm="ollama/llama3",
    description="Researches information and provides detailed answers"
)

writer = Agent(
    name="writer",
    llm="ollama/llama3",
    description="Writes concise and clear content"
)

# Create a multi-agent system
system = MultiAgentSystem(
    name="research_team",
    coordinator=coordinator,
    agents=[researcher, writer],
    description="A team that researches and writes about topics"
)

# Run the system
result = system.run({"input": "Explain quantum computing"})
print(result["output"])
```

## Using Different Model Providers

LG-ADK supports multiple model providers:

```python
# Using Ollama (local models)
local_agent = Agent(
    name="local_assistant",
    llm="ollama/llama3",
    description="An assistant running on a local model"
)

# Using Google's Gemini models
gemini_agent = Agent(
    name="gemini_assistant",
    llm="gemini/gemini-pro",
    description="An assistant powered by Gemini"
)

# Using OpenAI models
openai_agent = Agent(
    name="openai_assistant",
    llm="openai/gpt-4",
    description="An assistant powered by GPT-4"
)
```

## Evaluating Your Agent

LG-ADK includes tools for evaluating agent performance:

```python
from lg_adk import Agent, EvalDataset, Evaluator

# Create an agent
agent = Agent(
    name="assistant",
    llm="ollama/llama3",
    description="A helpful assistant"
)

# Create or load an evaluation dataset
dataset = EvalDataset(
    name="Simple Questions",
    description="Basic knowledge questions",
    examples=[
        {
            "id": "q1",
            "input": "What is the capital of France?",
            "expected_output": "The capital of France is Paris."
        },
        {
            "id": "q2",
            "input": "Who wrote Romeo and Juliet?",
            "expected_output": "William Shakespeare wrote Romeo and Juliet."
        }
    ]
)

# Create an evaluator and run evaluation
evaluator = Evaluator()
results = evaluator.evaluate(agent, dataset)

# Print evaluation results
print(f"Accuracy: {results.metric_scores.get('AccuracyMetric', 0)}")
print(f"Latency: {results.metric_scores.get('LatencyMetric', 0)} seconds")
```

## Running an Interactive Session

You can use the built-in CLI to run an interactive session:

```bash
# Run an agent interactively
lg-adk run path/to/your_agent.py

# Evaluate an agent against a dataset
lg-adk eval path/to/your_agent.py path/to/dataset.json

# Debug an agent visually (requires langgraph-cli)
lg-adk debug path/to/your_agent.py
```

## Next Steps

Now that you've seen the basics, check out these resources:

- [Agent Guide](../guides/basic_agents.md) - Learn more about creating and customizing agents
- [Multi-Agent Systems](../guides/multi_agent.md) - Explore building complex multi-agent systems
- [Examples](../examples/) - Browse complete code examples
- [API Reference](../reference/core_concepts.md) - Detailed API documentation
