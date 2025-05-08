# Multi-Agent Systems

LG-ADK provides powerful abstractions for building multi-agent systems using LangGraph. This guide explains how to create, configure, and use multi-agent systems with LG-ADK.

## What is a Multi-Agent System?

A multi-agent system is a collection of specialized agents working together to accomplish tasks. Each agent has a specific role and expertise, and a coordinator agent orchestrates their interactions and workflow.

LG-ADK's implementation simplifies the creation of multi-agent systems by handling the complex LangGraph orchestration behind the scenes.

## Creating a Multi-Agent System

### Basic Structure

The basic structure of a multi-agent system in LG-ADK includes:

1. A coordinator agent that manages workflow
2. Specialized agents that handle specific tasks
3. A `MultiAgentSystem` instance that orchestrates everything

### Example

Here's a simple example of creating a multi-agent system:

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

summarizer = Agent(
    name="summarizer",
    llm="ollama/llama3",
    description="Summarizes information concisely"
)

# Create the multi-agent system
multi_agent_system = MultiAgentSystem(
    name="research_team",
    coordinator=coordinator,
    agents=[researcher, summarizer],
    description="A team that researches topics and creates summaries"
)

# Run the system
result = multi_agent_system.run({"input": "Tell me about climate change"})
print(result["output"])
```

## How It Works

When you run a multi-agent system, the following happens:

1. The user input is sent to the coordinator agent
2. The coordinator analyzes the request and decides which specialized agent(s) should handle it
3. The coordinator routes the request to the appropriate agent(s)
4. The specialized agent(s) process the request and return results
5. The coordinator compiles the results and provides a final response

Under the hood, LG-ADK creates a LangGraph orchestration graph that handles the message routing and state management.

## Advanced Usage: Conversation History

For multi-turn conversations, LG-ADK provides a `Conversation` class that maintains conversation history:

```python
from lg_adk import Conversation

# Create a conversation handler
conversation = Conversation(multi_agent_system=multi_agent_system)

# First user message
response1 = conversation.send_message("Tell me about climate change")
print(response1)

# Follow-up question (conversation history is maintained)
response2 = conversation.send_message("What are the main mitigation strategies?")
print(response2)
```

## Customizing Agent Behavior

Each agent in the system can be customized with:

- Different language models
- System messages
- Tools (if supported by your implementation)

```python
# Customizing an agent with a specific system message
researcher = Agent(
    name="researcher",
    llm="ollama/llama3",
    description="Researches information and provides detailed answers",
    system_message="""You are a research specialist who provides detailed,
    accurate information. Always cite your sources and provide
    comprehensive answers with evidence-based reasoning."""
)
```

## Scaling with Multiple Agents

You can add any number of specialized agents to your multi-agent system:

```python
# Add agents after creation
critique_agent = Agent(
    name="critique",
    llm="ollama/llama3",
    description="Provides critical analysis and identifies potential biases"
)

multi_agent_system.add_agent(critique_agent)

# Or add multiple agents at once
fact_checker = Agent(...)
source_finder = Agent(...)
multi_agent_system.add_agents([fact_checker, source_finder])
```

## Complete Example

For a complete working example of a multi-agent system, see the [Multi-Agent Example](../examples/multi_agent_example.py) in the examples directory.

## Best Practices

- **Clear Agent Roles**: Give each agent a clear and specific role
- **Descriptive Names**: Use descriptive names for your agents
- **Coordinator Instructions**: The coordinator agent works best when it has a clear understanding of all available agents
- **Model Selection**: Choose appropriate models for each agent based on their tasks
- **Testing**: Test your multi-agent system with a variety of inputs to ensure proper routing
