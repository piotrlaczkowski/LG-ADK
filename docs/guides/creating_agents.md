# Creating Agents with LG-ADK

This guide covers how to create agents using the LangGraph Agent Development Kit (LG-ADK).

## Agent Fundamentals

An agent in LG-ADK is an entity that can:
- Process and understand natural language
- Make decisions based on provided information
- Use tools to perform actions
- Maintain state across interactions

## Basic Agent Creation

### Step 1: Import the Required Classes

```python
from lg_adk.agents import Agent
from lg_adk.models import get_model
```

### Step 2: Initialize Your Agent

```python
# Create a simple agent
agent = Agent(
    name="assistant",
    model=get_model("openai/gpt-4"),
    system_prompt="You are a helpful AI assistant that answers user questions."
)
```

The basic parameters for creating an agent are:
- `name`: A unique identifier for your agent
- `model`: The language model the agent will use (from the model registry)
- `system_prompt`: Instructions that define the agent's behavior and capabilities

### Step 3: Using Your Agent

```python
# Run the agent with a user input
response = agent.run("What is the capital of France?")
print(response)
```

## Advanced Agent Configuration

You can configure more advanced settings for your agents:

```python
from lg_adk.tools import WebSearchTool
from lg_adk.memory import MemoryManager
from lg_adk.database import DatabaseManager

# Create a more advanced agent
advanced_agent = Agent(
    name="research_assistant",
    model=get_model("openai/gpt-4"),
    system_prompt="""You are a research assistant that helps users find and summarize information.
    Use your tools when appropriate to provide accurate and up-to-date information.""",
    tools=[WebSearchTool()],
    memory_manager=MemoryManager(
        database_manager=DatabaseManager(connection_string="sqlite:///memory.db")
    ),
    max_tokens=1024,
    temperature=0.7
)
```

## Agent with Custom Behavior

You can customize how your agent processes inputs and outputs:

```python
class CustomAgent(Agent):
    def preprocess_input(self, user_input: str) -> str:
        """Customize how user input is processed before being sent to the model"""
        # Add a timestamp to each input
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp}] User said: {user_input}"
    
    def postprocess_output(self, model_output: str) -> str:
        """Customize how model output is processed before being returned to the user"""
        # Add a signature to each response
        return f"{model_output}\n\n- CustomAgent"

# Initialize the custom agent
custom_agent = CustomAgent(
    name="custom_assistant",
    model=get_model("openai/gpt-4"),
    system_prompt="You are a helpful assistant with a custom personality."
)
```

## Streaming Responses

LG-ADK supports streaming responses from your agent:

```python
# Initialize an agent with streaming enabled
streaming_agent = Agent(
    name="streaming_assistant",
    model=get_model("openai/gpt-4"),
    system_prompt="You are a helpful AI assistant.",
    stream=True
)

# Process a streaming response
for chunk in streaming_agent.stream("Tell me a short story about a robot."):
    print(chunk, end="", flush=True)
```

## Asynchronous Agents

For applications requiring non-blocking operation, use async versions:

```python
import asyncio

async def main():
    # Initialize an async agent
    agent = Agent(
        name="async_assistant",
        model=get_model("openai/gpt-4"),
        system_prompt="You are a helpful AI assistant."
    )
    
    # Run the agent asynchronously
    response = await agent.arun("What is the meaning of life?")
    print(response)
    
    # Stream responses asynchronously
    async for chunk in agent.astream("Tell me about quantum computing."):
        print(chunk, end="", flush=True)

# Run the async function
asyncio.run(main())
```

## Agent with Multiple Models

You can create agents that can switch between different models:

```python
from lg_adk.models import ModelRegistry

# Register multiple models
registry = ModelRegistry()
registry.register("default", "openai/gpt-4")
registry.register("fast", "openai/gpt-3.5-turbo")
registry.register("local", "ollama/llama2")

# Create an agent that can switch models
multi_model_agent = Agent(
    name="adaptive_assistant",
    model=registry.get("default"),
    system_prompt="You are a helpful assistant that adapts to user needs."
)

# Later, switch to a different model
multi_model_agent.model = registry.get("fast")
response = multi_model_agent.run("Give me a quick answer!")
```

## Best Practices

1. **Clear System Prompts**: Define precisely what your agent should and shouldn't do.

2. **Appropriate Tools**: Only give your agent the tools it needs for its specific tasks.

3. **Memory Management**: Configure memory appropriately for your use case.

4. **Error Handling**: Implement proper error handling, especially for tool usage.

5. **Testing**: Test your agents thoroughly with different inputs to ensure they behave as expected.

## Example: Complete Agent Setup

Here's a complete example that brings together various concepts:

```python
from lg_adk.agents import Agent
from lg_adk.models import get_model
from lg_adk.tools import WebSearchTool, CalculatorTool
from lg_adk.memory import MemoryManager
from lg_adk.database import DatabaseManager

# Setup tools
tools = [
    WebSearchTool(),
    CalculatorTool()
]

# Setup memory
memory_manager = MemoryManager(
    database_manager=DatabaseManager(connection_string="sqlite:///agent_memory.db")
)

# Create the agent
research_agent = Agent(
    name="research_assistant",
    model=get_model("openai/gpt-4"),
    system_prompt="""You are a research assistant with these capabilities:
    1. You can search the web for current information
    2. You can perform calculations
    3. You remember previous conversations with the user
    
    Always be helpful, accurate, and concise in your responses.
    When you don't know something, use your tools rather than guessing.
    """,
    tools=tools,
    memory_manager=memory_manager,
    temperature=0.2,
    max_tokens=1024
)

# Use the agent
response = research_agent.run("What was the population of Tokyo in 2022, and what percentage of Japan's total population does that represent?")
print(response)
```

With this guide, you should now have a good understanding of how to create and configure agents using LG-ADK. For more advanced use cases, see the related guides on [Building Graphs](building_graphs.md), [Tool Integration](tool_integration.md), and [Memory Management](memory_management.md). 