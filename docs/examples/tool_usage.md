# Tool Usage with LG-ADK

This guide demonstrates how to incorporate tools into your agents using LG-ADK. Tools extend your agent's capabilities, allowing them to perform actions like retrieving information, calculating values, or interacting with external systems.

## Basic Tool Integration

### Step 1: Define a Simple Tool

First, let's define a simple calculator tool:

```python
from typing import Dict, Any
from lg_adk.tools.base import BaseTool

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Performs basic arithmetic operations"
    
    def _run(self, operation: str, a: float, b: float) -> Dict[str, Any]:
        """
        Perform a basic arithmetic operation.
        
        Args:
            operation: The operation to perform (add, subtract, multiply, divide)
            a: First number
            b: Second number
            
        Returns:
            Dictionary containing the result
        """
        result = None
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                return {"error": "Cannot divide by zero"}
            result = a / b
        else:
            return {"error": f"Unknown operation: {operation}"}
            
        return {"result": result}
```

### Step 2: Register the Tool with Your Agent

Now let's create an agent that can use this tool:

```python
from lg_adk.agents import Agent
from lg_adk.models import get_model

# Initialize your agent
agent = Agent(
    name="math_assistant",
    model=get_model("openai/gpt-4"),
    system_prompt="You are a helpful math assistant. Use the calculator tool when needed.",
    tools=[CalculatorTool()]
)

# Now the agent can use the calculator tool during conversations
response = agent.run("What is 1234 × 5678?")
print(response)
```

## Example: Web Search Tool

Here's a more complex example using a web search tool:

```python
from lg_adk.tools.web import WebSearchTool
from lg_adk.agents import Agent
from lg_adk.models import get_model

# Initialize an agent with the web search tool
agent = Agent(
    name="research_assistant",
    model=get_model("openai/gpt-4"),
    system_prompt="You are a research assistant. Use the web search tool to find up-to-date information.",
    tools=[WebSearchTool(api_key="your_search_api_key")]
)

# Ask the agent a question that requires searching for information
response = agent.run("What were the major tech news headlines yesterday?")
print(response)
```

## Complete Example: Multi-Tool Agent

This example shows how to create an agent with multiple tools:

```python
import os
from lg_adk.agents import Agent
from lg_adk.models import get_model
from lg_adk.tools.web import WebSearchTool
from lg_adk.tools.file import FileReadTool, FileWriteTool
from lg_adk.tools.memory import MemoryTool

# Setup tools
web_search = WebSearchTool(api_key=os.environ.get("SEARCH_API_KEY"))
file_read = FileReadTool()
file_write = FileWriteTool()
memory = MemoryTool()

# Create an agent with multiple tools
agent = Agent(
    name="assistant",
    model=get_model("openai/gpt-4"),
    system_prompt="""You are a helpful assistant with multiple capabilities:
    - You can search the web for information
    - You can read and write files
    - You can store and retrieve information from your memory
    Use these tools whenever appropriate to help the user.""",
    tools=[web_search, file_read, file_write, memory]
)

# Example interaction
conversation = agent.run("""
Please help me with these tasks:
1. Find the current price of Bitcoin
2. Save that information to a file called 'crypto_prices.txt'
3. Remember that I'm interested in cryptocurrency prices
""")

print(conversation)
```

## Tool Output Processing

LG-ADK automatically handles parsing tool outputs and sending them back to the model. You can also customize how tool outputs are processed:

```python
from lg_adk.agents import Agent
from lg_adk.models import get_model
from lg_adk.tools.base import BaseTool

class CustomTool(BaseTool):
    name = "custom_tool"
    description = "A custom tool with output processing"
    
    def _run(self, query: str) -> dict:
        # Tool implementation
        return {"data": f"Processed: {query}"}
    
    def process_output(self, output: dict) -> str:
        """Custom processing of tool output before sending to model"""
        return f"TOOL RESULT: {output['data'].upper()}"

# Create agent with custom tool
agent = Agent(
    name="custom_agent",
    model=get_model("openai/gpt-4"),
    system_prompt="Use the custom tool when appropriate.",
    tools=[CustomTool()]
)
```

## Tool Error Handling

Tools in LG-ADK include built-in error handling mechanisms:

```python
from lg_adk.tools.base import BaseTool

class RiskyTool(BaseTool):
    name = "risky_tool"
    description = "A tool that might fail"
    
    def _run(self, input_param: str) -> dict:
        try:
            # Some operation that might fail
            if input_param == "fail":
                raise ValueError("Demonstration error")
            return {"result": f"Successfully processed {input_param}"}
        except Exception as e:
            # Error handling
            return {
                "error": str(e),
                "status": "failed"
            }
```

With proper tool design, your agents can handle errors gracefully and provide helpful feedback to users. 