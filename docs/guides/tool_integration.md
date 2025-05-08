# Tool Integration in LG-ADK

This guide covers how to integrate and use tools with agents in the LangGraph Agent Development Kit (LG-ADK).

## Understanding Tools in LG-ADK

Tools give agents the ability to interact with external systems and perform actions beyond just generating text. In LG-ADK, tools:

1. Extend agent capabilities by enabling interaction with systems, APIs, and data sources
2. Allow agents to retrieve information or perform actions in the real world
3. Can be synchronous or asynchronous
4. Can be shared across multiple agents in a graph

## Built-in Tools

LG-ADK comes with several built-in tools ready for immediate use:

```python
from lg_adk.tools import (
    WebSearchTool,
    MemoryTool,
    DelegationTool,
    UserInfoTool,
    FileReaderTool,
    FileWriterTool
)
```

### Web Search Tool

Allows agents to search the internet for information:

```python
from lg_adk.agents import Agent
from lg_adk.models import get_model
from lg_adk.tools import WebSearchTool

agent = Agent(
    name="researcher",
    model=get_model("openai/gpt-4"),
    system_prompt="You are a research assistant that finds information.",
    tools=[WebSearchTool()]
)

# The agent can now use the web search tool
result = agent.run("What are the latest developments in quantum computing?")
```

### Memory Tool

Enables agents to store and retrieve information:

```python
from lg_adk.tools import MemoryTool
from lg_adk.memory import MemoryManager
from lg_adk.database import DatabaseManager

# Create memory manager
memory_manager = MemoryManager(
    database_manager=DatabaseManager(connection_string="sqlite:///agent_memory.db")
)

# Create memory tool
memory_tool = MemoryTool(memory_manager=memory_manager)

# Add to agent
agent = Agent(
    name="memory_agent",
    model=get_model("openai/gpt-4"),
    system_prompt="You are an assistant that remembers important information.",
    tools=[memory_tool]
)
```

### Delegation Tool

Allows one agent to delegate tasks to another:

```python
from lg_adk.tools import DelegationTool

# Create a specialized math agent
math_agent = Agent(
    name="math_expert",
    model=get_model("openai/gpt-4"),
    system_prompt="You are a mathematics expert who can solve complex problems."
)

# Create the delegation tool pointing to the math agent
delegation_tool = DelegationTool(target_agent=math_agent)

# Add to the main agent
primary_agent = Agent(
    name="coordinator",
    model=get_model("openai/gpt-4"),
    system_prompt="You coordinate and delegate specialized tasks when needed.",
    tools=[delegation_tool]
)
```

## Creating Custom Tools

You can create custom tools by extending the `BaseTool` class:

```python
from typing import Dict, Any, Optional
from lg_adk.tools import BaseTool

class WeatherTool(BaseTool):
    """Tool for getting current weather conditions."""

    name: str = "weather_tool"
    description: str = "Get weather information for a specified location."

    def _run(self, location: str) -> Dict[str, Any]:
        """Get weather for a location."""
        # In a real implementation, you would call a weather API
        # This is a simplified example
        weather_data = self._call_weather_api(location)
        return {
            "temperature": weather_data["temp"],
            "conditions": weather_data["conditions"],
            "location": location
        }

    async def _arun(self, location: str) -> Dict[str, Any]:
        """Async version of _run."""
        # Implement async API call here
        return await self._async_call_weather_api(location)

    def _call_weather_api(self, location: str) -> Dict[str, Any]:
        """Mock API call to weather service."""
        # In a real implementation, this would be an actual API call
        return {"temp": "72°F", "conditions": "Sunny"}

    async def _async_call_weather_api(self, location: str) -> Dict[str, Any]:
        """Mock async API call to weather service."""
        # In a real implementation, this would be an actual async API call
        return {"temp": "72°F", "conditions": "Sunny"}
```

## Tool Parameters and Schema

Define the expected input parameters for a tool:

```python
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from lg_adk.tools import BaseTool

class StockLookupParams(BaseModel):
    """Parameters for stock lookup tool."""
    symbol: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    timeframe: str = Field("1d", description="Timeframe for stock data (1d, 5d, 1m, etc.)")

class StockLookupTool(BaseTool):
    """Tool for looking up stock information."""

    name: str = "stock_lookup"
    description: str = "Get stock price and information for a ticker symbol."
    parameters_schema: type = StockLookupParams

    def _run(self, symbol: str, timeframe: str = "1d") -> Dict[str, Any]:
        """Look up stock information."""
        # In a real implementation, call a financial API
        return {
            "symbol": symbol,
            "price": "150.00",
            "change": "+2.5%",
            "timeframe": timeframe
        }
```

## Tool Results Schema

Define the expected output from a tool:

```python
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from lg_adk.tools import BaseTool

class WeatherResult(BaseModel):
    """Weather data result schema."""
    temperature: str = Field(..., description="Current temperature")
    conditions: str = Field(..., description="Weather conditions (Sunny, Rainy, etc.)")
    humidity: Optional[str] = Field(None, description="Current humidity percentage")
    wind_speed: Optional[str] = Field(None, description="Current wind speed")

class DetailedWeatherTool(BaseTool):
    """Tool for getting detailed weather information."""

    name: str = "detailed_weather"
    description: str = "Get comprehensive weather data for a location."
    result_schema: type = WeatherResult

    def _run(self, location: str) -> Dict[str, Any]:
        """Get detailed weather for a location."""
        # Implementation would call a weather API
        return {
            "temperature": "72°F",
            "conditions": "Sunny",
            "humidity": "45%",
            "wind_speed": "5 mph"
        }
```

## Stateful Tools

Create tools that maintain state between invocations:

```python
from typing import Dict, Any, List, Optional
from lg_adk.tools import BaseTool

class ConversationTrackingTool(BaseTool):
    """Tool that tracks conversation history."""

    name: str = "conversation_tracker"
    description: str = "Track and analyze conversation patterns."

    def __init__(self):
        super().__init__()
        self.conversation_history = []

    def _run(self, message: str, sender: str) -> Dict[str, Any]:
        """Track a message in the conversation."""
        self.conversation_history.append({
            "message": message,
            "sender": sender,
            "timestamp": self._get_current_time()
        })

        return {
            "history_length": len(self.conversation_history),
            "summary": self._analyze_conversation()
        }

    def _analyze_conversation(self) -> str:
        """Analyze conversation patterns."""
        # Implement conversation analysis logic
        return f"Conversation has {len(self.conversation_history)} messages."

    def _get_current_time(self) -> str:
        """Get current time string."""
        from datetime import datetime
        return datetime.now().isoformat()
```

## Error Handling in Tools

Implement robust error handling:

```python
from typing import Dict, Any, Optional
from lg_adk.tools import BaseTool

class DatabaseQueryTool(BaseTool):
    """Tool for querying a database."""

    name: str = "database_query"
    description: str = "Query a database for information."

    def _run(self, query: str) -> Dict[str, Any]:
        """Execute a database query."""
        try:
            # Attempt to execute the query
            result = self._execute_query(query)
            return {
                "success": True,
                "results": result,
                "row_count": len(result)
            }
        except Exception as e:
            # Handle the error gracefully
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }

    def _execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute the actual database query."""
        # In a real implementation, this would query a database
        # For demonstration, we'll return mock data
        if "invalid" in query.lower():
            raise ValueError("Invalid SQL query syntax")
        return [{"id": 1, "name": "Example"}, {"id": 2, "name": "Test"}]
```

## Combining Multiple Tools

Add multiple tools to an agent:

```python
from lg_adk.agents import Agent
from lg_adk.models import get_model
from lg_adk.tools import WebSearchTool, MemoryTool, FileReaderTool

# Create agent with multiple tools
assistant = Agent(
    name="multi_tool_assistant",
    model=get_model("openai/gpt-4"),
    system_prompt="You are a helpful assistant with multiple capabilities.",
    tools=[
        WebSearchTool(),
        MemoryTool(memory_manager=memory_manager),
        FileReaderTool()
    ]
)

# The agent can now use any of these tools
result = assistant.run("Search for information about climate change, save the important points to memory, and then read the local report.txt file.")
```

## Tool Security Best Practices

When implementing tools, follow these security best practices:

1. **Validate Inputs**: Always validate inputs to prevent injection attacks.
2. **Limit Permissions**: Give tools only the minimum permissions needed for their function.
3. **Avoid Sensitive Data**: Never expose API keys or credentials directly in the tool code.
4. **Rate Limiting**: Implement rate limiting to prevent abuse of external APIs.
5. **Sanitize Outputs**: Clean tool outputs before returning them to the agent.
6. **Audit Tool Usage**: Log tool invocations for security auditing.

## Complete Example: Research Assistant with Tools

Here's a complete example of an agent with multiple tools:

```python
from lg_adk.agents import Agent
from lg_adk.models import get_model
from lg_adk.tools import WebSearchTool, MemoryTool, FileWriterTool
from lg_adk.memory import MemoryManager
from lg_adk.database import DatabaseManager

# Setup memory manager
memory_manager = MemoryManager(
    database_manager=DatabaseManager(connection_string="sqlite:///research_assistant.db")
)

# Create tools
web_search = WebSearchTool()
memory_tool = MemoryTool(memory_manager=memory_manager)
file_writer = FileWriterTool()

# Create the research assistant agent
researcher = Agent(
    name="research_assistant",
    model=get_model("openai/gpt-4"),
    system_prompt="""You are a research assistant that helps users find information.
    Use your web search tool to find information on the internet.
    Store important findings in memory for later retrieval.
    Write comprehensive reports to files when requested.""",
    tools=[web_search, memory_tool, file_writer]
)

# Run the agent with a research task
result = researcher.run("""
Research the latest advancements in renewable energy storage technologies.
Save important findings to memory and create a comprehensive report in a file called 'renewable_energy_report.txt'.
""")

print(result)
```

By following this guide, you can effectively integrate tools into your agents, enhancing their capabilities and enabling them to perform a wide range of tasks beyond simple text generation. For more information on specific components, refer to the [Creating Agents](creating_agents.md), [Building Graphs](building_graphs.md), and [Memory Management](memory_management.md) guides.
