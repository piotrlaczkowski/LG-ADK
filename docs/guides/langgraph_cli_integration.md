# LangGraph CLI Integration

This guide explains how to integrate your LG-ADK applications with the LangGraph CLI for development, debugging, and deployment.

## Understanding LangGraph CLI

The LangGraph CLI provides several useful commands for working with LangGraph applications:

- `langgraph dev`: Start a development server for testing your graphs
- `langgraph serve`: Serve your graphs in production
- `langgraph deploy`: Deploy your graphs to the cloud
- `langgraph list`: List available graphs in your project

## Project Configuration

To use the LangGraph CLI with your LG-ADK project, you need a proper configuration file:

### The `langgraph.json` File

Create a `langgraph.json` file in your project root:

```json
{
  "graphs": {
    "chat": "lg_adk.graphs.chat:graph",
    "rag": "lg_adk.graphs.rag:graph",
    "multi_agent": "lg_adk.graphs.multi_agent:graph"
  }
}
```

Each entry specifies:
- A name for the graph (e.g., "chat")
- The import path in the format: `module.path:graph_variable_name`

## Structuring Your Graphs for Discovery

To make your graphs discoverable by the LangGraph CLI, follow these conventions:

### 1. Export Graph Variables

When defining graphs, export them as top-level variables:

```python
# In lg_adk/graphs/chat.py
from langgraph.graph import Graph
from lg_adk.agents import Agent
from lg_adk.builders import GraphBuilder

# Create and build the graph
builder = GraphBuilder(name="chat")
builder.add_agent(Agent(name="assistant", model="openai/gpt-4"))
graph = builder.build()  # This is the variable the CLI will look for
```

### 2. Use Type Annotations

Properly type your graph for better tooling support:

```python
from langgraph.graph import Graph
from typing import Dict, Any, TypedDict

class ChatState(TypedDict):
    messages: list
    session_id: str

# Create typed graph
graph: Graph[ChatState] = builder.build()
```

## Session Management with LangGraph CLI

Proper session handling is crucial for multi-turn conversations with LangGraph CLI:

### Implementing Session-Aware Graphs

Here's how to create graphs that correctly handle session state:

```python
from langgraph.graph import Graph
from typing import Dict, Any, TypedDict
from lg_adk.agents import Agent
from lg_adk.builders import GraphBuilder

# Define state type
class GraphState(TypedDict):
    messages: list
    session_id: str
    metadata: Dict[str, Any]

def build_graph():
    # Create an agent
    agent = Agent(
        name="assistant",
        model="openai/gpt-4",
        system_prompt="You are a helpful assistant."
    )

    # Create graph builder
    builder = GraphBuilder(name="chat")
    builder.add_agent(agent)

    # Configure to track session in state
    builder.configure_state_tracking(
        include_session_id=True,
        include_metadata=True
    )

    # Build the graph
    return builder.build()

# Export the graph for LangGraph CLI
graph = build_graph()
```

### Correct Context Handling

Ensure your agent functions properly handle the checkpointer state:

```python
from langgraph.checkpoint.base import Checkpointer
from lg_adk.agents import Agent
from typing import Dict, Any

class SessionAwareAgent(Agent):
    def run(self, input_text: str, session_id: str = None, config: Dict[str, Any] = None):
        """Run the agent with proper session handling."""
        # Configure for checkpointer
        if not config:
            config = {}

        # Set session ID in config for langgraph checkpointer
        config["configurable"] = {
            "thread_id": session_id
        }

        # Get current state from checkpointer if available
        current_state = None
        if hasattr(self, "graph") and self.graph:
            try:
                current_state = self.graph.get_state(config)
            except Exception:
                current_state = None

        # Process messages based on state
        graph_messages = (
            current_state.values.get("messages", [])
            if current_state and hasattr(current_state, "values")
            else []
        )

        # Initialize with system message if needed
        messages = []
        if self.system_prompt and not graph_messages:
            messages.append({"role": "system", "content": self.system_prompt})

        # Add user message
        messages.append({"role": "user", "content": input_text})

        # Invoke the graph
        result = self.model.generate(
            messages=messages,
            config=config
        )

        return result
```

## Using the Development Server

Start the development server to test your graphs:

```bash
langgraph dev
```

This will:
1. Discover graphs in your `langgraph.json`
2. Start a local server (by default at http://localhost:3000)
3. Provide a web interface for testing your graphs

## Simplified Development with Makefile

Create a `Makefile` in your project root to simplify common tasks:

```makefile
.PHONY: dev serve deploy list install test docs

# Installation
install:
	pip install -e ".[dev]"

# LangGraph CLI commands
dev:
	langgraph dev

serve:
	langgraph serve

deploy:
	langgraph deploy

list:
	langgraph list

# Testing
test:
	pytest tests/

# Documentation
docs:
	mkdocs serve
```

This allows you to use simple commands like:
- `make dev` - Start development server
- `make test` - Run tests
- `make docs` - Start documentation server

## Example Graphs in LG-ADK

LG-ADK includes several example graphs that are ready to use with the LangGraph CLI:

### Chat Graph

The [`lg_adk.graphs.chat`](https://github.com/yourusername/lg-adk/blob/main/lg_adk/graphs/chat.py) module provides a simple chat graph that:

- Processes user messages
- Maintains conversation context across multiple turns
- Uses a single agent to generate responses
- Properly handles session state for the LangGraph CLI

Key features of this implementation:
- Type-annotated state
- Session ID tracking in state
- Proper checkpointer configuration
- Graph exported as a top-level variable

### RAG Graph

The [`lg_adk.graphs.rag`](https://github.com/yourusername/lg-adk/blob/main/lg_adk/graphs/rag.py) module implements a Retrieval-Augmented Generation graph that:

- Retrieves relevant documents based on user queries
- Incorporates document content into the agent's response
- Maintains session and conversation context
- Uses typed state for better tooling support

### Multi-Agent Graph

The [`lg_adk.graphs.multi_agent`](https://github.com/yourusername/lg-adk/blob/main/lg_adk/graphs/multi_agent.py) module provides a sophisticated multi-agent collaboration system:

- Uses a researcher, writer, and critic agent working together
- Implements task-based workflows
- Tracks agent contributions in state
- Properly handles routing between agents
- Maintains session context across the entire workflow

Each of these examples can be run directly with LangGraph CLI using the `langgraph dev` command or imported and extended for your own applications.

## Complete Example: Chat Application with CLI Support

Here's a complete example of a chat application designed to work with LangGraph CLI:

```python
# File: lg_adk/graphs/chat.py
from typing import TypedDict, List, Dict, Any
from pydantic import BaseModel
from langgraph.graph import Graph
from lg_adk.agents import Agent
from lg_adk.models import get_model
from lg_adk.memory import MemoryManager
from lg_adk.builders import GraphBuilder
from lg_adk.sessions import SessionManager

# Define message type
class Message(BaseModel):
    role: str
    content: str

# Define state type
class ChatState(TypedDict):
    messages: List[Message]
    session_id: str
    metadata: Dict[str, Any]

# Create components
agent = Agent(
    name="chat_assistant",
    model=get_model("openai/gpt-4"),
    system_prompt="You are a helpful, friendly assistant."
)

memory_manager = MemoryManager()
session_manager = SessionManager()

# Build graph
builder = GraphBuilder(name="chat")
builder.add_agent(agent)
builder.add_memory(memory_manager)
builder.configure_state_tracking(
    include_session_id=True,
    include_metadata=True
)

# Define state handlers
def add_message_to_state(state: ChatState, message: Message) -> ChatState:
    """Add a message to the state."""
    messages = state.get("messages", [])
    return {
        **state,
        "messages": messages + [message]
    }

# Configure message processing
builder.on_message(add_message_to_state)

# Build and export the graph
graph: Graph[ChatState] = builder.build()
```

When you run `langgraph dev`, this graph will be available for testing via the web interface.

## Advanced: Custom State Persistence

For applications requiring custom state persistence:

```python
from lg_adk.database import DatabaseManager
from lg_adk.sessions import DatabaseSessionManager

# Create a database-backed session manager
db_manager = DatabaseManager(connection_string="postgresql://user:pass@localhost:5432/db")
session_manager = DatabaseSessionManager(database_manager=db_manager)

# Configure the builder to use this session manager
builder = GraphBuilder(name="persistent_chat")
builder.add_agent(agent)
builder.configure_session_management(session_manager)
builder.build()
```

## Best Practices for LangGraph CLI Integration

1. **Typed Graphs**: Always use type hints for graph states
2. **State Immutability**: Treat graph states as immutable to avoid unexpected behaviors
3. **Session IDs**: Generate consistent session IDs for multi-turn conversations
4. **Error Handling**: Add proper error handling for state persistence
5. **Development/Production Split**: Use separate configurations for development and production
6. **Environment Variables**: Use environment variables for sensitive configuration
7. **Testing Support**: Create test utilities for your graphs

By following these guidelines, you can build LG-ADK applications that work seamlessly with the LangGraph CLI tools for development, debugging, and deployment.
