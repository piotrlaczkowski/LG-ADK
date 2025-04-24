# Memory Management in LG-ADK

This guide explains how to implement and use memory management in the LangGraph Agent Development Kit (LG-ADK), allowing agents to store, retrieve, and utilize information across conversations.

## Understanding Memory in LG-ADK

Memory in LG-ADK enables agents and graphs to:

1. Maintain conversation context across multiple interactions
2. Store important information for later retrieval
3. Build knowledge bases specific to user sessions
4. Enable collaborative work between multiple agents

The primary components of the memory system are:

- **MemoryManager**: Orchestrates memory operations
- **DatabaseManager**: Handles the underlying storage
- **Memory Tools**: Enable agents to interact with memory

## Setting Up Memory

To use memory in your agents and graphs, start by setting up the memory components:

```python
from lg_adk.memory import MemoryManager
from lg_adk.database import DatabaseManager

# Create a database manager
db_manager = DatabaseManager(
    connection_string="sqlite:///agent_memory.db"
)

# Create a memory manager
memory_manager = MemoryManager(
    database_manager=db_manager
)
```

The `connection_string` can be configured for different database types:

- SQLite (local): `"sqlite:///agent_memory.db"`
- PostgreSQL: `"postgresql://username:password@localhost:5432/db_name"`
- MySQL: `"mysql://username:password@localhost:3306/db_name"`

## Adding Memory to Agents

To equip an agent with memory capabilities:

```python
from lg_adk.agents import Agent
from lg_adk.models import get_model
from lg_adk.tools import MemoryTool

# Create a memory tool
memory_tool = MemoryTool(memory_manager=memory_manager)

# Add to an agent
agent = Agent(
    name="assistant",
    model=get_model("openai/gpt-4"), 
    system_prompt="You are a helpful assistant that remembers information.",
    tools=[memory_tool]
)
```

The agent can now use the memory tool to store and retrieve information:

```python
# The agent will use the memory tool to store this information
agent.run("Remember that the user's favorite color is blue.")

# Later, the agent can retrieve this information
agent.run("What is the user's favorite color?")
```

## Adding Memory to Graphs

To add memory capabilities to a graph:

```python
from lg_adk.builders import GraphBuilder

# Create agents
researcher = Agent(
    name="researcher",
    model=get_model("openai/gpt-4"),
    system_prompt="You are a research agent that finds information.",
    tools=[MemoryTool(memory_manager=memory_manager)]
)

writer = Agent(
    name="writer",
    model=get_model("openai/gpt-4"),
    system_prompt="You are a writer that summarizes information.",
    tools=[MemoryTool(memory_manager=memory_manager)]
)

# Create a graph with shared memory
builder = GraphBuilder()
builder.add_agent(researcher)
builder.add_agent(writer)
builder.add_memory(memory_manager)

# Connect agents
builder.add_edge(researcher, writer)

# Build the graph
graph = builder.build()
```

With this setup, both the researcher and writer agents can share information through memory.

## Memory Operations

The MemoryTool provides several operations for agents to use:

### Storing Information

Agents can store information in memory:

```python
# Example of how an agent would use the memory tool to store information
agent.run("""
Use the memory tool to store the following information:
The capital of France is Paris.
""")
```

The memory tool will generate a unique memory ID, store the content with metadata, and return the ID to the agent.

### Retrieving Information

Agents can retrieve information from memory:

```python
# Example of how an agent would retrieve information from memory
agent.run("What is the capital of France?")
```

The agent will use the memory tool to search for relevant information and incorporate it into its response.

### Retrieving by Tags

Agents can use tags to organize and retrieve related information:

```python
# Example of how an agent would store information with tags
agent.run("""
Store this information with the tags 'geography', 'europe':
France is a country in Western Europe.
""")

# Later, retrieve information by tags
agent.run("Tell me about European geography.")
```

### Deleting Information

Agents can delete information when it's no longer needed:

```python
# Example of how an agent would delete a specific memory by ID
agent.run("Delete the information about France's capital.")
```

## Session-Based Memory

LG-ADK memory is session-based, allowing for separate memory contexts for different users or conversations:

```python
# Create a session for a specific user
session_id = "user_123"

# Run the agent or graph with the session ID
result = agent.run("Remember my name is Alice.", session_id=session_id)

# Later, continue the same session
result = agent.run("What's my name?", session_id=session_id)
```

Different sessions won't share memories, ensuring privacy and context separation.

## Memory Schemas and Structure

Memory in LG-ADK is stored as documents with the following structure:

```python
{
    "id": "unique_memory_id",
    "session_id": "session_identifier",
    "content": "The actual information stored",
    "tags": ["tag1", "tag2"],
    "metadata": {
        "source": "user_input",
        "importance": "high",
        # Any additional metadata
    },
    "timestamp": "2023-10-23T14:30:00Z"
}
```

You can customize how agents use these fields through their system prompts.

## Advanced Memory Patterns

### Collaborative Memory Use

Multiple agents can collaborate using shared memory:

```python
# First agent stores information
agent1.run("Store that the meeting is scheduled for Tuesday at 3 PM.", session_id="team_project")

# Second agent accesses the same information
agent2.run("When is our next meeting?", session_id="team_project")
```

### Memory Prioritization

Guide agents to prioritize certain memories:

```python
# Store information with importance metadata
agent.run("""
Store this information with metadata importance=high:
The client deadline is October 30th.
""")

# Update the agent's system prompt to use importance
agent.system_prompt = """You are a helpful assistant.
When retrieving memories, prioritize those with high importance.
"""
```

### Memory Summarization

Implement periodic memory summarization:

```python
from lg_adk.agents import Agent
from lg_adk.models import get_model

# Create a summarizer agent
summarizer = Agent(
    name="memory_summarizer",
    model=get_model("openai/gpt-4"),
    system_prompt="""You summarize memories into concise knowledge.
    Create a single paragraph summary of all related memories.""",
    tools=[MemoryTool(memory_manager=memory_manager)]
)

# Function to periodically summarize memories
def summarize_memories(session_id):
    # Retrieve all memories for the session
    all_memories = memory_manager.retrieve(session_id=session_id)
    
    # Run the summarizer agent
    summary = summarizer.run(f"Summarize the following memories: {all_memories}")
    
    # Store the summary as a new memory with a special tag
    memory_manager.store(
        session_id=session_id,
        content=summary,
        tags=["summary"],
        metadata={"type": "memory_summary"}
    )
```

## Memory Persistence

LG-ADK memory is persistent across application restarts. To ensure data isn't lost:

1. Use a production-grade database for the `DatabaseManager`
2. Implement regular database backups
3. Consider database migration strategies for version upgrades

## Complete Example: Question-Answering with Memory

Here's a complete example of a question-answering agent with memory:

```python
from lg_adk.agents import Agent
from lg_adk.models import get_model
from lg_adk.memory import MemoryManager
from lg_adk.database import DatabaseManager
from lg_adk.tools import MemoryTool, WebSearchTool

# Setup memory
db_manager = DatabaseManager(connection_string="sqlite:///qa_system.db")
memory_manager = MemoryManager(database_manager=db_manager)

# Create tools
memory_tool = MemoryTool(memory_manager=memory_manager)
search_tool = WebSearchTool()

# Create the QA agent
qa_agent = Agent(
    name="qa_system",
    model=get_model("openai/gpt-4"),
    system_prompt="""You are a question-answering assistant that learns from interactions.
    
    When a user asks a question:
    1. Check your memory for relevant information first
    2. If the answer isn't in memory, use web search to find it
    3. Store any new information you learn in memory for future reference
    4. Answer the user's question accurately and concisely
    
    Always cite your sources, whether from memory or search results.""",
    tools=[memory_tool, search_tool]
)

# Example usage with a session
session_id = "user_session_123"

# First interaction - agent will search and remember
result1 = qa_agent.run("What is the capital of Canada?", session_id=session_id)
print(result1)

# Second interaction - agent will use memory
result2 = qa_agent.run("What's the capital city of Canada again?", session_id=session_id)
print(result2)

# New question - agent will search again
result3 = qa_agent.run("What's the population of Toronto?", session_id=session_id)
print(result3)
```

## Best Practices for Memory Management

1. **Session Management**: Use consistent session IDs to maintain conversation context
2. **Memory Cleanup**: Implement policies for removing outdated or unused memories
3. **Privacy Considerations**: Be transparent about data storage and implement retention policies
4. **Tagging Strategy**: Develop a consistent tagging strategy for easy information retrieval
5. **Memory Validation**: Consider validating information before storing it in memory
6. **Performance Optimization**: Index frequently accessed memory fields for faster retrieval
7. **Memory Monitoring**: Implement monitoring to track memory usage and growth

By following this guide, you can effectively implement and utilize memory management in your LG-ADK applications, enabling more contextually aware and helpful agent interactions. For more information on related topics, see the [Building Graphs](building_graphs.md), [Creating Agents](creating_agents.md), and [Tool Integration](tool_integration.md) guides. 