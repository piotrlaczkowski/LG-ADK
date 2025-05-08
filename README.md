# LG-ADK - LangGraph Agent Development Kit

<p align="center">
  <img src="logo.png" width="350"/>
</p>

A Python development kit designed to simplify the creation of LangGraph-based agents, providing an experience similar to Google's Agent Development Kit.

## Features

- **Modular Agent Architecture**: Easily define and customize agents with different capabilities
- **Flexible Graph Construction**: Build complex agent workflows using LangGraph's powerful graph-based approach
- **Multi-Agent Collaboration**: Use group chats and routers for sophisticated agent interactions
- **Retrieval-Augmented Generation**: Simplified interfaces for creating RAG applications
- **Memory Management**: Built-in support for short-term and long-term memory
- **Session Management**: Handle conversations and maintain context across interactions
- **Human-in-the-Loop Capabilities**: Seamlessly integrate human feedback and intervention
- **Tool Integration**: Easily connect agents to external tools and APIs
- **Local Model Support**: Run with Ollama for enhanced privacy and reduced costs
- **Streaming Responses**: Real-time streaming of agent responses
- **Visual Debugging**: Inspect and debug agent workflows with langgraph-cli
- **Database Flexibility**: Use various databases (local or PostgreSQL) for storage
- **Vector Store Integration**: Works with different vector stores for semantic search

## Installation

```bash
pip install lg-adk
```

Or with Poetry:

```bash
poetry add lg-adk
```

## Quick Start

### Basic Agent

```python
from lg_adk import Agent, GraphBuilder
from lg_adk.memory import MemoryManager
from lg_adk.tools import WebSearchTool

# Create an agent
agent = Agent(
    agent_name="research_assistant",
    llm="gpt-3.5-turbo",  # Or use Ollama: llm="ollama/llama3"
    system_prompt="You are a research assistant that searches the web and answers questions"
)

# Add tools to the agent
agent.add_tool(WebSearchTool())

# Create a graph with the agent
builder = GraphBuilder()
builder.add_agent(agent)
builder.add_memory(MemoryManager())

# Build and run the graph
graph = builder.build()
response = graph.invoke({"input": "What are the latest developments in AI?"})
print(response)
```

### RAG (Retrieval-Augmented Generation)

```python
from lg_adk import Agent, get_model
from lg_adk.tools.retrieval import SimpleVectorRetrievalTool

# Create a retrieval tool
retrieval_tool = SimpleVectorRetrievalTool(
    name="knowledge_base",
    description="Use this to retrieve information from the knowledge base",
    vector_store=your_vector_store,  # Any LangChain-compatible vector store
    top_k=5
)

# Create a RAG agent
rag_agent = Agent(
    agent_name="rag_assistant",
    system_prompt="You are an assistant with access to a knowledge base. Use the retrieval tool to answer questions.",
    llm=get_model("gpt-4"),
    tools=[retrieval_tool]
)

# Run the agent
response = rag_agent.run({"input": "What information do we have about X?"})
print(response["output"])
```

### Multi-Agent Collaboration

```python
from lg_adk import Agent, get_model
from lg_adk.tools.agent_router import AgentRouter, RouterType

# Create specialized agents
researcher = Agent(
    agent_name="researcher",
    system_prompt="You research facts and information thoroughly",
    llm=get_model("gpt-4")
)

writer = Agent(
    agent_name="writer",
    system_prompt="You write clear, engaging content based on research",
    llm=get_model("gpt-4")
)

# Create a sequential router
router = AgentRouter(
    name="research_and_write",
    agents=[researcher, writer],
    router_type=RouterType.SEQUENTIAL
)

# Process a task through both agents sequentially
result = router.run("Explain quantum computing for beginners")
print(result["output"])
```

## Development

```bash
# Clone the repository
git clone https://github.com/yourusername/lg-adk.git
cd lg-adk

# Install dependencies
poetry install

# Run tests
poetry run pytest

# Build documentation
poetry run mkdocs build
```

## Documentation

Comprehensive documentation is available at [https://yourusername.github.io/lg-adk/](https://yourusername.github.io/lg-adk/)

## License

MIT
