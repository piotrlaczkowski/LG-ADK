# LG-ADK - LangGraph Agent Development Kit 🚀

<p align="center">
  <img src="docs/logo.png" width="350"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/lg-adk/"><img src="https://img.shields.io/pypi/v/lg-adk.svg?color=blue" alt="PyPI version"></a>
  <a href="https://github.com/piotrlaczkowski/LG-ADK/actions"><img src="https://github.com/yourusername/LG-ADK/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://piotrlaczkowski.github.io/LG-ADK/"><img src="https://img.shields.io/badge/docs-online-brightgreen" alt="Docs"></a>
</p>

> **LG-ADK** is a Python development kit designed to simplify the creation of LangGraph-based agents, providing an experience similar to Google's Agent Development Kit.

---

## ✨ Why LG-ADK?

- ⚡ **Fast Prototyping**: Build and iterate on agent workflows quickly
- 🧩 **Modular**: Plug-and-play agents, tools, and memory
- 🤝 **Multi-Agent Collaboration**: Orchestrate teams of agents
- 🧠 **Memory & RAG**: Built-in memory and retrieval-augmented generation
- 🛠️ **Tool Integration**: Connect to APIs, databases, and more
- 🖥️ **Local & Cloud Models**: Use Ollama, OpenAI, Gemini, and more
- 🧑‍💻 **Human-in-the-Loop**: Seamlessly add human feedback
- 🖼️ **Visual Debugging**: Inspect and debug with langgraph-cli

---

## 🚀 Features

- 🤖 **Modular Agent Architecture**: Easily define and customize agents
- 🔗 **Flexible Graph Construction**: Build complex agent workflows
- 👥 **Multi-Agent Collaboration**: Use group chats and routers
- 📚 **Retrieval-Augmented Generation**: Simplified RAG interfaces
- 🧠 **Memory Management**: Short-term and long-term memory
- 🗂️ **Session Management**: Maintain context across interactions
- 🧑‍💻 **Human-in-the-Loop**: Integrate human feedback
- 🛠️ **Tool Integration**: Connect to external tools and APIs
- 🖥️ **Local Model Support**: Run with Ollama for privacy/cost
- 🌊 **Streaming Responses**: Real-time streaming
- 🖼️ **Visual Debugging**: langgraph-cli integration
- 🗄️ **Database Flexibility**: Local or PostgreSQL storage
- 🧬 **Vector Store Integration**: Semantic search support

---

## 📦 Installation

```bash
pip install lg-adk
```

Or with Poetry:

```bash
poetry add lg-adk
```

---

## ⚡ Quick Start

### 🤖 Basic Agent

```python
from lg_adk import Agent
from lg_adk.memory import MemoryManager
from lg_adk.sessions import SessionManager
from lg_adk.tools import WebSearchTool

# Create an agent with session and memory management
agent = Agent(
    name="research_assistant",
    llm="gpt-3.5-turbo",  # Or use Ollama: llm="ollama/llama3"
    system_message="You are a research assistant that searches the web and answers questions.",
    session_manager=SessionManager(),
    memory_manager=MemoryManager(),
)

# Add tools to the agent
agent.add_tool(WebSearchTool())

# Run the agent synchronously
result = agent.run(user_input="What are the latest developments in AI?", session_id="user1")
print(result["output"])

# Or run asynchronously
# import asyncio
# result = asyncio.run(agent.arun(user_input="What are the latest developments in AI?", session_id="user1"))
# print(result["output"])
```

### 📚 RAG (Retrieval-Augmented Generation)

```python
from lg_adk import Agent
from lg_adk.memory import MemoryManager
from lg_adk.sessions import SessionManager
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
    name="rag_assistant",
    system_message="You are an assistant with access to a knowledge base. Use the retrieval tool to answer questions.",
    llm="gpt-4",
    tools=[retrieval_tool],
    session_manager=SessionManager(),
    memory_manager=MemoryManager(),
)

# Run the agent
response = rag_agent.run(user_input="What information do we have about X?", session_id="user2")
print(response["output"])
```

### 🤝 Multi-Agent Collaboration

```python
from lg_adk import Agent
from lg_adk.memory import MemoryManager
from lg_adk.sessions import SessionManager
from lg_adk.tools.agent_router import AgentRouter, RouterType

# Create specialized agents
researcher = Agent(
    name="researcher",
    system_message="You research facts and information thoroughly",
    llm="gpt-4",
    session_manager=SessionManager(),
    memory_manager=MemoryManager(),
)

writer = Agent(
    name="writer",
    system_message="You write clear, engaging content based on research",
    llm="gpt-4",
    session_manager=SessionManager(),
    memory_manager=MemoryManager(),
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

---

## 🧑‍💻 Advanced Agent Features

- **Session Management**: Use `session_manager=SessionManager()` for multi-user, persistent context.
- **Memory Management**: Use `memory_manager=MemoryManager()` for conversation history and summarization.
- **Async Support**: Use `await agent.arun(...)` for async models or memory backends.
- **Custom Nodes**: Pass `custom_nodes=[...]` to extend agent workflows (e.g., analytics, human-in-the-loop).
- **Arbitrary Types**: The Agent class supports arbitrary types for managers and tools (no Pydantic schema errors).

---

## 🛠️ Development

```bash
# Clone the repository
git clone https://github.com/piotrlaczkowski/lg-adk.git
cd lg-adk

# Install dependencies
poetry install

# Run tests
poetry run pytest

# Build documentation
poetry run mkdocs build
```

---

## 📖 Documentation

Comprehensive documentation is available at [https://piotrlaczkowski.github.io/lg-adk/](https://piotrlaczkowski.github.io/LG-ADK/1.0.0/)

---

## 💬 Community & Support

- [GitHub Issues](https://github.com/piotrlaczkowski/LG-ADK/issues) — Report bugs or request features
- [Discussions](https://github.com/piotrlaczkowski/LG-ADK/discussions) — Ask questions, share ideas

---

## 📝 License

MIT

---

## 🧬 Morphik Integration

LG-ADK supports [Morphik](https://morphik.ai), a powerful platform for AI applications that provides advanced document processing, knowledge graph capabilities, and structured context integration via Model Context Protocol (MCP).

### Features
- Semantic search and retrieval from Morphik
- Knowledge graph creation and querying
- Structured context via MCP for LLMs
- Multi-agent collaboration on Morphik knowledge

### Prerequisites
- A running Morphik instance ([installation guide](https://morphik.ai/docs))
- The Morphik Python package: `pip install morphik`
- (Optional) OpenAI API key for MCP features

### Configuration
Set the following environment variables to configure Morphik integration:

```bash
export MORPHIK_HOST=localhost
export MORPHIK_PORT=8000
export MORPHIK_API_KEY=your_api_key
export MORPHIK_DEFAULT_USER=default
export MORPHIK_DEFAULT_FOLDER=default
export USE_MORPHIK_AS_DEFAULT=true
```

### Example Usage
See [docs/examples/morphik_example](docs/examples/morphik_example) for full code examples.

```python
from lg_adk.database import MorphikDatabaseManager
from lg_adk.tools import MorphikRetrievalTool, MorphikGraphTool, MorphikGraphCreationTool, MorphikMCPTool

# Initialize Morphik DB manager
settings = ...  # get your Settings instance
morphik_db = MorphikDatabaseManager(
    host=settings.morphik_host,
    port=settings.morphik_port,
    api_key=settings.morphik_api_key,
    default_user=settings.morphik_default_user,
    default_folder=settings.morphik_default_folder
)

# Use Morphik tools
retrieval_tool = MorphikRetrievalTool(morphik_db=morphik_db)
graph_tool = MorphikGraphTool(morphik_db=morphik_db)
graph_creation_tool = MorphikGraphCreationTool(morphik_db=morphik_db)
mcp_tool = MorphikMCPTool(morphik_db=morphik_db)
```

For more, see the [Morphik Example README](docs/examples/morphik_example/README.md) and [Morphik Documentation](https://morphik.ai/docs).
