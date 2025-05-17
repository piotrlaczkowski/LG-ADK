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

### 📚 RAG (Retrieval-Augmented Generation)

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

### 🤝 Multi-Agent Collaboration

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
