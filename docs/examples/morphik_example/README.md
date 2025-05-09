# Morphik Integration Examples

This directory contains examples demonstrating how to integrate [Morphik](https://morphik.ai) with LG-ADK. Morphik is a powerful platform for AI applications that provides advanced document processing, knowledge graph capabilities, and structured context integration via Model Context Protocol (MCP).

## Prerequisites

Before running these examples, you'll need:

1. A running Morphik instance - follow Morphik's installation guide
2. The Morphik Python package: `pip install morphik`
3. OpenAI API key (for MCP examples) - set as environment variable `OPENAI_API_KEY`

## Configuration

You can configure Morphik integration using environment variables:

```bash
# Morphik connection settings
export MORPHIK_HOST=localhost  # Default is localhost
export MORPHIK_PORT=8000       # Default is 8000
export MORPHIK_API_KEY=your_api_key  # Optional API key if your instance requires it

# Default user and folder for Morphik operations
export MORPHIK_DEFAULT_USER=default  # Default user for Morphik operations
export MORPHIK_DEFAULT_FOLDER=default  # Default folder for documents

# Set to use Morphik as the default database in LG-ADK
export USE_MORPHIK_AS_DEFAULT=true  # Set to "true" to use Morphik as default
```

## Examples

### Basic Morphik Integration

Run the basic example to see how to connect to Morphik, create a folder, add documents, and perform a simple query:

```bash
python morphik_integration.py
```

This demonstrates:
- Connecting to a Morphik instance
- Creating folders and adding documents
- Using a Morphik retrieval tool with an agent
- Basic query processing

### Advanced Morphik Features

Run the advanced example to see how to use Morphik's knowledge graph capabilities and MCP integration:

```bash
python advanced_morphik.py
```

This example showcases:
- Creating knowledge graphs from documents
- Managing knowledge graphs (creating, updating, listing, deleting)
- Querying entity relationships from knowledge graphs
- Using Model Context Protocol for structured information retrieval
- Multiple agents working with the same Morphik knowledge base

## Using Morphik in Your LG-ADK Applications

### Setting Up the Morphik Database Manager

```python
from lg_adk.database import MorphikDatabaseManager
from lg_adk.config import get_settings

settings = get_settings()
db_manager = MorphikDatabaseManager(
    host=settings.morphik_host,
    port=settings.morphik_port,
    api_key=settings.morphik_api_key,
    default_user=settings.morphik_default_user,
    default_folder=settings.morphik_default_folder
)
```

### Creating Retrieval Tools

```python
from lg_adk.tools import MorphikRetrievalTool

# Basic semantic search tool
retrieval_tool = MorphikRetrievalTool(
    folder_path="my_folder",
    user="my_user"
)
```

### Creating Knowledge Graph Tools

```python
from lg_adk.tools import MorphikGraphTool, MorphikGraphCreationTool

# Tool for querying knowledge graphs
graph_tool = MorphikGraphTool(
    folder_path="my_folder",
    user="my_user",
    graph_name="my_knowledge_graph",
    hop_depth=2,
    include_paths=True
)

# Tool for creating and managing knowledge graphs
graph_creation_tool = MorphikGraphCreationTool(
    folder_path="my_folder",
    user="my_user"
)
```

### Using Morphik with Agents

```python
from lg_adk.agents import Agent
from lg_adk.models import OpenAIModelProvider

agent = Agent(
    name="MorphikAgent",
    model_provider=OpenAIModelProvider(model="gpt-4-turbo"),
    tools=[retrieval_tool, graph_tool],
    system_prompt="You are an assistant with access to a Morphik knowledge base."
)
```

### Using MCP for Advanced Context Integration

```python
from lg_adk.tools import MorphikMCPTool

# Create an MCP tool
mcp_tool = MorphikMCPTool(
    folder_path="my_folder",
    user="my_user"
)

# Use with an MCP-compatible model
mcp_agent = Agent(
    name="MCPAgent",
    model_provider=OpenAIModelProvider(model="gpt-4-turbo"),
    tools=[mcp_tool],
    system_prompt="You are an assistant with access to structured context via MCP."
)
```

### Creating and Managing Knowledge Graphs

```python
# Create a knowledge graph from documents
graph_creation_tool._run(
    action="create",
    graph_name="tech_knowledge",
    document_ids=["doc1", "doc2", "doc3"],
    entity_extraction_prompt="Extract technology entities and their relationships"
)

# List available knowledge graphs
graphs = graph_creation_tool._run(action="list", graph_name="")

# Update a knowledge graph with new documents
graph_creation_tool._run(
    action="update",
    graph_name="tech_knowledge",
    document_ids=["doc4", "doc5"]
)

# Delete a knowledge graph
graph_creation_tool._run(action="delete", graph_name="tech_knowledge")
```

### Using Morphik as the Default Database

To configure LG-ADK to use Morphik as the default database:

1. Set the environment variable: `USE_MORPHIK_AS_DEFAULT=true`
2. In your code, confirm using `get_settings().use_morphik_as_default`

## Additional Resources

- [Morphik Documentation](https://morphik.ai/docs)
- [Model Context Protocol (MCP) Specification](https://github.com/model-context-protocol/model-context-protocol)
