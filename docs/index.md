![LG-ADK Logo](../logo.png)

# LG-ADK: LangGraph Agent Development Kit

LG-ADK is a development kit designed to streamline the creation of LangGraph-based agents. It provides a Python-based framework for building complex agent systems with features similar to Google's Agent Development Kit, but powered by LangGraph.

## Features

- **Modular Agent Architecture**: Easily define and customize agents with different capabilities
- **Flexible Graph Construction**: Build complex agent workflows using LangGraph's powerful graph-based approach
- **Memory Management**: Built-in support for short-term and long-term memory
- **Session Management**: Handle conversations and maintain context across interactions
- **Human-in-the-Loop Capabilities**: Seamlessly integrate human feedback and intervention
- **Tool Integration**: Easily connect agents to external tools and APIs
- **Local Model Support**: Run with Ollama or Gemini for enhanced privacy and reduced costs
- **Streaming Responses**: Real-time streaming of agent responses
- **Visual Debugging**: Inspect and debug agent workflows with langgraph-cli
- **Database Flexibility**: Use various databases (local or PostgreSQL) for storage
- **Vector Store Integration**: Works with different vector stores for semantic search

## Quick Install

```bash
pip install lg-adk
```

## Basic Usage

```python
from lg_adk import Agent, GraphBuilder
from lg_adk.memory import MemoryManager
from lg_adk.tools import WebSearchTool

# Create an agent
agent = Agent(
    name="assistant",
    llm="ollama/llama3",  # Or use "gemini/gemini-pro"
    description="A helpful AI assistant"
)

# Add a tool
agent.add_tool(WebSearchTool())

# Create a graph
builder = GraphBuilder()
builder.add_agent(agent)

# Build and run
graph = builder.build()
response = graph.invoke({"input": "Hello, how can you help me today?"})
print(response["output"])
```

## License

MIT License

## Acknowledgements

This project is inspired by Google's [Agent Development Kit](https://github.com/google/agent-development-kit) and built on top of [LangGraph](https://github.com/langchain-ai/langgraph) by LangChain.
