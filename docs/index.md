# 👋 Welcome to LG-ADK: LangGraph Agent Development Kit 🚀

<p align="center">
  <img src="logo.png" width="350"/>
</p>

**Build next-generation AI agents and workflows with LangGraph, made easy!**

---

> **LG-ADK** is a Python framework for rapidly building, composing, and deploying powerful agent systems using LangGraph. Inspired by Google's ADK, but supercharged for the open-source ecosystem.

---

## ✨ Why LG-ADK?

- 🤖 **Modular Agent Architecture**: Easily define and customize agents with different capabilities
- 🔗 **Flexible Graph Construction**: Build complex agent workflows using LangGraph's powerful graph-based approach
- 🧠 **Memory Management**: Built-in support for short-term and long-term memory
- 🗂️ **Session Management**: Handle conversations and maintain context across interactions
- 🧑‍💻 **Human-in-the-Loop**: Seamlessly integrate human feedback and intervention
- 🛠️ **Tool Integration**: Easily connect agents to external tools and APIs
- 🖥️ **Local Model Support**: Run with Ollama or Gemini for privacy and cost savings
- 🌊 **Streaming Responses**: Real-time streaming of agent responses
- 🖼️ **Visual Debugging**: Inspect and debug agent workflows with langgraph-cli
- 🗄️ **Database Flexibility**: Use various databases (local or PostgreSQL) for storage
- 🧬 **Vector Store Integration**: Works with different vector stores for semantic search

---

## 📦 Quick Install

!!! tip "Install with pip or Poetry"
    ```bash
    pip install lg-adk
    # or
    poetry add lg-adk
    ```

---

## ⚡ Basic Usage

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

---

## 🚦 Quick Links

- [Getting Started](getting_started/quick_start.md) 🚦
- [Guides](guides/building_graphs.md) 📚
- [Examples](examples/index.md) 💡
- [API Reference](reference/) 🛠️

---

## 📝 License

MIT License

---

## 🙏 Acknowledgements

This project is inspired by Google's [Agent Development Kit](https://github.com/google/agent-development-kit) and built on top of [LangGraph](https://github.com/langchain-ai/langgraph) by LangChain.

---

<footer align="center">
  <b>Made with ❤️ by the LG-ADK community.</b><br>
  <a href="https://github.com/piotrlaczkowski/lg-adk">GitHub</a> · <a href="https://pypi.org/project/lg-adk/">PyPI</a>
</footer>
