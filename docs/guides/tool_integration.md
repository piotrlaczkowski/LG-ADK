# 🛠️ Tool Integration with LG-ADK

---

## 🤔 Why Integrate Tools?

Tools let your agents access external data, APIs, and perform actions beyond LLMs! 🌍

---

## 🧩 Types of Tools

- 🔍 **Retrieval Tools**: Search databases, vector stores, or the web
- 🗃️ **Database Tools**: Read/write to SQL, NoSQL, or custom stores
- 🌐 **API Tools**: Call external APIs (weather, news, etc.)
- 🛠️ **Custom Tools**: Any Python function or class

---

## 🚦 Quick Example

!!! tip "Add a web search tool to your agent"
    ```python
    from lg_adk.tools import WebSearchTool
    agent.add_tool(WebSearchTool())
    ```

---

## 🧠 How Tools Work

- Tools are added to agents via `add_tool()` or at initialization
- Each tool exposes a name, description, and a callable interface
- Agents can decide when to use tools based on prompts or logic

---

## 🛠️ Creating a Custom Tool

!!! example "Minimal custom tool"
    ```python
    from lg_adk.tools import Tool

    class MyTool(Tool):
        name = "my_tool"
        description = "Returns a greeting."

        def run(self, input):
            return f"Hello, {input}!"

    agent.add_tool(MyTool())
    ```

---

## 🔗 Tool Chaining

- You can chain tools together in a graph for complex workflows
- Tools can pass state to each other or to agents

---

## 🚨 Common Pitfalls

!!! warning "Tool name conflicts"
    Make sure each tool has a unique name in your agent or graph.

---

## 🌟 Next Steps

- [Building Graphs](building_graphs.md) 🏗️
- [Memory Management](memory_management.md) 🧠
- [Examples](../examples/) 💡

---

## 🧬 Morphik Integration (Advanced)

LG-ADK supports [Morphik](https://morphik.ai), a platform for advanced document retrieval, knowledge graph creation, and structured context (MCP) for LLMs.

- Use `MorphikRetrievalTool`, `MorphikGraphTool`, `MorphikGraphCreationTool`, and `MorphikMCPTool` for seamless integration.
- Supports semantic search, graph queries, and multi-agent collaboration on Morphik knowledge.

See:
- [Morphik Example](../examples/morphik_example/README.md)
- [Main README Morphik Section](../../README.md#morphik-integration)
- [Morphik Documentation](https://morphik.ai/docs)
