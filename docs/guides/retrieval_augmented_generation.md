# 📚 Retrieval-Augmented Generation (RAG) in LG-ADK

---

## 🤔 Why Use RAG?

RAG lets your agents answer with up-to-date, factual, or domain-specific information! 🔍

---

## 🧩 Key RAG Components

- 🧠 **Retriever**: Finds relevant documents or facts
- 📝 **Generator**: LLM that creates answers using retrieved info
- 🗄️ **Vector Store**: Stores and indexes documents for semantic search

---

## 🚦 Quick Example

!!! tip "Add a retrieval tool to your agent"
    ```python
    from lg_adk.tools.retrieval import SimpleVectorRetrievalTool
    agent.add_tool(SimpleVectorRetrievalTool(...))
    ```

---

## 🛠️ How RAG Works

- User query is enhanced with context/history
- Retriever finds relevant docs from a vector store
- Generator (LLM) uses both the query and docs to answer

---

## 🚨 Common Pitfalls

!!! warning "Poor retrieval quality"
    Make sure your vector store is well-populated and embeddings are high quality for best results.

---

## 🌟 Next Steps

- [Tool Integration](tool_integration.md) 🛠️
- [Examples](../examples/) 💡

---

## 🧬 Advanced: Morphik for RAG

LG-ADK supports [Morphik](https://morphik.ai) as a backend for advanced retrieval, semantic search, and knowledge graph RAG workflows.

- Use Morphik tools for large-scale, multi-agent, or knowledge graph-based retrieval.
- See [Morphik Example](../examples/morphik_example/README.md) and [README Morphik Section](../../README.md#morphik-integration) for details.
