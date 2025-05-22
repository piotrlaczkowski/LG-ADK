"""
Minimal example for AgentRAG usage.
"""
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from lg_adk.agents.agent_rag import AgentRAG, AgentRAGConfig

# Prepare your documents and embeddings
# (Replace with your own documents for real use)
documents = [
    # ... your Document objects ...
]
vectorstore = FAISS.from_documents(documents, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

config = AgentRAGConfig(
    model="ollama/llama3",
    vectorstore=vectorstore,
    enable_human_in_loop=True,
    async_memory=True,
    max_history_tokens=8000,
    debug=True,
)

rag_agent = AgentRAG(config)

result = rag_agent.run("Tell me a joke")
print(result["output"])
print(result["trace"])  # See the full trace for debugging/collab
