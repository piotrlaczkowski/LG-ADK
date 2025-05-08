# Retrieval-Augmented Generation (RAG)

This guide explains how to use LG-ADK's retrieval tools to build powerful RAG applications that enhance LLM capabilities with external knowledge.

## Overview

LG-ADK provides a set of retrieval tools that make it easy to build RAG applications:

1. **SimpleVectorRetrievalTool**: For retrieving from vector stores (FAISS, Chroma, etc.)
2. **ChromaDBRetrievalTool**: A specialized tool for ChromaDB

These tools can be used with any LangChain-compatible vector store to create agents that can retrieve and reason over external knowledge.

## Basic RAG Setup

Here's a simple example of setting up a RAG application:

```python
import os
from dotenv import load_dotenv
from lg_adk import Agent, get_model
from lg_adk.tools.retrieval import SimpleVectorRetrievalTool

# Load environment variables
load_dotenv()

# Set up vector store (using FAISS as an example)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load and process your documents
loader = TextLoader("path/to/your/document.txt")
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(chunks, embeddings)

# Create a retrieval tool
retrieval_tool = SimpleVectorRetrievalTool(
    name="document_retrieval",
    description="Use this tool to retrieve information from the knowledge base.",
    vector_store=vector_store,
    top_k=3,  # Number of documents to retrieve
    score_threshold=0.7  # Minimum similarity score (0-1)
)

# Create a RAG agent
rag_agent = Agent(
    agent_name="KnowledgeAssistant",
    system_prompt="""
    You are a helpful assistant with access to a knowledge base.
    When answering questions, use the retrieval tool to find relevant information.
    Always cite where your information came from.
    If the information is not available in the knowledge base, acknowledge that.
    """,
    llm=get_model("gpt-4"),
    tools=[retrieval_tool]
)

# Use the agent
response = rag_agent.run({"input": "What information do we have about X?"})
print(response["output"])
```

## Using ChromaDB

ChromaDB is a popular vector database that can be used for RAG applications. LG-ADK provides a specialized tool for ChromaDB:

```python
import chromadb
from lg_adk.tools.retrieval import ChromaDBRetrievalTool

# Set up ChromaDB
chroma_client = chromadb.PersistentClient(path="path/to/chromadb")

# Create embedding function wrapper for ChromaDB
class OpenAIEmbeddingFunction:
    def __call__(self, texts):
        return embeddings.embed_documents(texts)

embedding_function = OpenAIEmbeddingFunction()

# Create a ChromaDB retrieval tool
chromadb_retrieval = ChromaDBRetrievalTool(
    name="chromadb_retrieval",
    description="Use this tool to retrieve information from the ChromaDB knowledge base.",
    collection_name="your_collection",
    chroma_client=chroma_client,
    embedding_function=embedding_function,
    top_k=5,
    score_threshold=0.3
)

# Create a RAG agent with ChromaDB
chromadb_agent = Agent(
    agent_name="ChromaDBAssistant",
    system_prompt="You are an assistant with access to a ChromaDB knowledge base. Use the retrieval tool to find information.",
    llm=get_model("gpt-4"),
    tools=[chromadb_retrieval]
)

# Use the agent
response = chromadb_agent.run({"input": "What information do we have about project X?"})
print(response["output"])
```

## Google ADK-Style Simplified API

LG-ADK allows you to create RAG applications in a style similar to Google's Agent Development Kit:

```python
import os
from dotenv import load_dotenv

# Import LG-ADK components
from lg_adk import Agent
from lg_adk.tools.retrieval import SimpleVectorRetrievalTool

# Load environment variables
load_dotenv()

# Set up vector store (assuming it's already created)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# System instructions
def return_instructions():
    return """
    You are a helpful assistant with access to a knowledge base.
    Use the retrieval tool to find information when answering questions.

    When responding:
    1. First retrieve relevant information using the tool
    2. Then synthesize a clear and helpful response
    3. If the information isn't in the knowledge base, say so

    Always cite where you found your information.
    """

# Create the retrieval tool
knowledge_base_retrieval = SimpleVectorRetrievalTool(
    name='retrieve_documents',
    description='Use this tool to retrieve documents from the knowledge base',
    vector_store=your_vector_store,
    top_k=5,
    score_threshold=0.6,
)

# Create the agent
from lg_adk.models import get_model

rag_agent = Agent(
    agent_name='knowledge_base_agent',
    system_prompt=return_instructions(),
    llm=get_model('gpt-4'),
    tools=[knowledge_base_retrieval]
)

# Use the agent
response = rag_agent.run({"input": "What does the documentation say about X?"})
print(response["output"])
```

## Advanced RAG Patterns

### 1. Query Enhancement

Enhance queries based on conversation context to improve retrieval:

```python
from lg_adk import Agent, GraphBuilder
from lg_adk.tools.retrieval import SimpleVectorRetrievalTool

# Create an agent for query enhancement
query_enhancer = Agent(
    agent_name="QueryEnhancer",
    system_prompt="Your job is to enhance user queries to improve retrieval. Based on the conversation history and current query, create a more detailed query that will help retrieve relevant information.",
    llm=get_model("gpt-4")
)

# Create a retriever agent
retriever_agent = Agent(
    agent_name="Retriever",
    system_prompt="You search for relevant information using the retrieval tool and return it.",
    llm=get_model("gpt-4"),
    tools=[retrieval_tool]
)

# Create a response generator
response_generator = Agent(
    agent_name="ResponseGenerator",
    system_prompt="Based on the retrieved information and the original query, generate a helpful response. Always cite your sources.",
    llm=get_model("gpt-4")
)

# Build the RAG graph
builder = GraphBuilder()
builder.add_agent("query_enhancer", query_enhancer)
builder.add_agent("retriever", retriever_agent)
builder.add_agent("response_generator", response_generator)

# Define the flow
flow = [
    (None, "query_enhancer"),
    ("query_enhancer", "retriever"),
    ("retriever", "response_generator"),
    ("response_generator", None)
]

# Build and use the graph
rag_graph = builder.build(flow=flow)
result = rag_graph.invoke({"input": "Tell me about X", "conversation_history": previous_messages})
print(result["output"])
```

### 2. Multi-Source RAG

Retrieve from multiple knowledge sources and combine the results:

```python
# Create multiple retrieval tools
kb1_retrieval = SimpleVectorRetrievalTool(
    name="kb1_retrieval",
    description="Retrieve from knowledge base 1",
    vector_store=kb1_store
)

kb2_retrieval = SimpleVectorRetrievalTool(
    name="kb2_retrieval",
    description="Retrieve from knowledge base 2",
    vector_store=kb2_store
)

# Create a RAG agent with multiple retrieval tools
multi_source_agent = Agent(
    agent_name="MultiSourceAgent",
    system_prompt="""
    You have access to multiple knowledge bases. Use the appropriate retrieval tool(s)
    based on the question. For general questions, you may need to query multiple sources.
    Always cite which knowledge base provided the information.
    """,
    llm=get_model("gpt-4"),
    tools=[kb1_retrieval, kb2_retrieval]
)
```

### 3. RAG with Memory

Combine RAG with memory management for persistent context:

```python
from lg_adk import Agent, GraphBuilder
from lg_adk.memory import MemoryManager
from lg_adk.sessions import SessionManager
from lg_adk.tools.retrieval import SimpleVectorRetrievalTool

# Set up components
memory_manager = MemoryManager()
session_manager = SessionManager()
retrieval_tool = SimpleVectorRetrievalTool(...)

# Create the RAG agent
rag_agent = Agent(
    agent_name="RagWithMemory",
    system_prompt="You answer questions using the retrieval tool and conversation history.",
    llm=get_model("gpt-4"),
    tools=[retrieval_tool]
)

# Build the graph with memory
builder = GraphBuilder()
builder.add_agent(rag_agent)
builder.add_memory(memory_manager)
builder.enable_session_management(session_manager)

# Build and use the graph
rag_graph = builder.build()

# Example of using the graph with session tracking
session_id = "user123"
response = rag_graph.invoke({
    "input": "What did we discuss about X previously?",
    "session_id": session_id
})
```

## Conclusion

The retrieval tools in LG-ADK make it easy to build powerful RAG applications that can access external knowledge. Whether you're using FAISS, ChromaDB, or another vector store, these tools provide a consistent interface and can be combined with other LG-ADK features like memory management and multi-agent systems to create sophisticated AI applications.
