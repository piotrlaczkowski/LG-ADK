#!/usr/bin/env python
"""
Simple RAG Example

This example demonstrates how to create a RAG (Retrieval-Augmented Generation) agent
with LG-ADK that is similar to Google's ADK.
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv

# Import LG-ADK components
from lg_adk import Agent, get_model
from lg_adk.tools.retrieval import SimpleVectorRetrievalTool, ChromaDBRetrievalTool

# Load environment variables
load_dotenv()


def create_simple_rag() -> None:
    """Create a simple RAG agent using a vector store."""
    try:
        # Import LangChain components
        from langchain_community.vectorstores import FAISS, Chroma
        from langchain_community.embeddings import OpenAIEmbeddings
        from langchain_community.document_loaders import TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        print("LangChain packages are required for this example. Install with:")
        print("pip install langchain langchain-community faiss-cpu")
        return

    print("\n=== Creating a Simple RAG Agent ===\n")
    
    # Load documents (replace with your own documents)
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(docs_dir, exist_ok=True)
    
    # Create a sample document if it doesn't exist
    sample_doc_path = os.path.join(docs_dir, "sample_doc.txt")
    if not os.path.exists(sample_doc_path):
        with open(sample_doc_path, "w") as f:
            f.write("""
# LG-ADK Documentation

## Overview
LG-ADK (LangGraph Agent Development Kit) is a framework for building AI agents and multi-agent systems.
It provides tools for creating, orchestrating, and deploying AI agents.

## Features
- Agent-based architecture
- Multi-agent collaboration
- Memory management
- Tool integration
- Human-in-the-loop capabilities

## Getting Started
To get started with LG-ADK, install the package and import the necessary components:

```python
pip install lg-adk

from lg_adk import Agent, GraphBuilder
```

## Agent Creation
Create an agent by specifying a model and system prompt:

```python
agent = Agent(
    agent_name="MyAgent",
    system_prompt="You are a helpful assistant.",
    llm=get_model("gpt-4")
)
```

## Building Graphs
Create a graph by connecting multiple agents:

```python
builder = GraphBuilder()
builder.add_agent("assistant", assistant_agent)
builder.add_agent("researcher", research_agent)
builder.enable_human_feedback()
graph = builder.build()
```
            """)
    
    # Load and process documents
    loader = TextLoader(sample_doc_path)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings using OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY environment variable is required.")
        print("Please set it before running this example.")
        return
    
    embeddings = OpenAIEmbeddings()
    
    # Create a vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Create a retrieval tool
    retrieval_tool = SimpleVectorRetrievalTool(
        name="retrieve_documentation",
        description="Use this tool to retrieve documentation and reference materials from the knowledge base.",
        vector_store=vector_store,
        top_k=3,
        score_threshold=0.7
    )
    
    # System prompt for the RAG agent
    system_prompt = """
    You are a helpful assistant with access to documentation about LG-ADK.
    When asked questions, use the retrieval tool to find relevant information.
    
    Follow these steps when responding:
    1. Analyze the question to understand what information is needed
    2. Use the retrieval tool to find relevant documentation
    3. Synthesize a helpful response based on the retrieved information
    4. If the information is not available, acknowledge that and provide general help
    
    Always reference where your information comes from in the retrieved documents.
    """
    
    # Create the RAG agent
    model = get_model("gpt-4")
    rag_agent = Agent(
        agent_name="DocumentationAssistant",
        system_prompt=system_prompt,
        llm=model,
        tools=[retrieval_tool]
    )
    
    # Test the agent with a sample question
    sample_questions = [
        "How do I create an agent with LG-ADK?",
        "What are the main features of LG-ADK?",
        "Can you explain how to build a graph with multiple agents?",
        "What is the purpose of LG-ADK?"
    ]
    
    # Run the agent on each sample question
    for question in sample_questions:
        print(f"\nQuestion: {question}")
        response = rag_agent.run({"input": question})
        print(f"Answer: {response.get('output', 'No response')}")


def create_chromadb_rag() -> None:
    """Create a RAG agent using ChromaDB."""
    try:
        import chromadb
        from langchain_community.embeddings import OpenAIEmbeddings
        from langchain_community.document_loaders import TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        print("ChromaDB and LangChain packages are required for this example. Install with:")
        print("pip install chromadb langchain langchain-community")
        return
    
    print("\n=== Creating a ChromaDB RAG Agent ===\n")
    
    # Sample document path
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    sample_doc_path = os.path.join(docs_dir, "sample_doc.txt")
    
    if not os.path.exists(sample_doc_path):
        print(f"Sample document not found: {sample_doc_path}")
        print("Please run the create_simple_rag() function first.")
        return
    
    # Load and process documents
    loader = TextLoader(sample_doc_path)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings using OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY environment variable is required.")
        return
    
    embeddings = OpenAIEmbeddings()
    
    # Create a ChromaDB client
    chroma_dir = os.path.join(docs_dir, "chroma_db")
    os.makedirs(chroma_dir, exist_ok=True)
    
    client = chromadb.PersistentClient(path=chroma_dir)
    
    # Create embedding function wrapper for ChromaDB
    class OpenAIEmbeddingFunction:
        def __call__(self, texts):
            return embeddings.embed_documents(texts)
    
    embedding_function = OpenAIEmbeddingFunction()
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name="lg_adk_docs",
        embedding_function=embedding_function
    )
    
    # Add documents if collection is empty
    if collection.count() == 0:
        collection.add(
            documents=[chunk.page_content for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
            ids=[f"doc_{i}" for i in range(len(chunks))]
        )
    
    # Create a ChromaDB retrieval tool
    retrieval_tool = ChromaDBRetrievalTool(
        name="chromadb_retrieval",
        description="Use this tool to retrieve documentation about LG-ADK from the ChromaDB knowledge base.",
        collection_name="lg_adk_docs",
        chroma_client=client,
        embedding_function=embedding_function,
        top_k=3,
        score_threshold=0.3
    )
    
    # System prompt for the RAG agent
    system_prompt = """
    You are a helpful assistant with access to documentation about LG-ADK.
    When asked questions, use the ChromaDB retrieval tool to find relevant information.
    
    Follow these steps when responding:
    1. Analyze the question to understand what information is needed
    2. Use the retrieval tool to find relevant documentation
    3. Synthesize a helpful response based on the retrieved information
    4. If the information is not available, acknowledge that and provide general help
    
    Always cite where your information comes from in the retrieved documents.
    """
    
    # Create the RAG agent
    model = get_model("gpt-4")
    chroma_rag_agent = Agent(
        agent_name="ChromaDocumentationAssistant",
        system_prompt=system_prompt,
        llm=model,
        tools=[retrieval_tool]
    )
    
    # Test the agent with a sample question
    question = "How do I create a multi-agent system with LG-ADK?"
    print(f"\nQuestion: {question}")
    response = chroma_rag_agent.run({"input": question})
    print(f"Answer: {response.get('output', 'No response')}")


def main() -> None:
    """Run the examples."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable")
        return
    
    # Example with simple vector store
    create_simple_rag()
    
    # Example with ChromaDB
    create_chromadb_rag()


if __name__ == "__main__":
    main() 