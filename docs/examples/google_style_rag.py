#!/usr/bin/env python
"""
Google-Style RAG Example

This example demonstrates how to create a RAG agent with LG-ADK
in a style very similar to Google's ADK.
"""

import os
from dotenv import load_dotenv

# Import LG-ADK components
from lg_adk import Agent
from lg_adk.tools.retrieval import SimpleVectorRetrievalTool

# Load environment variables
load_dotenv()

# System prompt
def return_instructions_root():
    return """
    You are a helpful assistant with access to a knowledge base about LG-ADK.
    Use the retrieval tool to find information when answering questions.
    
    When responding:
    1. First retrieve relevant information using the tool
    2. Then synthesize a clear and helpful response
    3. If the information isn't in the knowledge base, say so
    
    Always cite where you found your information.
    """

def setup_rag_agent():
    """Set up a RAG agent similar to Google's ADK style."""
    try:
        # Import necessary components
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import OpenAIEmbeddings
        from langchain_community.document_loaders import TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        print("Required packages not found. Install with: pip install langchain langchain-community faiss-cpu")
        return None
    
    print("Setting up RAG agent...")
    
    # Create sample document if needed
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(docs_dir, exist_ok=True)
    
    sample_doc_path = os.path.join(docs_dir, "sample_doc.txt")
    if not os.path.exists(sample_doc_path):
        print("Creating sample document...")
        with open(sample_doc_path, "w") as f:
            f.write("""
# LG-ADK Documentation

LG-ADK (LangGraph Agent Development Kit) is a framework for building AI agents and multi-agent systems.
It provides tools for creating, orchestrating, and deploying AI agents.

## Key Features
- Agent-based architecture
- Multi-agent collaboration
- Memory management
- Tool integration
- Human-in-the-loop capabilities

## Agent Creation
Create an agent by specifying a model and system prompt:

```python
agent = Agent(
    agent_name="MyAgent",
    system_prompt="You are a helpful assistant.",
    llm=get_model("gpt-4")
)
```

## Graph Building
Build a graph to connect multiple agents:

```python
builder = GraphBuilder()
builder.add_agent("assistant", assistant_agent)
builder.add_agent("researcher", research_agent)
builder.enable_human_feedback()
graph = builder.build()
```
            """)
    
    # Process the document
    loader = TextLoader(sample_doc_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Create the retrieval tool (similar to VertexAiRagRetrieval in Google ADK)
    lg_adk_retrieval = SimpleVectorRetrievalTool(
        name='retrieve_documentation',
        description=(
            'Use this tool to retrieve documentation and reference materials about LG-ADK'
        ),
        vector_store=vector_store,
        top_k=5,
        score_threshold=0.6,
    )
    
    # Create the agent (similar to Google ADK style)
    from lg_adk.models import get_model
    
    root_agent = Agent(
        agent_name='lg_adk_rag_agent',
        system_prompt=return_instructions_root(),
        llm=get_model('gpt-4'),
        tools=[
            lg_adk_retrieval,
        ]
    )
    
    return root_agent


def main():
    """Run the example."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable")
        return
    
    # Set up the agent
    agent = setup_rag_agent()
    if not agent:
        return
    
    # Example queries
    queries = [
        "What is LG-ADK?",
        "How do I create an agent?",
        "What's the difference between an agent and a graph?",
        "How can I build a multi-agent system?"
    ]
    
    # Run queries
    for query in queries:
        print(f"\nQuestion: {query}")
        response = agent.run({"input": query})
        print(f"Answer: {response.get('output', 'No response')}")


if __name__ == "__main__":
    main() 