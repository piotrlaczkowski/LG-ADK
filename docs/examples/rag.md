# RAG Examples

This section provides examples of building Retrieval-Augmented Generation (RAG) applications using LG-ADK.

## Simple RAG Example

The following example demonstrates how to create a simple RAG application using FAISS as the vector store:

```python
import os
from typing import Dict, Any, List
from dotenv import load_dotenv

# Import LG-ADK components
from lg_adk import Agent, get_model
from lg_adk.tools.retrieval import SimpleVectorRetrievalTool

# Load environment variables
load_dotenv()

# Set up vector store
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load and process documents
loader = TextLoader("path/to/document.txt")
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
    name="retrieve_documentation",
    description="Use this tool to retrieve documentation and reference materials from the knowledge base.",
    vector_store=vector_store,
    top_k=3,
    score_threshold=0.7
)

# Create the RAG agent
rag_agent = Agent(
    agent_name="DocumentationAssistant",
    system_prompt="""
    You are a helpful assistant with access to documentation.
    When asked questions, use the retrieval tool to find relevant information.
    Always reference where your information came from.
    """,
    llm=get_model("gpt-4"),
    tools=[retrieval_tool]
)

# Use the agent
response = rag_agent.run({"input": "What information do we have about X?"})
print(response["output"])
```

## ChromaDB Example

This example shows how to use ChromaDB as the vector store:

```python
import chromadb
from lg_adk import Agent, get_model
from lg_adk.tools.retrieval import ChromaDBRetrievalTool
from langchain_community.embeddings import OpenAIEmbeddings

# Set up ChromaDB
chroma_client = chromadb.PersistentClient(path="path/to/chromadb")

# Create embedding function wrapper for ChromaDB
class OpenAIEmbeddingFunction:
    def __call__(self, texts):
        return embeddings.embed_documents(texts)

embeddings = OpenAIEmbeddings()
embedding_function = OpenAIEmbeddingFunction()

# Get or create collection
collection = chroma_client.get_or_create_collection(
    name="your_collection",
    embedding_function=embedding_function
)

# Add documents if needed
# collection.add(
#     documents=["doc1", "doc2", "doc3"],
#     metadatas=[{"source": "source1"}, {"source": "source2"}, {"source": "source3"}],
#     ids=["id1", "id2", "id3"]
# )

# Create a ChromaDB retrieval tool
retrieval_tool = ChromaDBRetrievalTool(
    name="chromadb_retrieval",
    description="Use this tool to retrieve information from the ChromaDB knowledge base.",
    collection_name="your_collection",
    chroma_client=chroma_client,
    embedding_function=embedding_function,
    top_k=3,
    score_threshold=0.3
)

# Create a RAG agent
chromadb_agent = Agent(
    agent_name="ChromaDBAssistant",
    system_prompt="You are an assistant with access to a ChromaDB knowledge base. Use the retrieval tool to find information.",
    llm=get_model("gpt-4"),
    tools=[retrieval_tool]
)

# Use the agent
response = chromadb_agent.run({"input": "What information do we have about X?"})
print(response["output"])
```

## Google ADK-Style RAG

This example shows how to create a RAG application in a style similar to Google's Agent Development Kit:

```python
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
    You are a helpful assistant with access to a knowledge base.
    Use the retrieval tool to find information when answering questions.

    When responding:
    1. First retrieve relevant information using the tool
    2. Then synthesize a clear and helpful response
    3. If the information isn't in the knowledge base, say so

    Always cite where you found your information.
    """

def setup_rag_agent():
    """Set up a RAG agent similar to Google's ADK style."""
    # Set up vector store (code omitted for brevity)
    from lg_adk.models import get_model

    # Create the retrieval tool
    lg_adk_retrieval = SimpleVectorRetrievalTool(
        name='retrieve_documentation',
        description=(
            'Use this tool to retrieve documentation and reference materials'
        ),
        vector_store=your_vector_store,
        top_k=5,
        score_threshold=0.6,
    )

    # Create the agent
    rag_agent = Agent(
        agent_name='documentation_agent',
        system_prompt=return_instructions_root(),
        llm=get_model('gpt-4'),
        tools=[
            lg_adk_retrieval,
        ]
    )

    return rag_agent

# Create and use the agent
agent = setup_rag_agent()
response = agent.run({"input": "What does the documentation say about X?"})
print(response["output"])
```

## Advanced RAG with Memory

This example shows how to create a RAG application with memory to maintain context across interactions:

```python
from lg_adk import Agent, GraphBuilder
from lg_adk.memory import MemoryManager
from lg_adk.sessions import SessionManager
from lg_adk.tools.retrieval import SimpleVectorRetrievalTool

# Set up components
memory_manager = MemoryManager()
session_manager = SessionManager()
retrieval_tool = SimpleVectorRetrievalTool(...)

# Define node functions
def get_or_create_session(state):
    """Get or create a session."""
    import uuid

    session_id = state.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session_manager.create_session(session_id)

    session_data = session_manager.get_session(session_id)
    return {"session_id": session_id, "session_data": session_data}

def retrieve_history(state):
    """Retrieve conversation history."""
    session_id = state["session_id"]
    conversation_history = memory_manager.get_conversation_history(session_id)
    return {"conversation_history": conversation_history}

def retrieve_context(state):
    """Retrieve relevant documents."""
    query = state["input"]
    context = retrieval_tool.run(query)
    return {"context": context}

def generate_response(state):
    """Generate a response based on context and history."""
    rag_agent = Agent(
        agent_name="RAGWithMemory",
        system_prompt="Answer based on the context and conversation history.",
        llm=get_model("gpt-4")
    )

    result = rag_agent.run({
        "input": state["input"],
        "context": state["context"],
        "conversation_history": state["conversation_history"]
    })

    return {"output": result["output"]}

def update_memory(state):
    """Add the interaction to memory."""
    session_id = state["session_id"]
    memory_manager.add_message(session_id, {"role": "user", "content": state["input"]})
    memory_manager.add_message(session_id, {"role": "assistant", "content": state["output"]})
    return state

# Build the graph
builder = GraphBuilder()
builder.add_node("get_or_create_session", get_or_create_session)
builder.add_node("retrieve_history", retrieve_history)
builder.add_node("retrieve_context", retrieve_context)
builder.add_node("generate_response", generate_response)
builder.add_node("update_memory", update_memory)

# Define the flow
flow = [
    (None, "get_or_create_session"),
    ("get_or_create_session", "retrieve_history"),
    ("retrieve_history", "retrieve_context"),
    ("retrieve_context", "generate_response"),
    ("generate_response", "update_memory"),
    ("update_memory", None)
]

# Build and use the graph
rag_graph = builder.build(flow=flow)
result = rag_graph.invoke({"input": "Tell me about X"})
print(result["output"])

# Continue the conversation
result = rag_graph.invoke({"input": "Tell me more about it", "session_id": result["session_id"]})
print(result["output"])
```

## Full Examples

For more detailed examples, see the full code in the `docs/examples` directory:

- [simple_rag.py](https://github.com/yourusername/lg-adk/blob/main/docs/examples/simple_rag.py): A complete example of creating RAG agents with FAISS and ChromaDB
- [google_style_rag.py](https://github.com/yourusername/lg-adk/blob/main/docs/examples/google_style_rag.py): An example showing the Google ADK-style approach
- [rag_with_memory.py](https://github.com/yourusername/lg-adk/blob/main/docs/examples/rag_with_memory.py): An example demonstrating RAG with conversation memory
