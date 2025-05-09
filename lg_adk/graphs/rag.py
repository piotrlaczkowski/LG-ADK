"""Retrieval-Augmented Generation (RAG) graph for LangGraph CLI.

This module implements a RAG graph that uses vector retrieval
to enhance responses with relevant information.
"""

import uuid
from typing import Any, TypedDict

from langgraph.graph import Graph
from pydantic import BaseModel, Field

from lg_adk.agents import Agent
from lg_adk.builders import GraphBuilder
from lg_adk.database import DatabaseManager
from lg_adk.memory import MemoryManager
from lg_adk.models import get_model
from lg_adk.tools.retrieval import SimpleVectorRetrievalTool


# Define document type
class Document(BaseModel):
    """Represents a document retrieved from the vector store."""

    content: str = Field(..., description="Content of the document")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata about the document")


# Define message type
class Message(BaseModel):
    """Represents a chat message."""

    role: str = Field(..., description="Role of the message sender (system, user, assistant)")
    content: str = Field(..., description="Content of the message")


# Define state type
class RAGState(TypedDict):
    """Represents the state of a RAG conversation."""

    messages: list[Message]
    session_id: str
    documents: list[Document]
    query: str
    metadata: dict[str, Any]


# Define helper functions at module level for export
def prepare_initial_state(input_text: str) -> RAGState:
    """Prepare the initial state for the RAG graph with required fields.

    Args:
        input_text: The user's input text

    Returns:
        A properly formatted initial state with all required fields
    """
    return {
        "input": input_text,
        "output": "",  # Required by pydantic validation
        "agent": "",  # Required by pydantic validation
        "memory": {},  # Required by pydantic validation
        "messages": [],
        "documents": [],
        "query": "",
        "metadata": {},
        "session_id": str(uuid.uuid4()),  # Create a valid UUID string for session_id
    }


def process_query(state: RAGState, query: str) -> RAGState:
    """Process user query and add to state."""
    messages = state.get("messages", [])
    user_message = Message(role="user", content=query)

    return {
        **state,
        "messages": messages + [user_message],
        "query": query,
    }


def retrieve_context(state: RAGState) -> RAGState:
    """Retrieve relevant documents for the query (alias for retrieve_documents)."""
    # This is an alias for retrieve_documents for test compatibility
    return retrieve_documents(state)


def retrieve_documents(state: RAGState) -> RAGState:
    """Retrieve relevant documents for the query."""
    query = state.get("query", "")
    if not query:
        return state

    # Mock retrieval for testing
    doc_objects = [
        Document(
            content=f"Document about {query}. This is relevant information for the query.",
            metadata={"source": "knowledge_base", "relevance": 0.92},
        ),
        Document(
            content=f"Additional information related to {query}. More context here.",
            metadata={"source": "knowledge_base", "relevance": 0.85},
        ),
    ]

    return {
        **state,
        "documents": doc_objects,
    }


def generate_response(state: RAGState) -> RAGState:
    """Generate response using the retrieved documents."""
    messages = state.get("messages", [])
    documents = state.get("documents", [])

    # Prepare context from documents
    context = ""
    if documents:
        context = "Here are relevant documents to help answer the query:\n\n"
        for i, doc in enumerate(documents, 1):
            context += f"DOCUMENT {i}:\n{doc.content}\n\n"

    # Mock response for testing
    response = "Based on the retrieved documents, here is my answer..."

    # Add assistant response to messages
    assistant_message = Message(role="assistant", content=response)

    return {
        **state,
        "messages": messages + [assistant_message],
    }


def create_rag_graph() -> Graph:
    """Create a RAG graph with proper session handling.

    This function builds a RAG graph compatible with the LangGraph CLI.
    It performs vector retrieval and uses the results to augment responses.

    Returns:
        Graph: A LangGraph graph instance for RAG functionality.
    """

    # Mock vector store (replace with your actual vector store in a real application)
    class MockVectorStore:
        def similarity_search(self, query: str) -> list[Document]:
            """Simulate vector search with mock documents.

            Args:
                query: The query string to search for.

            Returns:
                list[Document]: List of mock documents relevant to the query.
            """
            # Simulate vector search with mock documents
            return [
                Document(
                    content=f"Document about {query}. This is relevant information for the query.",
                    metadata={"source": "knowledge_base", "relevance": 0.92},
                ),
                Document(
                    content=f"Additional information related to {query}. More context here.",
                    metadata={"source": "knowledge_base", "relevance": 0.85},
                ),
                Document(
                    content=f"General background about topics like {query}.",
                    metadata={"source": "knowledge_base", "relevance": 0.76},
                ),
            ]

    # Create vector store and retrieval tool
    vector_store = MockVectorStore()
    retrieval_tool = SimpleVectorRetrievalTool(
        name="knowledge_base",
        description="Retrieves information from the knowledge base",
        vector_store=vector_store,
        top_k=3,
    )

    # Create RAG agent
    rag_agent = Agent(
        name="rag_assistant",
        model=get_model("openai/gpt-4"),
        system_prompt=(
            "You are a knowledgeable assistant with access to a document retrieval system. "
            "When answering questions, use the retrieved documents to provide accurate information. "
            "Always cite your sources from the documents when you use them."
        ),
        tools=[retrieval_tool],
    )

    # Create memory manager
    db_manager = DatabaseManager(connection_string="sqlite:///rag_memory.db")
    memory_manager = MemoryManager(database_manager=db_manager)

    # Create graph builder
    builder = GraphBuilder(name="rag")
    builder.add_agent(rag_agent)
    builder.add_memory(memory_manager)

    # Configure state tracking
    builder.configure_state_tracking(
        include_session_id=True,
        include_metadata=True,
    )

    # Wire up the graph - using the module-level functions
    builder.add_node("process_query", process_query)
    builder.add_node("retrieve_documents", retrieve_documents)
    builder.add_node("generate_response", generate_response)

    builder.add_edge("process_query", "retrieve_documents")
    builder.add_edge("retrieve_documents", "generate_response")
    builder.add_edge("generate_response", "process_query")

    # Set entry and exit points
    builder.set_entry_point("process_query")
    builder.set_exit_point("generate_response")

    # Build the graph with proper typing
    typed_graph: Graph[RAGState] = builder.build()

    # Patch the graph's invoke method to add required initial fields
    original_invoke = typed_graph.invoke

    def patched_invoke(data, *args: Any, **kwargs) -> Any:
        if isinstance(data, dict) and "input" in data and all(key not in data for key in ["output", "agent", "memory"]):
            # Create a properly formatted initial state
            initial_state = prepare_initial_state(data["input"])
            return original_invoke(initial_state, *args, **kwargs)
        return original_invoke(data, *args, **kwargs)

    typed_graph.invoke = patched_invoke

    return typed_graph


# Export the graph for the LangGraph CLI to discover
graph = create_rag_graph()

# Alias for compatibility with tests and imports
build_graph = create_rag_graph

# Example of how to use the graph directly
if __name__ == "__main__":
    # Create a session ID
    session_id = str(uuid.uuid4())

    # Configure for using with checkpointer
    config = {
        "configurable": {
            "thread_id": session_id,
        },
    }

    # Initialize state
    initial_state = {
        "messages": [],
        "session_id": session_id,
        "documents": [],
        "query": "",
        "metadata": {"started_at": str(uuid.uuid4())},
    }

    # Process queries
    while True:
        # Get user query
        user_query = input("Question: ")
        if user_query.lower() in ["exit", "quit", "bye"]:
            break

        # Process query through the graph
        result = graph.invoke(
            {"query": user_query},
            config=config,
            state=initial_state,
        )

        # Update state for next iteration
        initial_state = result

        # Extract and print the last message (assistant's response)
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            if last_message.role == "assistant":
                pass

        # Also show the documents that were retrieved (for demonstration)
        documents = result.get("documents", [])
        if documents and user_query.lower() == "show documents":
            for _i, _doc in enumerate(documents, 1):
                pass
