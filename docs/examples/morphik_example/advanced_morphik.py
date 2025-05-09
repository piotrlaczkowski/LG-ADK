"""
Advanced Morphik Integration Example with LG-ADK

This example demonstrates how to use Morphik's advanced features with LG-ADK, including:
1. Creating and utilizing Knowledge Graphs with custom configuration
2. Using Model Context Protocol (MCP) with Morphik
3. Building multi-agent systems with shared Morphik knowledge

Requirements:
- Running Morphik instance
- Morphik Python package installed
- OpenAI API key (for MCP features)

This example shows how to create and manage knowledge graphs in Morphik
and how agents can interact with these graphs to retrieve information about
relationships between entities.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from lg_adk.agents import Agent
from lg_adk.builders import GraphBuilder
from lg_adk.config import get_settings
from lg_adk.database import MorphikDatabaseManager
from lg_adk.models import OpenAIModelProvider
from lg_adk.tools import MorphikGraphCreationTool, MorphikGraphTool, MorphikMCPTool, MorphikRetrievalTool

# Check if Morphik is available
try:
    import morphik
except ImportError:
    print("Morphik package not installed. Please install it with 'pip install morphik'")
    sys.exit(1)


def setup_morphik_example_documents(db_manager: MorphikDatabaseManager) -> List[str]:
    """Create example documents in Morphik for knowledge graph creation.

    Args:
        db_manager: Configured Morphik database manager

    Returns:
        List of created document IDs
    """
    # Create a folder for our example if it doesn't exist
    folder_id = db_manager.create_folder("lg-adk-kg-example")

    # Add some documents about AI technologies
    documents = [
        {
            "title": "LangGraph Overview",
            "content": """
            LangGraph is a library for building stateful, multi-actor applications with LLMs.
            It extends LangChain with a powerful state machine and graph-based orchestration capabilities.
            LangGraph makes it easy to create complex workflows involving multiple agents and tools.
            The library was created by the LangChain team to simplify the creation of multi-agent systems.
            """,
        },
        {
            "title": "LG-ADK Introduction",
            "content": """
            LG-ADK (LangGraph Agent Development Kit) provides high-level abstractions over LangGraph.
            It simplifies agent creation, session management, and tool integration.
            LG-ADK comes with built-in support for various model providers and databases.
            It was designed to make development with LangGraph even easier and more standardized.
            The toolkit includes components for creating agents, building graphs, and managing state.
            """,
        },
        {
            "title": "Morphik Technology",
            "content": """
            Morphik is an advanced document processing system that provides semantic search,
            knowledge graph creation, and structured context retrieval through MCP.
            It allows for natural language rules, folder organization, and user-specific document scoping.
            Morphik integrates well with LLM agent frameworks like LG-ADK.
            The system can extract entities and relationships from documents to build knowledge graphs automatically.
            Morphik's MCP capabilities enable models to receive structured context for better reasoning.
            """,
        },
        {
            "title": "Knowledge Graph Processing",
            "content": """
            Knowledge graphs represent relationships between entities in a structured format.
            They allow for complex queries about relationships and hierarchies.
            Knowledge graphs can be generated automatically from unstructured text using NLP techniques.
            LLMs can use knowledge graphs to reason about entity relationships effectively.
            Entities in a knowledge graph can be people, organizations, concepts, or technologies.
            Relationships define how entities connect, such as "created by", "part of", or "depends on".
            """,
        },
    ]

    # Add the documents to Morphik
    document_ids = []
    for i, doc in enumerate(documents):
        doc_id = db_manager.add_document(
            content=doc["content"],
            metadata={"title": doc["title"], "index": i, "domain": "ai_technology"},
            folder_id=folder_id,
        )
        document_ids.append(doc_id)
        print(f"Added document '{doc['title']}' with ID: {doc_id}")

    return document_ids


def create_kg_manager_agent(db_manager: MorphikDatabaseManager) -> Agent:
    """Create an agent that can manage knowledge graphs in Morphik.

    Args:
        db_manager: Configured Morphik database manager

    Returns:
        An agent with knowledge graph management capabilities
    """
    # Create a tool for knowledge graph management
    kg_creation_tool = MorphikGraphCreationTool(folder_path=db_manager.default_folder, user=db_manager.default_user)

    # Create a basic retrieval tool for document access
    retrieval_tool = MorphikRetrievalTool(folder_path=db_manager.default_folder, user=db_manager.default_user)

    # Create the model provider
    model_provider = OpenAIModelProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4-turbo",
    )

    # Create the agent with both tools
    system_prompt = """
    You are a Knowledge Graph Management Assistant that helps create and manage knowledge graphs in Morphik.

    You can:
    1. Create knowledge graphs from documents
    2. Update existing knowledge graphs with new documents
    3. List available knowledge graphs
    4. Delete knowledge graphs when they are no longer needed

    When creating knowledge graphs, you can use custom configurations for entity extraction and resolution.
    Always provide clear explanations about the operations you perform on knowledge graphs.
    """

    agent = Agent(
        name="KGManagerAgent",
        system_prompt=system_prompt,
        model_provider=model_provider,
        tools=[kg_creation_tool, retrieval_tool],
    )

    return agent


def create_kg_query_agent(db_manager: MorphikDatabaseManager, graph_name: str) -> Agent:
    """Create an agent that can query knowledge graphs in Morphik.

    Args:
        db_manager: Configured Morphik database manager
        graph_name: Name of the knowledge graph to query

    Returns:
        An agent with knowledge graph querying capabilities
    """
    # Create a tool for knowledge graph querying
    kg_tool = MorphikGraphTool(
        folder_path=db_manager.default_folder,
        user=db_manager.default_user,
        graph_name=graph_name,
        hop_depth=2,
        include_paths=True,
    )

    # Create the model provider
    model_provider = OpenAIModelProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4-turbo",
    )

    # Create the agent with the graph tool
    system_prompt = f"""
    You are a Knowledge Graph Expert who can analyze relationships between entities in the '{graph_name}' knowledge graph.

    You can:
    1. Find relationships between specific entities
    2. Explore entity properties and their connections
    3. Trace paths between entities through their relationships

    Always provide rich explanations about the relationships you discover, explaining why they are relevant.
    """

    agent = Agent(name="KGQueryAgent", system_prompt=system_prompt, model_provider=model_provider, tools=[kg_tool])

    return agent


def create_mcp_agent(db_manager: MorphikDatabaseManager) -> Agent:
    """Create an agent that uses Model Context Protocol with Morphik.

    Args:
        db_manager: Configured Morphik database manager

    Returns:
        An agent with MCP capabilities
    """
    # Create an MCP tool for structured context retrieval
    mcp_tool = MorphikMCPTool(folder_path=db_manager.default_folder, user=db_manager.default_user)

    # Create the model provider
    model_provider = OpenAIModelProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4-turbo",  # Ensure this model supports MCP
    )

    # Create the agent with the MCP tool
    system_prompt = """
    You are a Context-Aware Assistant using Model Context Protocol to retrieve structured information.

    You can access structured context about topics in the document repository, which helps you provide
    more accurate and detailed responses based on the available information.

    Always utilize the structured context provided via MCP to enhance your answers.
    """

    agent = Agent(name="MCPAgent", system_prompt=system_prompt, model_provider=model_provider, tools=[mcp_tool])

    return agent


def main():
    """Run the advanced Morphik examples."""
    # Load settings
    settings = get_settings()

    # Check if Morphik is configured
    if not settings.use_morphik_as_default:
        print("Morphik is not set as the default database. Please set USE_MORPHIK_AS_DEFAULT=true")
        return

    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable")
        return

    # Configure Morphik database manager
    db_manager = MorphikDatabaseManager(
        host=settings.morphik_host,
        port=settings.morphik_port,
        api_key=settings.morphik_api_key,
        default_user=settings.morphik_default_user,
        default_folder=settings.morphik_default_folder,
    )

    # Check if Morphik is available
    if not db_manager.is_available():
        print(f"Morphik is not available at {settings.morphik_host}:{settings.morphik_port}")
        return

    # Create example documents for our knowledge graph
    print("\n=== Creating Example Documents ===")
    document_ids = setup_morphik_example_documents(db_manager)
    print(f"Created {len(document_ids)} documents for knowledge graph creation")

    # Create the KG Manager Agent
    print("\n=== Creating Knowledge Graph Manager Agent ===")
    kg_manager = create_kg_manager_agent(db_manager)

    # Build a graph with the KG Manager Agent
    graph_builder = GraphBuilder()
    kg_manager_graph = graph_builder.build(kg_manager)

    # Use the agent to create a knowledge graph with custom entity extraction examples
    print("\n=== Creating Knowledge Graph with Custom Configuration ===")
    create_kg_query = """
    Please create a knowledge graph named 'ai_technologies' using the documents I've provided.
    Use the following custom entity extraction examples:
    - "LangGraph" as type "FRAMEWORK"
    - "LG-ADK" as type "TOOLKIT"
    - "Morphik" as type "DATABASE"

    And the following entity resolution examples:
    - "LangGraph" and "LangGraph library" should be considered the same entity
    - "LG-ADK", "LangGraph Agent Development Kit", and "Agent Development Kit" should be considered the same entity
    - "Morphik", "Morphik system", and "Morphik database" should be considered the same entity

    After creating the graph, list all available knowledge graphs.
    """

    create_kg_response = kg_manager_graph.invoke({"message": create_kg_query})
    print(f"KG Manager Agent Response:\n{create_kg_response['message']}")

    # Allow time for the knowledge graph to be created
    print("Waiting for the knowledge graph to be processed...")
    time.sleep(5)  # Wait a moment for the graph to be processed

    # Create the KG Query Agent
    print("\n=== Creating Knowledge Graph Query Agent ===")
    kg_query_agent = create_kg_query_agent(db_manager, "ai_technologies")

    # Build a graph with the KG Query Agent
    kg_query_graph = graph_builder.build(kg_query_agent)

    # Test some knowledge graph queries
    print("\n=== Querying Knowledge Graph ===")
    relationship_query = "What's the relationship between LangGraph and LG-ADK? Can you also tell me how Morphik relates to these technologies?"
    print(f"Query: {relationship_query}")

    query_response = kg_query_graph.invoke({"message": relationship_query})
    print(f"KG Query Agent Response:\n{query_response['message']}")

    # Test another query exploring entity properties
    print("\n=== Querying Entity Properties ===")
    property_query = "What are the key features or capabilities of Morphik? How does it complement AI frameworks?"
    print(f"Query: {property_query}")

    property_response = kg_query_graph.invoke({"message": property_query})
    print(f"KG Query Agent Response (Properties):\n{property_response['message']}")

    # Create the MCP Agent
    print("\n=== Creating MCP Agent ===")
    mcp_agent = create_mcp_agent(db_manager)

    # Build a graph with the MCP Agent
    mcp_graph = graph_builder.build(mcp_agent)

    # Test an MCP query
    print("\n=== Using MCP for Context Retrieval ===")
    mcp_query = "Explain the key features of LG-ADK and how it relates to both LangGraph and Morphik integration."
    print(f"Query: {mcp_query}")

    mcp_response = mcp_graph.invoke({"message": mcp_query})
    print(f"MCP Agent Response:\n{mcp_response['message']}")

    print("\n=== Example Complete ===")
    print(
        "Note: The example documents and knowledge graph 'ai_technologies' have been created in your Morphik instance"
    )
    print("To clean up, you can delete the 'lg-adk-kg-example' folder in Morphik")


if __name__ == "__main__":
    main()
