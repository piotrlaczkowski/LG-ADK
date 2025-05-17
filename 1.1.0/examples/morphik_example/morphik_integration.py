"""
Example of using LG-ADK with Morphik database.

This example demonstrates how to:
1. Configure Morphik integration
2. Create agents with Morphik retrieval capabilities
3. Use Knowledge Graph capabilities
4. Use Morphik's MCP functionality with LG-ADK agents

Requirements:
- A running Morphik instance (see https://docs.morphik.ai/introduction)
- Morphik Python package installed: pip install morphik
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import lg_adk
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from lg_adk.agents import Agent
from lg_adk.builders import GraphBuilder
from lg_adk.config.settings import Settings
from lg_adk.database import MORPHIK_AVAILABLE, MorphikDatabaseManager
from lg_adk.models import get_model
from lg_adk.tools import MorphikMCPTool, MorphikRetrievalTool
from lg_adk.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Run the Morphik integration example."""
    # Check if Morphik is available
    if not MORPHIK_AVAILABLE:
        logger.error("Morphik package not installed. Please install with: pip install morphik")
        sys.exit(1)

    # Load settings from environment variables
    settings = Settings.from_env()

    # Configure Morphik database
    morphik_db = MorphikDatabaseManager(
        morphik_host=settings.morphik_host,
        morphik_port=settings.morphik_port,
        morphik_api_key=settings.morphik_api_key.get_secret_value() if settings.morphik_api_key else None,
        default_user_id=settings.morphik_default_user,
        default_folder=settings.morphik_default_folder,
    )

    # Check connection to Morphik
    if not morphik_db.is_available():
        logger.error("Could not connect to Morphik. Make sure it's running and check your settings.")
        sys.exit(1)

    logger.info(f"Connected to Morphik at {settings.morphik_host}:{settings.morphik_port}")

    # Create a folder for this example if it doesn't exist
    example_folder = "lg-adk-example"
    morphik_db.create_folder(example_folder)

    # Add a sample document to Morphik
    sample_doc = """
    LangGraph is a Python framework for building stateful, multi-actor applications
    with LLMs. It provides features like memory management, graph-based workflows,
    and tools for agent development.

    LG-ADK (LangGraph Agent Development Kit) is a higher-level abstraction layer over
    LangGraph, providing ready-to-use components for building sophisticated AI agents.
    """

    doc_id = morphik_db.add_document(
        content=sample_doc,
        metadata={"source": "example", "type": "documentation"},
        folder=example_folder,
    )

    logger.info(f"Added sample document with ID: {doc_id}")

    # Create retrieval tool
    morphik_tool = MorphikRetrievalTool(
        name="knowledge_base",
        description="Search for information about LangGraph and LG-ADK",
        morphik_db=morphik_db,
        folder=example_folder,
    )

    # Create a basic agent with the Morphik retrieval tool
    agent = Agent(
        name="morphik_agent",
        model=get_model(settings.default_llm),
        tools=[morphik_tool],
        system_prompt="""
        You are a helpful assistant with access to a knowledge base.
        When you receive a question, use the knowledge_base tool to find relevant information
        before answering.

        Always cite your sources and be transparent about what you know and don't know.
        """,
    )

    # Create graph builder
    builder = GraphBuilder(name="morphik_example")
    builder.add_agent(agent)

    # Build the graph
    graph = builder.build()

    # Test the agent
    query = "What is LangGraph and how does it relate to LG-ADK?"
    logger.info(f"Querying agent: {query}")

    result = graph.invoke({"messages": [{"role": "user", "content": query}]})

    logger.info(f"Agent response: {result['messages'][-1]['content']}")

    # Clean up
    logger.info(f"Example complete. Keeping created documents for reference.")
    logger.info(f"To clean up, manually delete the '{example_folder}' folder in Morphik.")


if __name__ == "__main__":
    main()
