"""Tests for Morphik examples."""

import os
from unittest.mock import MagicMock, patch

import pytest

from docs.examples.morphik_example.advanced_morphik import (
    create_kg_manager_agent,
    create_kg_query_agent,
    create_mcp_agent,
    setup_morphik_example_documents,
)


@pytest.fixture
def mock_morphik_db_manager():
    """Create a mock MorphikDatabaseManager."""
    mock_db = MagicMock()
    mock_db.is_available.return_value = True
    mock_db.default_folder = "test-folder"
    mock_db.default_user = "test-user"

    # Setup methods for document creation
    mock_db.create_folder.return_value = "test-folder-id"
    mock_db.add_document.side_effect = lambda content, metadata, folder_id: f"doc-{metadata.get('index', 0)}"

    return mock_db


@pytest.mark.parametrize("os_env", [{"OPENAI_API_KEY": "test-key"}])
@patch("docs.examples.morphik_example.advanced_morphik.OpenAIModelProvider")
def test_create_kg_manager_agent(mock_model_provider, os_env, mock_morphik_db_manager):
    """Test creating a knowledge graph manager agent."""
    # Set up the environment variables
    for key, value in os_env.items():
        os.environ[key] = value

    # Create the agent
    agent = create_kg_manager_agent(mock_morphik_db_manager)

    # Check that the agent has the right tools
    assert agent.name == "KGManagerAgent"
    assert len(agent.tools) == 2
    assert any(tool.name == "morphik_graph_creation" for tool in agent.tools)
    assert any(tool.name == "morphik_retrieval" for tool in agent.tools)

    # Check that the model provider was created with the right parameters
    mock_model_provider.assert_called_once_with(api_key="test-key", model="gpt-4-turbo")


@pytest.mark.parametrize("os_env", [{"OPENAI_API_KEY": "test-key"}])
@patch("docs.examples.morphik_example.advanced_morphik.OpenAIModelProvider")
def test_create_kg_query_agent(mock_model_provider, os_env, mock_morphik_db_manager):
    """Test creating a knowledge graph query agent."""
    # Set up the environment variables
    for key, value in os_env.items():
        os.environ[key] = value

    # Create the agent
    graph_name = "test-kg"
    agent = create_kg_query_agent(mock_morphik_db_manager, graph_name)

    # Check that the agent has the right tools
    assert agent.name == "KGQueryAgent"
    assert len(agent.tools) == 1
    assert agent.tools[0].name == "morphik_graph"
    assert agent.tools[0].graph_name == graph_name
    assert agent.tools[0].hop_depth == 2
    assert agent.tools[0].include_paths is True

    # Check that the model provider was created with the right parameters
    mock_model_provider.assert_called_once_with(api_key="test-key", model="gpt-4-turbo")


@pytest.mark.parametrize("os_env", [{"OPENAI_API_KEY": "test-key"}])
@patch("docs.examples.morphik_example.advanced_morphik.OpenAIModelProvider")
def test_create_mcp_agent(mock_model_provider, os_env, mock_morphik_db_manager):
    """Test creating an MCP agent."""
    # Set up the environment variables
    for key, value in os_env.items():
        os.environ[key] = value

    # Create the agent
    agent = create_mcp_agent(mock_morphik_db_manager)

    # Check that the agent has the right tools
    assert agent.name == "MCPAgent"
    assert len(agent.tools) == 1
    assert agent.tools[0].name == "morphik_mcp_retrieval"

    # Check that the model provider was created with the right parameters
    mock_model_provider.assert_called_once_with(api_key="test-key", model="gpt-4-turbo")


def test_setup_morphik_example_documents(mock_morphik_db_manager):
    """Test setting up example documents for the knowledge graph."""
    document_ids = setup_morphik_example_documents(mock_morphik_db_manager)

    # Check that folder was created
    mock_morphik_db_manager.create_folder.assert_called_once_with("lg-adk-kg-example")

    # Check that documents were added
    assert mock_morphik_db_manager.add_document.call_count == 4
    assert len(document_ids) == 4

    # Check that document IDs were generated
    assert all(doc_id.startswith("doc-") for doc_id in document_ids)
