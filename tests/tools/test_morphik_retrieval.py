"""Tests for Morphik retrieval tools."""

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from lg_adk.database.morphik_db import GraphEntity, GraphRelationship, MorphikDatabaseManager
from lg_adk.tools.morphik_retrieval import (
    MorphikGraphCreationTool,
    MorphikGraphTool,
    MorphikMCPTool,
    MorphikRetrievalTool,
)


@pytest.fixture
def mock_morphik_db() -> MagicMock:
    """Create a mock MorphikDatabaseManager."""
    mock_db = MagicMock(spec=MorphikDatabaseManager)
    mock_db.is_available.return_value = True

    # Mock query method to return test documents
    mock_db.query.return_value = [
        {
            "content": "Sample document content 1",
            "metadata": {"source": "test-1"},
            "document_id": "doc-1",
            "score": 0.95,
        },
        {
            "content": "Sample document content 2",
            "metadata": {"source": "test-2"},
            "document_id": "doc-2",
            "score": 0.85,
        },
    ]

    # Mock MCP context
    mock_db.get_mcp_context.return_value = json.dumps(
        {
            "query": "test query",
            "sources": [
                {
                    "content": "Sample MCP content",
                    "metadata": {"source": "test"},
                    "document_id": "doc-1",
                    "score": 0.95,
                }
            ],
        }
    )

    # Mock knowledge graph methods
    mock_db.get_knowledge_graphs.return_value = ["test-graph-1", "test-graph-2"]
    mock_db.create_knowledge_graph.return_value = True
    mock_db.update_knowledge_graph.return_value = True
    mock_db.delete_knowledge_graph.return_value = True

    return mock_db


@pytest.fixture
def mock_graph_entities() -> List[Any]:
    """Create mock graph entities and relationships for testing."""
    entities = [
        GraphEntity(
            entity_id="entity-1",
            label="LangGraph",
            entity_type="FRAMEWORK",
            properties={"description": "A framework for LLM applications"},
            document_ids=["doc-1"],
        ),
        GraphEntity(
            entity_id="entity-2",
            label="LG-ADK",
            entity_type="TOOLKIT",
            properties={"description": "LangGraph Agent Development Kit"},
            document_ids=["doc-2"],
        ),
    ]

    relationships = [
        GraphRelationship(
            relationship_id="rel-1",
            source="entity-1",
            target="entity-2",
            relationship_type="EXTENDS",
            document_ids=["doc-1", "doc-2"],
        )
    ]

    paths = [
        {
            "path": [
                {"id": "entity-1", "label": "LangGraph", "type": "FRAMEWORK"},
                {"id": "entity-2", "label": "LG-ADK", "type": "TOOLKIT"},
            ]
        }
    ]

    return entities + relationships + paths


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    with patch("lg_adk.tools.morphik_retrieval.Settings") as MockSettings:
        settings = MagicMock()
        settings.USE_MORPHIK_AS_DEFAULT = False
        settings.MORPHIK_HOST = "localhost"
        settings.MORPHIK_PORT = 8000
        settings.MORPHIK_API_KEY = "test-api-key"
        settings.MORPHIK_DEFAULT_USER = "test-user"
        settings.MORPHIK_DEFAULT_FOLDER = "test-folder"

        MockSettings.return_value = settings
        yield settings


class TestMorphikRetrievalTool:
    """Tests for MorphikRetrievalTool."""

    def test_init(self, mock_morphik_db):
        """Test initialization of the tool."""
        tool = MorphikRetrievalTool(morphik_db=mock_morphik_db)

        assert tool.name == "morphik_retrieval"
        assert "retrieve documents" in tool.description.lower()
        assert tool.morphik_db == mock_morphik_db

    def test_get_morphik_db_from_settings(self, mock_settings):
        """Test getting Morphik DB from settings."""
        with patch("lg_adk.database.managers.get_database_manager") as mock_get_db:
            # Test when USE_MORPHIK_AS_DEFAULT is True
            mock_settings.USE_MORPHIK_AS_DEFAULT = True
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db

            tool = MorphikRetrievalTool()
            db = tool._get_morphik_db()

            assert db == mock_db
            mock_get_db.assert_called_once()

    def test_get_morphik_db_create_new(self, mock_settings):
        """Test creating a new Morphik DB when not default."""
        with patch("lg_adk.tools.morphik_retrieval.MorphikDatabaseManager") as MockMorphikDB:
            # Test when USE_MORPHIK_AS_DEFAULT is False
            mock_settings.USE_MORPHIK_AS_DEFAULT = False
            mock_db = MagicMock()
            MockMorphikDB.return_value = mock_db

            tool = MorphikRetrievalTool()
            db = tool._get_morphik_db()

            assert db == mock_db
            MockMorphikDB.assert_called_once_with(
                host=mock_settings.MORPHIK_HOST,
                port=mock_settings.MORPHIK_PORT,
                api_key=mock_settings.MORPHIK_API_KEY,
                default_user=mock_settings.MORPHIK_DEFAULT_USER,
                default_folder=mock_settings.MORPHIK_DEFAULT_FOLDER,
            )

    def test_run(self, mock_morphik_db):
        """Test running the tool to retrieve documents."""
        tool = MorphikRetrievalTool(morphik_db=mock_morphik_db)

        result = tool._run("test query", k=2)

        assert "Retrieved 2 documents" in result
        assert "Sample document content 1" in result
        assert "Sample document content 2" in result
        assert "(score: 0.9500)" in result  # Formatted score

        mock_morphik_db.query.assert_called_once_with("test query", k=2, filter_metadata=None)

    def test_run_morphik_unavailable(self, mock_morphik_db):
        """Test run when Morphik is not available."""
        mock_morphik_db.is_available.return_value = False
        tool = MorphikRetrievalTool(morphik_db=mock_morphik_db)

        result = tool._run("test query")

        assert "not available" in result.lower()
        assert "check your connection settings" in result.lower()

    def test_arun(self, mock_morphik_db):
        """Test async run calls the sync version."""
        tool = MorphikRetrievalTool(morphik_db=mock_morphik_db)

        with patch.object(tool, "_run") as mock_run:
            mock_run.return_value = "Test result"
            result = tool._arun("test query", k=2)

            mock_run.assert_called_once_with("test query", k=2, filter_metadata=None)


class TestMorphikMCPTool:
    """Tests for MorphikMCPTool."""

    def test_init(self, mock_morphik_db):
        """Test initialization of the tool."""
        tool = MorphikMCPTool(morphik_db=mock_morphik_db, model_provider="test-provider")

        assert tool.name == "morphik_mcp_retrieval"
        assert "mcp" in tool.description.lower()
        assert tool.morphik_db == mock_morphik_db
        assert tool.model_provider == "test-provider"

    def test_run(self, mock_morphik_db):
        """Test running the tool to get MCP context."""
        tool = MorphikMCPTool(morphik_db=mock_morphik_db)

        result = tool._run("test query", k=3)

        # Should return the MCP context directly
        assert isinstance(result, str)
        context = json.loads(result)
        assert "query" in context
        assert "sources" in context

        mock_morphik_db.get_mcp_context.assert_called_once_with("test query", k=3, filter_metadata=None)

    def test_run_morphik_unavailable(self, mock_morphik_db):
        """Test run when Morphik is not available."""
        mock_morphik_db.is_available.return_value = False
        tool = MorphikMCPTool(morphik_db=mock_morphik_db)

        result = tool._run("test query")

        assert "not available" in result.lower()


class TestMorphikGraphTool:
    """Tests for MorphikGraphTool."""

    def test_init(self, mock_morphik_db):
        """Test initialization of the tool."""
        tool = MorphikGraphTool(
            morphik_db=mock_morphik_db,
            graph_name="test-graph",
            hop_depth=2,
            include_paths=True,
        )

        assert tool.name == "morphik_graph"
        assert "query knowledge graphs" in tool.description.lower()
        assert tool.morphik_db == mock_morphik_db
        assert tool.graph_name == "test-graph"
        assert tool.hop_depth == 2
        assert tool.include_paths is True

    def test_run_list_graphs(self, mock_morphik_db):
        """Test running the tool to list available graphs."""
        tool = MorphikGraphTool(morphik_db=mock_morphik_db)

        result = tool._run("list graphs")

        assert "Available knowledge graphs:" in result
        assert "test-graph-1" in result
        assert "test-graph-2" in result

        mock_morphik_db.get_knowledge_graphs.assert_called_once()

    def test_run_query_graph(self, mock_morphik_db, mock_graph_entities):
        """Test running the tool to query a graph."""
        # Setup mock to return graph entities
        mock_morphik_db.query.return_value = mock_graph_entities

        tool = MorphikGraphTool(
            morphik_db=mock_morphik_db,
            graph_name="test-graph",
            hop_depth=2,
            include_paths=True,
        )

        result = tool._run("relationship between LangGraph and LG-ADK")

        # Check sections are present
        assert "## ENTITIES" in result
        assert "## RELATIONSHIPS" in result
        assert "## PATHS" in result

        # Check entity details
        assert "Entity: LangGraph (Type: FRAMEWORK" in result
        assert "Entity: LG-ADK (Type: TOOLKIT" in result

        # Check relationship details
        assert "Relationship: entity-1 -> EXTENDS -> entity-2" in result

        # Check query parameters
        mock_morphik_db.query.assert_called_once_with(
            "relationship between LangGraph and LG-ADK",
            filter_metadata=None,
            graph_name="test-graph",
            hop_depth=2,
            include_paths=True,
        )

    def test_run_morphik_unavailable(self, mock_morphik_db):
        """Test run when Morphik is not available."""
        mock_morphik_db.is_available.return_value = False
        tool = MorphikGraphTool(morphik_db=mock_morphik_db, graph_name="test-graph")

        result = tool._run("test query")

        assert "not available" in result.lower()

    def test_retrieve(self, mock_morphik_db, mock_graph_entities):
        """Test the retrieve method for programmatic access."""
        # Setup mock to return graph entities
        mock_morphik_db.query.return_value = mock_graph_entities

        tool = MorphikGraphTool(
            morphik_db=mock_morphik_db,
            graph_name="test-graph",
        )

        results = tool.retrieve("find entities")

        # Check that results are processed correctly
        assert len(results) == 4  # 2 entities, 1 relationship, 1 path

        # Check entity format
        entities = [r for r in results if r["type"] == "entity"]
        assert len(entities) == 2
        assert entities[0]["label"] == "LangGraph"
        assert entities[0]["entity_type"] == "FRAMEWORK"

        # Check relationship format
        relationships = [r for r in results if r["type"] == "relationship"]
        assert len(relationships) == 1
        assert relationships[0]["relationship_type"] == "EXTENDS"

        # Check path format
        paths = [r for r in results if r["type"] == "path"]
        assert len(paths) == 1
        assert "path" in paths[0]


class TestMorphikGraphCreationTool:
    """Tests for MorphikGraphCreationTool."""

    def test_init(self, mock_morphik_db):
        """Test initialization of the tool."""
        tool = MorphikGraphCreationTool(morphik_db=mock_morphik_db)

        assert tool.name == "morphik_graph_creation"
        assert "create or update knowledge graphs" in tool.description.lower()
        assert tool.morphik_db == mock_morphik_db

    def test_run_create_graph(self, mock_morphik_db):
        """Test running the tool to create a graph."""
        tool = MorphikGraphCreationTool(morphik_db=mock_morphik_db)

        request = json.dumps(
            {
                "action": "create",
                "graph_name": "test-new-graph",
                "document_ids": ["doc-1", "doc-2"],
                "filters": {"category": "test"},
            }
        )

        result = tool._run(request)

        assert "Successfully created knowledge graph" in result
        assert "test-new-graph" in result

        mock_morphik_db.create_knowledge_graph.assert_called_once_with(
            graph_name="test-new-graph",
            document_ids=["doc-1", "doc-2"],
            filters={"category": "test"},
            prompt_overrides=None,
        )

    def test_run_update_graph(self, mock_morphik_db):
        """Test running the tool to update a graph."""
        tool = MorphikGraphCreationTool(morphik_db=mock_morphik_db)

        request = json.dumps(
            {
                "action": "update",
                "graph_name": "test-existing-graph",
                "document_ids": ["doc-3"],
            }
        )

        result = tool._run(request)

        assert "Successfully updated knowledge graph" in result
        assert "test-existing-graph" in result

        mock_morphik_db.update_knowledge_graph.assert_called_once_with(
            graph_name="test-existing-graph",
            document_ids=["doc-3"],
            filters=None,
            prompt_overrides=None,
        )

    def test_run_delete_graph(self, mock_morphik_db):
        """Test running the tool to delete a graph."""
        tool = MorphikGraphCreationTool(morphik_db=mock_morphik_db)

        request = json.dumps(
            {
                "action": "delete",
                "graph_name": "test-delete-graph",
            }
        )

        result = tool._run(request)

        assert "Successfully deleted knowledge graph" in result
        assert "test-delete-graph" in result

        mock_morphik_db.delete_knowledge_graph.assert_called_once_with("test-delete-graph")

    def test_run_list_graphs(self, mock_morphik_db):
        """Test running the tool to list graphs."""
        tool = MorphikGraphCreationTool(morphik_db=mock_morphik_db)

        request = json.dumps(
            {
                "action": "list",
            }
        )

        result = tool._run(request)

        assert "Available knowledge graphs:" in result
        assert "test-graph-1" in result
        assert "test-graph-2" in result

        mock_morphik_db.get_knowledge_graphs.assert_called_once()

    def test_run_invalid_action(self, mock_morphik_db):
        """Test running the tool with an invalid action."""
        tool = MorphikGraphCreationTool(morphik_db=mock_morphik_db)

        request = json.dumps(
            {
                "action": "invalid",
                "graph_name": "test-graph",
            }
        )

        result = tool._run(request)

        assert "Unknown action" in result

    def test_run_missing_graph_name(self, mock_morphik_db):
        """Test running the tool without a graph name."""
        tool = MorphikGraphCreationTool(morphik_db=mock_morphik_db)

        request = json.dumps(
            {
                "action": "create",
            }
        )

        result = tool._run(request)

        assert "Error: graph_name is required" in result

    def test_run_invalid_json(self, mock_morphik_db):
        """Test running the tool with invalid JSON."""
        tool = MorphikGraphCreationTool(morphik_db=mock_morphik_db)

        result = tool._run("not a valid json")

        assert "Error: Invalid JSON input" in result

    def test_run_exception(self, mock_morphik_db):
        """Test running the tool when an exception occurs."""
        tool = MorphikGraphCreationTool(morphik_db=mock_morphik_db)

        # Make create_knowledge_graph raise an exception
        mock_morphik_db.create_knowledge_graph.side_effect = Exception("Test error")

        request = json.dumps(
            {
                "action": "create",
                "graph_name": "test-error-graph",
            }
        )

        result = tool._run(request)

        assert "Error: Test error" in result

    def test_run_morphik_unavailable(self, mock_morphik_db):
        """Test run when Morphik is not available."""
        mock_morphik_db.is_available.return_value = False
        tool = MorphikGraphCreationTool(morphik_db=mock_morphik_db)

        request = json.dumps(
            {
                "action": "create",
                "graph_name": "test-graph",
            }
        )

        result = tool._run(request)

        assert "not available" in result.lower()
