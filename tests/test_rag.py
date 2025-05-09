"""Tests for RAG components in LG-ADK."""

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lg_adk.graphs.rag import build_graph
from lg_adk.memory.memory_manager import MemoryManager


@pytest.fixture
def mock_memory_manager() -> Any:
    """Create a mock memory manager."""
    memory_manager = MagicMock(spec=MemoryManager)
    return memory_manager


@pytest.fixture
def sample_documents() -> Any:
    """Create a temporary directory with sample documents for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a sample document
        doc_path = Path(tmpdir) / "sample.txt"
        with doc_path.open("w") as f:
            f.write(
                """
            This is a sample document for testing RAG functionality.
            It contains information about AI and machine learning.
            """,
            )

        yield tmpdir


class TestRAG(unittest.TestCase):
    """Tests for the RAG graph and its components."""

    @patch("lg_adk.graphs.rag.process_query")
    @patch("lg_adk.graphs.rag.retrieve_context")
    @patch("lg_adk.graphs.rag.generate_response")
    @pytest.mark.skip(reason="Requires working with dynamic state model")
    def test_rag_graph_structure(self, mock_generate, mock_retrieve, mock_process) -> None:
        """Test that the RAG graph has the correct structure."""
        # Setup mocks
        mock_process.return_value = {"query": "processed query"}
        mock_retrieve.return_value = {"context": ["context"]}
        mock_generate.return_value = {"output": "response"}

        # Build the graph
        graph = build_graph()

        # Test with a sample input
        result = graph.invoke({"input": "test query"})

        # Verify that the flow is correct
        mock_process.assert_called_once()
        mock_retrieve.assert_called_once()
        mock_generate.assert_called_once()

        # Check the result
        self.assertEqual(result.get("output"), "response")

    def test_process_query(self) -> None:
        """Test that process_query correctly processes the input query."""
        from lg_adk.graphs.rag import process_query

        # Test with a simple input and query
        state = {}
        query = "What is AI?"
        result = process_query(state, query)

        # Check that the query is set
        self.assertEqual(result["query"], query)
        self.assertIn("messages", result)
        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0].role, "user")
        self.assertEqual(result["messages"][0].content, query)

    def test_retrieve_context(self) -> None:
        """Test that retrieve_context retrieves context for a query."""
        from lg_adk.graphs.rag import retrieve_context

        # Test with a sample query
        state = {"query": "What is AI?"}
        result = retrieve_context(state)

        # Check that documents are present
        self.assertIn("documents", result)
        self.assertTrue(len(result["documents"]) > 0)
        self.assertTrue(hasattr(result["documents"][0], "content"))

    def test_generate_response(self) -> None:
        """Test that generate_response generates a response from context."""
        from lg_adk.graphs.rag import Document, generate_response

        # Test with sample query and documents
        documents = [
            Document(content="AI stands for Artificial Intelligence.", metadata={}),
        ]
        state = {
            "query": "What is AI?",
            "documents": documents,
            "messages": [],
        }
        result = generate_response(state)

        # Check that the messages field is set
        self.assertIn("messages", result)
        self.assertTrue(len(result["messages"]) > 0)
        self.assertEqual(result["messages"][-1].role, "assistant")

    @pytest.mark.skip(reason="Requires working with dynamic state model")
    def test_end_to_end(self) -> None:
        """Test the complete RAG workflow."""
        from lg_adk.graphs.rag import build_graph

        # Build the graph
        graph = build_graph()

        # Test with a simple input
        result = graph.invoke({"input": "What is AI?"})

        # Check that the output is not empty
        self.assertIn("output", result)
        self.assertTrue(result["output"])
        self.assertIn("messages", result)
        self.assertTrue(len(result["messages"]) > 0)


if __name__ == "__main__":
    unittest.main()
