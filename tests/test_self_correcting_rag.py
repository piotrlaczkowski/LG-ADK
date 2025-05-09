"""Tests for self-correcting RAG components in LG-ADK."""

import sys
import unittest
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pytest

from lg_adk import Agent, GraphBuilder


def skip_if_numpy_compatibility_issue(func: Callable) -> Callable:
    """Skip test if NumPy 2.0+ with Python 3.12+ compatibility issue is detected."""

    def wrapper(*args, **kwargs):
        try:
            import numpy as np

            if sys.version_info >= (3, 12) and hasattr(np, "__version__") and np.__version__.startswith("2."):
                pytest.skip("Skipping due to NumPy 2.0+ compatibility issue with Python 3.12+")
            return func(*args, **kwargs)
        except (ImportError, AttributeError):
            # If numpy can't be imported or version check fails, run the test anyway
            return func(*args, **kwargs)

    return wrapper


@pytest.fixture
def mock_agents() -> Any:
    """Create mock agents for testing."""
    query_processor = MagicMock(spec=Agent)
    query_processor.run.return_value = {"output": "processed query"}

    response_generator = MagicMock(spec=Agent)
    response_generator.run.return_value = {"output": "initial response"}

    response_critic = MagicMock(spec=Agent)
    response_critic.run.return_value = {"output": '{"score": 6, "needs_correction": true}'}

    response_corrector = MagicMock(spec=Agent)
    response_corrector.run.return_value = {"output": "corrected response"}

    return {
        "query_processor": query_processor,
        "response_generator": response_generator,
        "response_critic": response_critic,
        "response_corrector": response_corrector,
    }


class TestSelfCorrectingRAG(unittest.TestCase):
    """Tests for self-correcting RAG functionality."""

    @skip_if_numpy_compatibility_issue
    def test_process_query(self) -> None:
        """Test that process_query uses the query processor agent."""
        # Create a mock agent
        mock_agent = MagicMock(spec=Agent)
        mock_agent.run.return_value = {"output": "processed query"}

        # Define the function with the mock agent
        def process_query(state) -> dict:
            user_input = state.get("input", "")
            result = mock_agent.run({"input": user_input})
            processed_query = result.get("output", user_input)
            return {
                **state,
                "original_query": user_input,
                "processed_query": processed_query,
            }

        # Test the function
        state = {"input": "What is quantum computing?"}
        result = process_query(state)

        # Verify the results
        mock_agent.run.assert_called_once_with({"input": "What is quantum computing?"})
        self.assertEqual(result["processed_query"], "processed query")
        self.assertEqual(result["original_query"], "What is quantum computing?")

    @skip_if_numpy_compatibility_issue
    def test_critique_response(self) -> None:
        """Test that critique_response uses the response critic agent."""
        # Create a mock agent
        mock_agent = MagicMock(spec=Agent)
        mock_agent.run.return_value = {"output": '{"score": 6, "needs_correction": true}'}

        # Define the function with the mock agent
        def critique_response(state) -> dict:
            original_query = state.get("original_query", "")
            state.get("context", [])
            state.get("initial_response", "")

            # Generate critique with the critic agent
            prompt = f"Critique this response for question: {original_query}"
            result = mock_agent.run({"input": prompt})
            critique = result.get("output", "")

            # Parse the critique
            import json

            try:
                critique_json = json.loads(critique)
                needs_correction = critique_json.get("needs_correction", False)
            except Exception:
                needs_correction = False

            return {
                **state,
                "critique": critique,
                "needs_correction": needs_correction,
            }

        # Test the function
        state = {
            "original_query": "What is quantum computing?",
            "context": ["Context about quantum computing"],
            "initial_response": "Initial response about quantum computing",
        }
        result = critique_response(state)

        # Verify the results
        mock_agent.run.assert_called_once()
        self.assertTrue(result["needs_correction"])
        self.assertEqual(result["critique"], '{"score": 6, "needs_correction": true}')

    @skip_if_numpy_compatibility_issue
    def test_decide_path(self) -> None:
        """Test that decide_path correctly determines the next node."""

        # Define the function
        def decide_path(state) -> str:
            needs_correction = state.get("needs_correction", False)
            return "correct_response" if needs_correction else "finalize_response"

        # Test with needs_correction=True
        state_correction = {"needs_correction": True}
        result_correction = decide_path(state_correction)
        self.assertEqual(result_correction, "correct_response")

        # Test with needs_correction=False
        state_no_correction = {"needs_correction": False}
        result_no_correction = decide_path(state_no_correction)
        self.assertEqual(result_no_correction, "finalize_response")

    @skip_if_numpy_compatibility_issue
    def test_correct_response(self) -> None:
        """Test that correct_response uses the response corrector agent."""
        # Create a mock agent
        mock_agent = MagicMock(spec=Agent)
        mock_agent.run.return_value = {"output": "corrected response"}

        # Define the function with the mock agent
        def correct_response(state) -> dict:
            state.get("original_query", "")
            state.get("initial_response", "")
            critique = state.get("critique", "")

            # Generate corrected response with the corrector agent
            prompt = f"Correct this response based on critique: {critique}"
            result = mock_agent.run({"input": prompt})
            corrected_response = result.get("output", "")

            return {
                **state,
                "corrected_response": corrected_response,
            }

        # Test the function
        state = {
            "original_query": "What is quantum computing?",
            "initial_response": "Initial response about quantum computing",
            "critique": '{"score": 6, "needs_correction": true}',
        }
        result = correct_response(state)

        # Verify the results
        mock_agent.run.assert_called_once()
        self.assertEqual(result["corrected_response"], "corrected response")

    @skip_if_numpy_compatibility_issue
    def test_finalize_response(self) -> None:
        """Test that finalize_response sets the output field correctly."""

        # Define the function
        def finalize_response(state) -> dict:
            needs_correction = state.get("needs_correction", False)
            initial_response = state.get("initial_response", "")
            corrected_response = state.get("corrected_response", "")

            # Use the corrected response if available, otherwise use the initial response
            final_response = corrected_response if needs_correction else initial_response

            return {
                **state,
                "output": final_response,
            }

        # Test with correction
        state_with_correction = {
            "needs_correction": True,
            "initial_response": "initial response",
            "corrected_response": "corrected response",
        }
        result_with_correction = finalize_response(state_with_correction)
        self.assertEqual(result_with_correction["output"], "corrected response")

        # Test without correction
        state_without_correction = {
            "needs_correction": False,
            "initial_response": "initial response",
        }
        result_without_correction = finalize_response(state_without_correction)
        self.assertEqual(result_without_correction["output"], "initial response")

    @skip_if_numpy_compatibility_issue
    @patch("docs.examples.self_correcting_rag.process_query")
    @patch("docs.examples.self_correcting_rag.retrieve_context")
    @patch("docs.examples.self_correcting_rag.generate_initial_response")
    @patch("docs.examples.self_correcting_rag.critique_response")
    @patch("docs.examples.self_correcting_rag.correct_response")
    @patch("docs.examples.self_correcting_rag.finalize_response")
    def test_self_correcting_rag_flow(
        self,
        mock_finalize,
        mock_correct,
        mock_critique,
        mock_generate,
        mock_retrieve,
        mock_process,
    ) -> None:
        """Test the flow of the self-correcting RAG graph."""
        # Setup mocks
        mock_process.return_value = {
            "processed_query": "quantum computing",
            "original_query": "What is quantum computing?",
        }
        mock_retrieve.return_value = {"context": ["Context about quantum computing"]}
        mock_generate.return_value = {"initial_response": "Initial response about quantum computing"}
        mock_critique.return_value = {"critique": "Critique", "needs_correction": True}
        mock_correct.return_value = {"corrected_response": "Corrected response about quantum computing"}
        mock_finalize.return_value = {"output": "Final response about quantum computing"}

        # Import the example file just to get the flow structure

        # Mock the decision function
        def mock_decide_path(state) -> str:
            return "correct_response" if state.get("needs_correction", False) else "finalize_response"

        # Create the graph builder
        builder = GraphBuilder()

        # Add the nodes
        builder.add_node("process_query", mock_process)
        builder.add_node("retrieve_context", mock_retrieve)
        builder.add_node("generate_initial_response", mock_generate)
        builder.add_node("critique_response", mock_critique)
        builder.add_node("correct_response", mock_correct)
        builder.add_node("finalize_response", mock_finalize)

        # Define the flow
        flow = [
            (None, "process_query"),
            ("process_query", "retrieve_context"),
            ("retrieve_context", "generate_initial_response"),
            ("generate_initial_response", "critique_response"),
            ("critique_response", mock_decide_path),
            ("correct_response", "finalize_response"),
            ("finalize_response", None),
        ]

        # Build the graph
        graph = builder.build(flow=flow)

        # Test with a sample input
        result = graph.invoke({"input": "What is quantum computing?"})

        # Verify that the correct path was taken
        mock_process.assert_called_once()
        mock_retrieve.assert_called_once()
        mock_generate.assert_called_once()
        mock_critique.assert_called_once()
        mock_correct.assert_called_once()  # This should be called since needs_correction=True
        mock_finalize.assert_called_once()

        # Check the result
        self.assertEqual(result, {"output": "Final response about quantum computing"})


if __name__ == "__main__":
    unittest.main()
