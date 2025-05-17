import unittest
from unittest.mock import MagicMock, patch


class TestGraphBuilderExample(unittest.TestCase):
    @patch("lg_adk.agents.base.Agent.run")
    def test_graph_example_main(self, mock_run):
        # Mock Agent.run to return a predictable response
        mock_run.return_value = {"output": "Mocked graph response", "agent": "analyzer"}
        # Import the example
        import docs.examples.graph_example as graph_example

        # Patch input to simulate a single user message and then exit
        with patch("builtins.input", side_effect=["Test input", "exit"]), patch("builtins.print") as mock_print:
            # Simulate the main loop
            if hasattr(graph_example, "graph"):
                # Run the graph as in the example, providing all required fields
                initial_state = {
                    "input": "Test input",
                    "output": "",
                    "agent": "analyzer",
                    "memory": {},
                }
                result = graph_example.graph.invoke(initial_state)
                self.assertIn("output", result)
                self.assertEqual(result["output"], "Mocked graph response")
            else:
                self.fail("graph not found in graph_example module")
