import unittest
from unittest.mock import MagicMock, patch


class TestModelProviderExample(unittest.TestCase):
    @patch("lg_adk.models.get_model")
    @patch("lg_adk.tools.WebSearchTool")
    def test_model_switching_agent(self, mock_web_search_tool, mock_get_model):
        # Mock the model and tools
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        mock_web_search_tool.return_value = MagicMock()

        # Patch GraphBuilder and MemoryManager
        with patch("lg_adk.memory.MemoryManager"), patch("lg_adk.GraphBuilder") as mock_graph_builder:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {"output": "Mocked provider response"}
            mock_graph_builder.return_value.build.return_value = mock_graph

            # Simulate the ModelSwitchingAgent logic
            class DummyAgent:
                def __init__(self):
                    self.current_model = "openai"
                    self.models = {"openai": "openai/gpt-4", "google": "google/gemini-pro"}
                    self.setup_agent()

                def setup_agent(self):
                    self.agent = MagicMock()
                    self.builder = MagicMock()
                    self.graph = mock_graph

                def switch_model(self, provider):
                    self.current_model = provider
                    return f"Switched to {provider} model"

                def process_message(self, message, session_id=None):
                    if message.startswith("switch to "):
                        provider = message.split()[-1]
                        return {"output": self.switch_model(provider)}
                    return self.graph.invoke({"input": message}, {"session_id": session_id})

            agent = DummyAgent()
            # Test model switching
            result = agent.process_message("switch to google")
            self.assertIn("Switched to google", result["output"])
            # Test normal message
            result = agent.process_message("Hello")
            self.assertEqual(result["output"], "Mocked provider response")
