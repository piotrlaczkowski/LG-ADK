import unittest
from unittest.mock import MagicMock, patch


class TestSimpleChatExample(unittest.TestCase):
    @patch("lg_adk.config.settings.Settings.from_env")
    @patch("lg_adk.tools.web_search.WebSearchTool")
    @patch("lg_adk.agents.base.Agent.run")
    def test_simple_chat_main(self, mock_run, mock_web_search_tool, mock_settings_from_env):
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.default_llm = "mock-llm"
        mock_settings_from_env.return_value = mock_settings
        # Mock WebSearchTool
        mock_web_search_tool.return_value = MagicMock()
        # Mock Agent.run to return a predictable response
        mock_run.return_value = {"output": "Mocked response", "agent": "assistant"}

        # Import and run the main logic (non-interactive)
        import docs.examples.simple_chat as simple_chat

        # Patch input to simulate a single user message and then exit
        with patch("builtins.input", side_effect=["Hello!", "exit"]), patch("builtins.print") as mock_print:
            simple_chat.main()
            # Check that the assistant responded
            printed = "".join(str(call) for call in mock_print.call_args_list)
            self.assertIn("Mocked response", printed)
