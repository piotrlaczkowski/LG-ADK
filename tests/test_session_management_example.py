import unittest
from unittest.mock import MagicMock, patch


class TestSessionManagementExample(unittest.TestCase):
    @patch("langchain_openai.ChatOpenAI")
    @patch("langchain_core.prompts.ChatPromptTemplate.from_template")
    def test_session_management_main(self, mock_prompt_template, mock_chat_openai):
        # Mock the LLM and prompt
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Mocked session response")
        mock_chat_openai.return_value = mock_llm
        mock_prompt = MagicMock()
        mock_prompt.__or__ = lambda self, other: MagicMock(invoke=mock_llm.invoke)
        mock_prompt_template.return_value = mock_prompt

        import docs.examples.session_management_example as session_example

        # Patch print to suppress output
        with patch("builtins.print") as mock_print:
            session_example.main()
            # Check that the mocked response was printed
            printed = "".join(str(call) for call in mock_print.call_args_list)
            self.assertIn("Mocked session response", printed)
