import unittest
from unittest.mock import MagicMock, patch

from langchain_core.tools import BaseTool


class TestSimpleRAGExample(unittest.TestCase):
    @patch("lg_adk.tools.retrieval.SimpleVectorRetrievalTool")
    @patch("lg_adk.agents.base.Agent.run")
    @patch("lg_adk.get_model")
    def test_create_simple_rag(self, mock_get_model, mock_agent_run, mock_retrieval_tool):
        # Mock the model and agent run
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        mock_agent_run.return_value = {"output": "Mocked RAG answer"}

        class DummyRetrievalTool(BaseTool):
            name: str = "dummy"
            description: str = "dummy"

            def _run(self, *args, **kwargs):
                return {}

        mock_retrieval_tool.return_value = DummyRetrievalTool()

        # Patch langchain and file IO dependencies, and set OPENAI_API_KEY
        with patch("os.path.exists", return_value=True), patch("os.makedirs"), patch(
            "builtins.open", unittest.mock.mock_open(read_data="test")
        ), patch("langchain_community.document_loaders.TextLoader.load", return_value=[MagicMock()]), patch(
            "langchain.text_splitter.RecursiveCharacterTextSplitter.split_documents", return_value=[MagicMock()]
        ), patch("langchain_community.embeddings.OpenAIEmbeddings"), patch(
            "langchain_community.vectorstores.FAISS.from_documents", return_value=MagicMock()
        ), patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
            import docs.examples.simple_rag as simple_rag

            # Run the function (should print mocked answers)
            simple_rag.create_simple_rag()
            # The Agent.run should have been called for each sample question
            self.assertGreaterEqual(mock_agent_run.call_count, 1)
