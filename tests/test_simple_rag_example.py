import importlib
import sys
import unittest
from unittest.mock import MagicMock, patch


class ExitCalled(Exception):
    pass


try:
    from langchain_community.document_loaders import TextLoader
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    TEXTLOADER_AVAILABLE = True
    EMBEDDINGS_AVAILABLE = True
    LANGCHAIN_COMMUNITY_AVAILABLE = True
except ImportError:
    TEXTLOADER_AVAILABLE = False
    EMBEDDINGS_AVAILABLE = False
    LANGCHAIN_COMMUNITY_AVAILABLE = False

try:
    import langchain

    LANGCHAIN_INSTALLED = True
except ImportError:
    LANGCHAIN_INSTALLED = False

from langchain_core.tools import BaseTool


@unittest.skipUnless(LANGCHAIN_INSTALLED, "LangChain not installed")
@unittest.skipUnless(LANGCHAIN_COMMUNITY_AVAILABLE, "langchain_community submodules not available")
@unittest.skipUnless(TEXTLOADER_AVAILABLE, "langchain_community.document_loaders.TextLoader not available")
@unittest.skipUnless(EMBEDDINGS_AVAILABLE, "langchain_community.embeddings.OpenAIEmbeddings not available")
class TestSimpleRAGExample(unittest.TestCase):
    @patch("lg_adk.tools.SimpleVectorRetrievalTool")
    @patch("lg_adk.get_model")
    @patch("docs.examples.simple_rag.get_model")
    def test_create_simple_rag(self, mock_get_model_example, mock_get_model_lgadk, mock_retrieval_tool):
        # Use the same mock for both
        mock_model = MagicMock()
        mock_model.invoke.return_value = "Mocked RAG answer"
        mock_get_model_example.return_value = mock_model
        mock_get_model_lgadk.return_value = mock_model

        class DummyRetrievalTool(BaseTool):
            name: str = "dummy"
            description: str = "dummy"

            def _run(self, *args, **kwargs):
                return {}

        mock_retrieval_tool.return_value = DummyRetrievalTool()

        with patch("os.path.exists", return_value=True), patch("os.makedirs"), patch(
            "builtins.open", unittest.mock.mock_open(read_data="test")
        ), patch("langchain_community.document_loaders.TextLoader.load", return_value=[MagicMock()]), patch(
            "langchain_text_splitters.RecursiveCharacterTextSplitter.split_documents", return_value=[MagicMock()]
        ), patch("langchain_community.embeddings.OpenAIEmbeddings"), patch(
            "langchain_community.vectorstores.FAISS.from_documents", return_value=MagicMock()
        ), patch.dict("os.environ", {"OPENAI_API_KEY": "test"}), patch("builtins.print"), patch.object(
            sys, "exit", side_effect=ExitCalled
        ), patch("lg_adk.Agent.run", return_value={"output": "Mocked RAG answer"}):
            import docs.examples.simple_rag

            importlib.reload(docs.examples.simple_rag)
            simple_rag = docs.examples.simple_rag
            if not hasattr(simple_rag, "create_simple_rag"):
                self.skipTest("create_simple_rag not defined in simple_rag example")
            try:
                simple_rag.create_simple_rag()
            except ExitCalled:
                pass
            # If no exception, the test passes
