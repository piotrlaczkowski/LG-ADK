import unittest
from unittest.mock import MagicMock, patch


class TestRAGExample(unittest.TestCase):
    @patch("lg_adk.agents.base.Agent.run")
    @patch("langchain_community.vectorstores.Chroma.from_documents")
    @patch("langchain_community.embeddings.HuggingFaceEmbeddings")
    @patch("langchain.text_splitter.RecursiveCharacterTextSplitter.split_documents")
    @patch("langchain_community.document_loaders.TextLoader.load")
    def test_rag_example_main(self, mock_loader, mock_split, mock_embed, mock_chroma, mock_agent_run):
        # Mock all vector store and agent/model calls
        mock_loader.return_value = [MagicMock()]
        mock_split.return_value = [MagicMock()]
        mock_embed.return_value = MagicMock()
        mock_chroma.return_value = MagicMock(similarity_search=lambda q, k: [MagicMock(page_content="context")])
        mock_agent_run.return_value = {"output": "Mocked RAG response"}

        import docs.examples.rag_example as rag_example

        # Run the graph as in the example
        if hasattr(rag_example, "graph"):
            initial_state = {
                "input": "What is AI?",
                "output": "",
                "agent": "query_processor",
                "memory": {},
            }
            result = rag_example.graph.invoke(initial_state)
            self.assertIn("output", result)
            self.assertEqual(result["output"], "Mocked RAG response")
        else:
            self.fail("graph not found in rag_example module")
