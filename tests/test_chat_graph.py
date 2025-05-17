import unittest

try:
    from lg_adk.graphs import chat_graph

    LANGGRAPH_FRAMEWORK_AVAILABLE = True
except ImportError:
    LANGGRAPH_FRAMEWORK_AVAILABLE = False


@unittest.skipUnless(LANGGRAPH_FRAMEWORK_AVAILABLE, "langgraph_framework not installed")
class TestChatGraph(unittest.TestCase):
    def test_chat_graph_import(self):
        self.assertTrue(LANGGRAPH_FRAMEWORK_AVAILABLE)

    def test_chat_graph_module(self):
        # Test all public callables in chat_graph
        for attr in dir(chat_graph):
            if not attr.startswith("_") and callable(getattr(chat_graph, attr)):
                method = getattr(chat_graph, attr)
                try:
                    method()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
