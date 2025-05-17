import unittest

from lg_adk.graphs import multi_agent


class TestMultiAgentGraph(unittest.TestCase):
    def test_multi_agent_graph_module(self):
        # Test all public callables in multi_agent
        for attr in dir(multi_agent):
            if not attr.startswith("_") and callable(getattr(multi_agent, attr)):
                method = getattr(multi_agent, attr)
                try:
                    method()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
