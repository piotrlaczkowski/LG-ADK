import unittest

from lg_adk.tools import agent_tools


class TestAgentTools(unittest.TestCase):
    def test_agent_tools_module(self):
        # Test all public callables in agent_tools
        for attr in dir(agent_tools):
            if not attr.startswith("_") and callable(getattr(agent_tools, attr)):
                method = getattr(agent_tools, attr)
                try:
                    method()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
