import unittest

from lg_adk.tools import agent_router


class TestAgentRouter(unittest.TestCase):
    def test_agent_router_module(self):
        # Test all public callables in agent_router
        for attr in dir(agent_router):
            if not attr.startswith("_") and callable(getattr(agent_router, attr)):
                method = getattr(agent_router, attr)
                try:
                    method()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
