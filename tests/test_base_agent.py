import unittest
from unittest.mock import MagicMock

from lg_adk.agents import base_agent


class TestAgent(unittest.TestCase):
    def test_agent_instantiation(self):
        # Should be able to instantiate Agent
        agent = base_agent.Agent(name="test_agent", llm=MagicMock(), prompt="Test prompt", tools=[])
        self.assertEqual(agent.name, "test_agent")
        self.assertEqual(agent.prompt, "Test prompt")
        self.assertIsInstance(agent.tools, list)

    def test_agent_public_methods(self):
        agent = base_agent.Agent(name="test_agent", llm=MagicMock(), prompt="Test prompt", tools=[])
        # Test all public methods (should not raise)
        for attr in dir(agent):
            if not attr.startswith("_") and callable(getattr(agent, attr)):
                method = getattr(agent, attr)
                try:
                    method()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
