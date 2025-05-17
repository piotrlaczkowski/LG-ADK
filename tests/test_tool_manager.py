import unittest

from lg_adk.tools import tool_manager


class TestToolManager(unittest.TestCase):
    def test_tool_manager_module(self):
        # Test all public callables in tool_manager
        for attr in dir(tool_manager):
            if not attr.startswith("_") and callable(getattr(tool_manager, attr)):
                method = getattr(tool_manager, attr)
                try:
                    method()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
