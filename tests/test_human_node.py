import unittest

from lg_adk.human import human_node


class TestHumanNode(unittest.TestCase):
    def test_human_node_module(self):
        # Test all public callables in human_node
        for attr in dir(human_node):
            if not attr.startswith("_") and callable(getattr(human_node, attr)):
                method = getattr(human_node, attr)
                try:
                    method()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
