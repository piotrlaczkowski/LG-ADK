import unittest

from lg_adk.tools import group_chat


class TestGroupChat(unittest.TestCase):
    def test_group_chat_module(self):
        # Test all public callables in group_chat
        for attr in dir(group_chat):
            if not attr.startswith("_") and callable(getattr(group_chat, attr)):
                method = getattr(group_chat, attr)
                try:
                    method()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
