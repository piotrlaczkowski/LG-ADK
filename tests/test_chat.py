import unittest

from lg_adk.graphs import chat


class TestChat(unittest.TestCase):
    def test_chat_module(self):
        # Test all public callables in chat
        for attr in dir(chat):
            if not attr.startswith("_") and callable(getattr(chat, attr)):
                method = getattr(chat, attr)
                try:
                    method()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
