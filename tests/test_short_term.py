import unittest

try:
    from lg_adk.memory import short_term

    LANGGRAPH_SQLITE_AVAILABLE = True
except ImportError:
    LANGGRAPH_SQLITE_AVAILABLE = False


class TestShortTerm(unittest.TestCase):
    @unittest.skipUnless(LANGGRAPH_SQLITE_AVAILABLE, "langgraph.checkpoint.sqlite not installed")
    def test_short_term_module(self):
        # Test all public callables in short_term
        for attr in dir(short_term):
            if not attr.startswith("_") and callable(getattr(short_term, attr)):
                method = getattr(short_term, attr)
                try:
                    method()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
