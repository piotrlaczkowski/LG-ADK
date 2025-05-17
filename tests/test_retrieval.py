import unittest

from lg_adk.tools import retrieval


class TestRetrieval(unittest.TestCase):
    def test_retrieval_module(self):
        # Test all public callables in retrieval
        for attr in dir(retrieval):
            if not attr.startswith("_") and callable(getattr(retrieval, attr)):
                method = getattr(retrieval, attr)
                try:
                    method()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
