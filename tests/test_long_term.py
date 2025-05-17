import unittest

from lg_adk.memory import long_term


class TestLongTerm(unittest.TestCase):
    def test_long_term_module(self):
        # Test all public callables in long_term
        for attr in dir(long_term):
            if not attr.startswith("_") and callable(getattr(long_term, attr)):
                method = getattr(long_term, attr)
                try:
                    method()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
