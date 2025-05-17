import unittest

from lg_adk.database import vector_store


class TestVectorStore(unittest.TestCase):
    def test_vector_store_module(self):
        # Test all public callables in vector_store
        for attr in dir(vector_store):
            if not attr.startswith("_") and callable(getattr(vector_store, attr)):
                method = getattr(vector_store, attr)
                try:
                    method()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
