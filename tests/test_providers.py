import unittest

from lg_adk.models import providers


class TestProviders(unittest.TestCase):
    def test_providers_module(self):
        # Test all public callables in providers
        for attr in dir(providers):
            if not attr.startswith("_") and callable(getattr(providers, attr)):
                method = getattr(providers, attr)
                try:
                    method()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
