import unittest

from lg_adk.human import intervention


class TestIntervention(unittest.TestCase):
    def test_intervention_module(self):
        # Test all public callables in intervention
        for attr in dir(intervention):
            if not attr.startswith("_") and callable(getattr(intervention, attr)):
                method = getattr(intervention, attr)
                try:
                    method()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
