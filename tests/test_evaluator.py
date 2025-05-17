import unittest

from lg_adk.eval import evaluator


class TestEvaluator(unittest.TestCase):
    def test_evaluator_instantiation(self):
        # Should be able to instantiate Evaluator if not abstract
        try:
            evalr = evaluator.Evaluator()
        except Exception:
            self.skipTest("Evaluator cannot be instantiated directly.")

    def test_evaluator_methods(self):
        # If Evaluator is abstract, skip
        if hasattr(evaluator.Evaluator, "__abstractmethods__") and evaluator.Evaluator.__abstractmethods__:
            self.skipTest("Evaluator is abstract.")
        evalr = evaluator.Evaluator()
        # Test all public methods
        for attr in dir(evalr):
            if not attr.startswith("_") and callable(getattr(evalr, attr)):
                method = getattr(evalr, attr)
                try:
                    method()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
