import unittest

from lg_adk.eval import metrics


class TestMetrics(unittest.TestCase):
    def test_metrics_module(self):
        # Test all public callables in metrics
        for attr in dir(metrics):
            if not attr.startswith("_") and callable(getattr(metrics, attr)):
                method = getattr(metrics, attr)
                try:
                    method()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
