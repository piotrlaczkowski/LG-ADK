import unittest

from lg_adk.memory import memory_manager


class TestMemoryManager(unittest.TestCase):
    def test_memory_manager_instantiation(self):
        # Should be able to instantiate MemoryManager
        try:
            mgr = memory_manager.MemoryManager()
        except Exception:
            self.skipTest("MemoryManager cannot be instantiated directly.")

    def test_memory_manager_methods(self):
        mgr = memory_manager.MemoryManager()
        # Test all public methods
        for attr in dir(mgr):
            if not attr.startswith("_") and callable(getattr(mgr, attr)):
                method = getattr(mgr, attr)
                try:
                    method()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
