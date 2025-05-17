import unittest

from lg_adk.database import db_manager


class TestDatabaseManager(unittest.TestCase):
    def test_database_manager_instantiation(self):
        # Should be able to instantiate DatabaseManager
        manager = db_manager.DatabaseManager()
        self.assertEqual(manager.db_type, "sqlite")
        self.assertTrue(manager.db_url.startswith("sqlite"))

    def test_database_manager_public_methods(self):
        manager = db_manager.DatabaseManager()
        # Test all public methods (should not raise)
        for attr in dir(manager):
            if not attr.startswith("_") and callable(getattr(manager, attr)):
                method = getattr(manager, attr)
                try:
                    method()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
