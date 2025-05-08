"""Database manager module for LG-ADK. Re-exports DatabaseManager from db_manager.py for backward compatibility."""

from lg_adk.database.db_manager import DatabaseManager

__all__ = ["DatabaseManager"]
