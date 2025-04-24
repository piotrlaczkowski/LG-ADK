"""
Database manager module - re-exports DatabaseManager from db_manager.py.

This file exists for backward compatibility. New code should import from db_manager directly.
"""

from lg_adk.database.db_manager import DatabaseManager

__all__ = ["DatabaseManager"] 