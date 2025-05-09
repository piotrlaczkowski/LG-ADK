"""Database module for LG-ADK. Provides classes for database interaction."""

from lg_adk.database.db_manager import DatabaseManager
from lg_adk.database.vector_store import VectorStore

try:
    from lg_adk.database.morphik_db import MORPHIK_AVAILABLE, MorphikDatabaseManager, MorphikDocument
except ImportError:
    # Create dummy variables for type checking if Morphik is not installed
    MORPHIK_AVAILABLE = False
    MorphikDatabaseManager = None
    MorphikDocument = None

__all__ = [
    "DatabaseManager",
    "VectorStore",
    "MorphikDatabaseManager",
    "MorphikDocument",
    "MORPHIK_AVAILABLE",
]
