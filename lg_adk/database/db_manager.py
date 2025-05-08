"""Database manager for LGBuilder."""

import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DatabaseManager(BaseModel):
    """Manager for database interactions.

    Attributes:
        db_url: Database connection URL.
        db_type: Type of database to use.
    """

    db_url: str = Field("sqlite:///lgbuilder.db", description="Database connection URL")
    db_type: str = Field("sqlite", description="Type of database")

    def __init__(self, **data: Any) -> None:
        """Initialize the DatabaseManager.

        Args:
            **data: Initialization data for the model.
        """
        super().__init__(**data)
        self._conn = None

        # Initialize the database if it's SQLite
        if self.db_type == "sqlite":
            self._initialize_sqlite()

    def _initialize_sqlite(self) -> None:
        """Initialize SQLite database with required tables."""
        if self.db_url.startswith("sqlite:///"):
            db_path = self.db_url.replace("sqlite:///", "")

            # Create directory if it doesn't exist using pathlib
            db_file = Path(db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)

            # Connect to the database
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()

            # Create tables
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    created_at TIMESTAMP,
                    last_active TIMESTAMP,
                    metadata TEXT
                )
            """,
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    content TEXT,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """,
            )

            # Commit and close
            conn.commit()
            conn.close()

    def _get_connection(self) -> Any:
        """Get a database connection.

        Returns:
            A database connection object.

        Raises:
            ValueError: If the database type is unsupported.
        """
        if self.db_type == "sqlite" and self.db_url.startswith("sqlite:///"):
            db_path = self.db_url.replace("sqlite:///", "")
            return sqlite3.connect(db_path)

        # In a real implementation, this would support more database types
        raise ValueError(f"Unsupported database type: {self.db_type}")

    def execute_query(
        self,
        query: str,
        params: tuple[Any, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a database query.

        Args:
            query: SQL query to execute.
            params: Parameters for the query.

        Returns:
            Results of the query as a list of dictionaries.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            # Get column names
            column_names = [description[0] for description in cursor.description] if cursor.description else []

            # Convert results to dictionaries
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(column_names, row, strict=False)))

            conn.commit()
            return results
        finally:
            conn.close()

    def execute_update(
        self,
        query: str,
        params: tuple[Any, ...] | None = None,
    ) -> int:
        """Execute a database update query.

        Args:
            query: SQL query to execute.
            params: Parameters for the query.

        Returns:
            Number of rows affected.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def save_session(
        self,
        session_id: str,
        user_id: str | None,
        metadata: dict[str, Any],
    ) -> None:
        """Save or update a session in the database.

        Args:
            session_id: The ID of the session.
            user_id: Optional user identifier.
            metadata: Session metadata.
        """
        # Convert metadata to string (in a real implementation, this would be JSON)
        metadata_str = str(metadata)

        # Check if the session exists
        results = self.execute_query(
            "SELECT id FROM sessions WHERE id = ?",
            (session_id,),
        )

        if results:
            # Update existing session
            self.execute_update(
                "UPDATE sessions SET user_id = ?, last_active = CURRENT_TIMESTAMP, metadata = ? WHERE id = ?",
                (user_id, metadata_str, session_id),
            )
        else:
            # Insert new session
            self.execute_update(
                (
                    "INSERT INTO sessions (id, user_id, created_at, last_active, metadata) "
                    "VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)"
                ),
                (session_id, user_id, metadata_str),
            )

    def save_memory(self, session_id: str, content: str) -> None:
        """Save a memory entry for a session.

        Args:
            session_id: The ID of the session.
            content: The memory content.
        """
        self.execute_update(
            "INSERT INTO memory (session_id, content, timestamp) VALUES (?, ?, CURRENT_TIMESTAMP)",
            (session_id, content),
        )

    def get_session_memory(self, session_id: str) -> list[dict[str, Any]]:
        """Get all memory entries for a session.

        Args:
            session_id: The ID of the session.

        Returns:
            List of memory entries.
        """
        return self.execute_query(
            "SELECT * FROM memory WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        )
