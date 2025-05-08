import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver


def get_short_term_memory(db_path: str = "memory.db") -> SqliteSaver:
    """Get a short-term memory object backed by SQLite.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        An instance of SqliteSaver for short-term memory.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)
    return memory
