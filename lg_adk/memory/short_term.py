import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

def get_short_term_memory(db_path: str = "memory.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)
    return memory
