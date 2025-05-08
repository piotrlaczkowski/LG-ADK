from langgraph.store.memory import InMemoryStore


def get_long_term_memory() -> InMemoryStore:
    """Create and return an in-memory store for long-term memory.

    Returns:
        InMemoryStore: The in-memory store instance for long-term memory.
    """
    store = InMemoryStore()
    return store
