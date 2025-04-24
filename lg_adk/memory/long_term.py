from langgraph.store.memory import InMemoryStore

def get_long_term_memory():
    store = InMemoryStore()
    return store
