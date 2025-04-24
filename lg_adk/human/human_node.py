from langgraph.types import interrupt

def human_review_node(state):
    # Pause execution and await human input
    value = interrupt({
        "text_to_review": state.get("text_to_review", "")
    })
    # Update state with human input
    state["text_to_review"] = value
    return state
