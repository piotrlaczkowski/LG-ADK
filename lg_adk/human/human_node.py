from langgraph.types import interrupt


def human_review_node(state: dict) -> dict:
    """Pause execution for human review and update state with human input.

    Args:
        state (dict): The current state containing 'text_to_review'.

    Returns:
        dict: The updated state after human review.
    """
    # Pause execution and await human input
    value = interrupt(
        {
            "text_to_review": state.get("text_to_review", ""),
        },
    )
    # Update state with human input
    state["text_to_review"] = value
    return state
