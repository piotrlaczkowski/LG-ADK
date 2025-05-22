from typing import Any


def condense_history(history: list[dict[str, Any]], max_tokens: int) -> list[dict[str, Any]]:
    """Condense conversation history to fit within a max token limit.
    Older messages are summarized if needed, recent messages are preserved.

    Args:
        history: List of message dicts (with 'role' and 'content').
        max_tokens: Maximum allowed tokens (approximate, uses word count as proxy).

    Returns:
        Condensed list of message dicts.
    """

    def estimate_tokens(msg: dict[str, Any]) -> int:
        return len(msg.get("content", "").split())

    total_tokens = sum(estimate_tokens(msg) for msg in history)
    if total_tokens <= max_tokens:
        return history

    # Start from the end (most recent), keep adding until over limit
    condensed = []
    running_tokens = 0
    for msg in reversed(history):
        t = estimate_tokens(msg)
        if running_tokens + t > max_tokens:
            break
        condensed.append(msg)
        running_tokens += t

    # Summarize the older messages
    to_summarize = history[: len(history) - len(condensed)]
    if to_summarize:
        summary_content = summarize_messages(to_summarize)
        summary_msg = {"role": "system", "content": f"Summary of earlier conversation: {summary_content}"}
        condensed.append(summary_msg)

    # Return in original order
    return list(reversed(condensed))


def summarize_messages(messages: list[dict[str, Any]]) -> str:
    """Summarize a list of messages into a short string.
    (This is a simple implementation; replace with LLM call for production.).
    """
    # Simple heuristic: concatenate first/last lines, mention number of messages
    if not messages:
        return ""
    first = messages[0]["content"][:50]
    last = messages[-1]["content"][:50]
    n = len(messages)
    return f"({n} messages summarized) First: '{first}...' Last: '{last}...'"
