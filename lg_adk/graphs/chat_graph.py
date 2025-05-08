from langgraph.graph import StateGraph
from langgraph_framework.human.human_node import human_review_node


def graph() -> StateGraph:
    """Create and compile a chat graph with a human review node.

    Returns:
        StateGraph: The compiled chat graph.
    """
    graph_builder = StateGraph()
    # Add nodes and edges
    graph_builder.add_node("human_review", human_review_node)
    # Compile and return the graph
    return graph_builder.compile()
