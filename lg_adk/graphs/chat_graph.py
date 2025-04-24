from langgraph.graph import StateGraph
from langgraph_framework.agents.base_agent import Agent
from langgraph_framework.human.human_node import human_review_node

def graph():
    graph_builder = StateGraph()
    # Add nodes and edges
    graph_builder.add_node("human_review", human_review_node)
    # Compile and return the graph
    return graph_builder.compile()
