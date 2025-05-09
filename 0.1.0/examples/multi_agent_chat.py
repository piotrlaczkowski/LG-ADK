#!/usr/bin/env python
"""
Multi-Agent Chat Example

This example demonstrates how to use the GroupChatTool and AgentRouter classes
to create a multi-agent system that can collaborate on tasks.
"""

import os
from typing import Any, Dict, List

from lg_adk import Agent, get_model
from lg_adk.tools.agent_router import AgentRouter, RouterType
from lg_adk.tools.group_chat import GroupChatTool


def create_specialized_agents() -> Dict[str, Agent]:
    """Create a set of specialized agents.

    Returns:
        Dictionary of specialized agents
    """
    # Get a model to use for all agents
    model = get_model("gpt-3.5-turbo")

    # Create a research agent
    research_agent = Agent(
        agent_name="ResearchAgent",
        system_prompt="You are a research specialist. Your role is to provide detailed, "
        "well-researched information about any topic. Focus on presenting "
        "factual information with relevant context and background.",
        llm=model,
    )

    # Create a code agent
    code_agent = Agent(
        agent_name="CodeAgent",
        system_prompt="You are a coding expert. Your role is to write, review, and explain "
        "code. You specialize in Python, JavaScript, and general software "
        "architecture. Provide clean, efficient code with proper documentation.",
        llm=model,
    )

    # Create a writing agent
    writing_agent = Agent(
        agent_name="WritingAgent",
        system_prompt="You are a writing specialist. Your role is to create well-structured, "
        "engaging content. You can write in various styles from formal to "
        "conversational and can adapt your tone to the target audience.",
        llm=model,
    )

    # Create a critic agent
    critic_agent = Agent(
        agent_name="CriticAgent",
        system_prompt="You are a critical thinker. Your role is to analyze information "
        "and identify potential issues, biases, or areas for improvement. "
        "Provide constructive criticism and alternative perspectives.",
        llm=model,
    )

    return {"research": research_agent, "code": code_agent, "writing": writing_agent, "critic": critic_agent}


def group_chat_example(agents: Dict[str, Agent], query: str) -> None:
    """Run a group chat example.

    Args:
        agents: Dictionary of agents
        query: The initial query
    """
    print("\n=== Running Group Chat Example ===\n")

    # Create the group chat tool
    chat_tool = GroupChatTool(agent_registry=agents)

    # Create a new chat with all agents
    chat_id = chat_tool.create_chat(
        name="Collaborative Problem Solving", agent_ids=list(agents.keys()), metadata={"topic": "Problem Solving"}
    )

    # Run a conversation starting with the research agent
    messages = chat_tool.run_conversation(chat_id=chat_id, initial_prompt=query, max_turns=4)

    # Print the conversation
    print(f"Group chat on query: {query}\n")
    for msg in messages:
        print(f"{msg.agent_id}: {msg.content}\n")


def agent_router_example(agents: Dict[str, Agent], query: str) -> None:
    """Run an agent router example.

    Args:
        agents: Dictionary of agents
        query: The initial query
    """
    print("\n=== Running Agent Router Example ===\n")

    # Create a sequential router
    sequential_router = AgentRouter(
        name="SequentialThoughtProcess", agents=list(agents.values()), router_type=RouterType.SEQUENTIAL
    )

    # Run the sequential router
    print(f"Sequential routing on query: {query}\n")
    sequential_result = sequential_router.run(query)
    print(f"Final output: {sequential_result.get('output', '')}\n")

    # Create a selector router
    selector_router = AgentRouter(name="ExpertSelector", agents=list(agents.values()), router_type=RouterType.SELECTOR)

    # Run the selector router
    print(f"Selector routing on query: {query}\n")
    selector_result = selector_router.run(query)
    selected_agent = selector_result.get("agent", "Unknown")
    print(f"Selected agent: {selected_agent}")
    print(f"Output: {selector_result.get('output', '')}\n")

    # Create a mixture router
    mixture_router = AgentRouter(
        name="CollaborativeThinking", agents=list(agents.values()), router_type=RouterType.MIXTURE
    )

    # Run the mixture router
    print(f"Mixture routing on query: {query}\n")
    mixture_result = mixture_router.run(query)
    print(f"Combined output: {mixture_result.get('output', '')}\n")


def main() -> None:
    """Run the examples."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable")
        return

    # Create specialized agents
    agents = create_specialized_agents()

    # Run examples with different queries
    group_chat_example(agents, "Explain how transformer models work and provide a simple example")
    agent_router_example(agents, "I need to create a Python function that calculates Fibonacci numbers")

    # More complex query for both examples
    complex_query = (
        "I'm building a web application that needs to process large datasets. "
        "Can you recommend an architecture and provide some sample code for "
        "handling data processing efficiently?"
    )
    group_chat_example(agents, complex_query)
    agent_router_example(agents, complex_query)


if __name__ == "__main__":
    main()
