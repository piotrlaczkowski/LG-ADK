# Multi-Agent Workflow Examples

This section provides examples of building multi-agent systems using LG-ADK.

## Group Chat Example

The following example demonstrates how to create a group chat where multiple agents can collaborate:

```python
import os
from typing import Dict, Any, List

from lg_adk import Agent, get_model
from lg_adk.tools.group_chat import GroupChatTool

# Create specialized agents
finance_agent = Agent(
    agent_name="FinanceExpert",
    system_prompt="You are a financial expert. Provide accurate financial advice.",
    llm=get_model("gpt-4")
)

legal_agent = Agent(
    agent_name="LegalExpert",
    system_prompt="You are a legal expert. Provide accurate legal advice.",
    llm=get_model("gpt-4")
)

# Create a registry of agents
agents = {
    "finance": finance_agent,
    "legal": legal_agent
}

# Create the group chat tool
chat_tool = GroupChatTool(agent_registry=agents)

# Create a new chat
chat_id = chat_tool.create_chat(
    name="Financial Legal Consultation",
    agent_ids=["finance", "legal"]
)

# Run a conversation
messages = chat_tool.run_conversation(
    chat_id=chat_id,
    initial_prompt="What are the tax implications of starting a small business?",
    max_turns=4  # Number of turns in the conversation
)

# Print the conversation
for msg in messages:
    print(f"{msg.agent_id}: {msg.content}")
```

## Agent Router Example

This example shows how to use the AgentRouter to route tasks to different agents based on their specialties:

```python
from lg_adk import Agent, get_model
from lg_adk.tools.agent_router import AgentRouter, RouterType

# Create specialized agents
research_agent = Agent(
    agent_name="Researcher",
    system_prompt="You are a research specialist. Find and present factual information.",
    llm=get_model("gpt-4")
)

writer_agent = Agent(
    agent_name="Writer",
    system_prompt="You are a writing specialist. Create well-structured content.",
    llm=get_model("gpt-4")
)

editor_agent = Agent(
    agent_name="Editor",
    system_prompt="You are an editor. Improve content for clarity and correctness.",
    llm=get_model("gpt-4")
)

# Create a sequential router (research -> write -> edit)
sequential_router = AgentRouter(
    name="ContentCreationPipeline",
    agents=[research_agent, writer_agent, editor_agent],
    router_type=RouterType.SEQUENTIAL
)

# Run a task through the sequential pipeline
result = sequential_router.run("Explain how blockchain technology works")
print(result["output"])

# Create a selector router
selector_router = AgentRouter(
    name="ExpertSelector",
    agents=[research_agent, writer_agent, editor_agent],
    router_type=RouterType.SELECTOR
)

# The router will select the most appropriate agent based on the task
result = selector_router.run("Research the latest advances in quantum computing")
print(f"Selected agent: {result.get('agent', 'Unknown')}")
print(f"Output: {result.get('output', '')}")

# Create a mixture router
mixture_router = AgentRouter(
    name="CollaborativeThinking",
    agents=[research_agent, writer_agent, editor_agent],
    router_type=RouterType.MIXTURE
)

# Get results from all agents
result = mixture_router.run("What are the best practices for writing technical documentation?")
print(result["output"])
```

## Complete Multi-Agent Workflow

This example demonstrates a complete multi-agent workflow that combines different types of collaboration:

```python
import os
from typing import Dict, Any, List

from lg_adk import Agent, get_model, GraphBuilder
from lg_adk.tools.group_chat import GroupChatTool
from lg_adk.tools.agent_router import AgentRouter, RouterType

# Create a team of specialized agents
research_agent = Agent(
    agent_name="Researcher",
    system_prompt="You are a research specialist. Find relevant information on any topic.",
    llm=get_model("gpt-4")
)

writer_agent = Agent(
    agent_name="Writer",
    system_prompt="You are a writing specialist. Create engaging, well-structured content.",
    llm=get_model("gpt-4")
)

editor_agent = Agent(
    agent_name="Editor",
    system_prompt="You are an editor. Review and improve content for clarity and correctness.",
    llm=get_model("gpt-4")
)

fact_checker_agent = Agent(
    agent_name="FactChecker",
    system_prompt="You are a fact checker. Verify the accuracy of information.",
    llm=get_model("gpt-4")
)

# Register all agents
agents = {
    "researcher": research_agent,
    "writer": writer_agent,
    "editor": editor_agent,
    "fact_checker": fact_checker_agent
}

# Phase 1: Planning using group chat
def planning_phase(topic):
    """Use group chat for collaborative planning."""
    print("\n=== Phase 1: Planning ===\n")

    chat_tool = GroupChatTool(agent_registry=agents)
    chat_id = chat_tool.create_chat(
        name="ContentPlanning",
        agent_ids=["researcher", "writer", "editor"]
    )

    messages = chat_tool.run_conversation(
        chat_id=chat_id,
        initial_prompt=f"We need to create comprehensive content about {topic}. Let's plan our approach.",
        max_turns=6
    )

    # Extract plan from the last message
    plan = messages[-1].content
    print("Planning completed:")
    for msg in messages:
        print(f"{msg.agent_id}: {msg.content}\n")

    return plan

# Phase 2: Research and content creation
def research_and_create_phase(topic, plan):
    """Use sequential router for research and content creation."""
    print("\n=== Phase 2: Research and Content Creation ===\n")

    # Create a sequential router for research and writing
    creation_router = AgentRouter(
        name="ContentCreation",
        agents=[agents["researcher"], agents["writer"]],
        router_type=RouterType.SEQUENTIAL
    )

    result = creation_router.run(
        f"Based on this plan: {plan}\nResearch and create content about {topic}"
    )

    draft_content = result.get("output", "")
    print(f"Draft content created:\n{draft_content}\n")

    return draft_content

# Phase 3: Review and improvement
def review_phase(draft_content):
    """Use mixture router for review and improvement."""
    print("\n=== Phase 3: Review and Improvement ===\n")

    # Create a mixture router for review
    review_router = AgentRouter(
        name="ContentReview",
        agents=[agents["editor"], agents["fact_checker"]],
        router_type=RouterType.MIXTURE
    )

    result = review_router.run(
        f"Review and improve this content:\n{draft_content}"
    )

    final_content = result.get("output", "")
    print(f"Final content:\n{final_content}\n")

    return final_content

# Main workflow function
def multi_agent_content_workflow(topic):
    """Run the complete multi-agent content creation workflow."""
    # Phase 1: Planning
    plan = planning_phase(topic)

    # Phase 2: Research and content creation
    draft_content = research_and_create_phase(topic, plan)

    # Phase 3: Review and improvement
    final_content = review_phase(draft_content)

    return final_content

# Run the workflow
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable")
    else:
        result = multi_agent_content_workflow("artificial intelligence ethics")
        print("\n=== Workflow Completed ===\n")
        print(result)
```

## Graph-Based Multi-Agent Workflow

This example demonstrates how to use LG-ADK's GraphBuilder to create a more complex multi-agent workflow:

```python
from lg_adk import Agent, GraphBuilder
from lg_adk.tools.agent_router import AgentRouter, RouterType

# Create specialized agents (code omitted for brevity)

# Define node functions
def planning_node(state):
    """Plan the content creation approach."""
    topic = state.get("input", "")

    planning_agent = Agent(
        agent_name="Planner",
        system_prompt="You create detailed content plans.",
        llm=get_model("gpt-4")
    )

    result = planning_agent.run({
        "input": f"Create a detailed plan for content about: {topic}"
    })

    plan = result.get("output", "")
    return {"topic": topic, "plan": plan}

def research_node(state):
    """Research the topic."""
    topic = state.get("topic", "")
    plan = state.get("plan", "")

    research_agent = Agent(
        agent_name="Researcher",
        system_prompt="You find factual information on any topic.",
        llm=get_model("gpt-4")
    )

    result = research_agent.run({
        "input": f"Research this topic based on the plan:\nTopic: {topic}\nPlan: {plan}"
    })

    research = result.get("output", "")
    return {"topic": topic, "plan": plan, "research": research}

def writing_node(state):
    """Write content based on research."""
    topic = state.get("topic", "")
    plan = state.get("plan", "")
    research = state.get("research", "")

    writer_agent = Agent(
        agent_name="Writer",
        system_prompt="You create well-structured content.",
        llm=get_model("gpt-4")
    )

    result = writer_agent.run({
        "input": f"Write content based on:\nTopic: {topic}\nPlan: {plan}\nResearch: {research}"
    })

    draft = result.get("output", "")
    return {"topic": topic, "plan": plan, "research": research, "draft": draft}

def editing_node(state):
    """Edit and improve the draft."""
    draft = state.get("draft", "")

    editor_agent = Agent(
        agent_name="Editor",
        system_prompt="You improve content for clarity and correctness.",
        llm=get_model("gpt-4")
    )

    result = editor_agent.run({
        "input": f"Edit and improve this draft:\n{draft}"
    })

    final_content = result.get("output", "")
    return {"output": final_content}

# Build the graph
builder = GraphBuilder()
builder.add_node("planning", planning_node)
builder.add_node("research", research_node)
builder.add_node("writing", writing_node)
builder.add_node("editing", editing_node)

# Define the flow
flow = [
    (None, "planning"),
    ("planning", "research"),
    ("research", "writing"),
    ("writing", "editing"),
    ("editing", None)
]

# Build and use the graph
content_graph = builder.build(flow=flow)
result = content_graph.invoke({"input": "renewable energy"})
print(result["output"])
```

## Full Examples

For more detailed examples, see the full code in the `docs/examples` directory:

- [multi_agent_chat.py](https://github.com/yourusername/lg-adk/blob/main/docs/examples/multi_agent_chat.py): A complete example of group chat and router implementations
