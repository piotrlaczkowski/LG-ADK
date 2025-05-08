# Multi-Agent Communication

This guide explains how to use LG-ADK's multi-agent communication tools to build sophisticated agent systems that can collaborate effectively.

## Overview

LG-ADK provides two main tools for multi-agent communication:

1. **GroupChatTool**: For facilitating conversations between multiple agents with different specialties
2. **AgentRouter**: For routing tasks to the most appropriate agent or for sequential agent workflows

These tools enable you to create systems where multiple agents with different capabilities can work together to solve complex problems.

## GroupChatTool

The `GroupChatTool` enables agents to have conversations with each other, similar to how humans might collaborate in a group chat.

### Basic Usage

```python
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

### Custom Speaker Selection

By default, agents take turns speaking in a round-robin fashion. You can customize this behavior by providing a speaker selection function:

```python
def expertise_based_selection(chat, history):
    """Select the next speaker based on keyword expertise."""
    if not history:
        return chat.agents[0]

    last_message = history[-1].content.lower()

    # If last message mentions taxes, select the finance expert
    if "tax" in last_message or "finance" in last_message:
        return "finance"

    # If last message mentions legal terms, select the legal expert
    if "legal" in last_message or "law" in last_message:
        return "legal"

    # Default to alternating speakers
    last_speaker_idx = chat.agents.index(history[-1].agent_id)
    next_speaker_idx = (last_speaker_idx + 1) % len(chat.agents)
    return chat.agents[next_speaker_idx]

# Run conversation with custom speaker selection
messages = chat_tool.run_conversation(
    chat_id=chat_id,
    initial_prompt="What are the legal and tax implications of starting a business?",
    max_turns=6,
    speaker_selection=expertise_based_selection
)
```

## AgentRouter

The `AgentRouter` allows you to route tasks to the most appropriate agent and supports different routing strategies.

### Routing Strategies

- **SEQUENTIAL**: Process a task through a sequence of agents, where each agent builds on the previous agent's output
- **CONCURRENT**: Process a task with multiple agents in parallel and combine their outputs
- **SELECTOR**: Select the most appropriate agent for a specific task
- **MIXTURE**: Get results from multiple agents and combine them into a comprehensive response

### Basic Usage

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
```

### Selector Router

The selector router automatically chooses the best agent for a given task:

```python
from lg_adk.tools.agent_router import RouterType

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
```

### Custom Agent Selection

You can provide a custom agent selection function for the selector router:

```python
def keyword_based_selector(task, agents):
    """Select an agent based on keywords in the task."""
    task_lower = task.lower()

    if "research" in task_lower or "find" in task_lower:
        return next(a for a in agents if a.agent_name == "Researcher")

    if "write" in task_lower or "create" in task_lower:
        return next(a for a in agents if a.agent_name == "Writer")

    if "edit" in task_lower or "improve" in task_lower:
        return next(a for a in agents if a.agent_name == "Editor")

    # Default to the first agent
    return agents[0]

# Create a router with custom agent selection
custom_router = AgentRouter(
    name="CustomSelector",
    agents=[research_agent, writer_agent, editor_agent],
    router_type=RouterType.SELECTOR,
    agent_selector=keyword_based_selector
)

# The router will use your custom function to select an agent
result = custom_router.run("Research the history of artificial intelligence")
```

## Combining GroupChat and Router

You can combine both tools for more complex agent systems:

```python
# Create a team of specialized agents
agents = {
    "researcher": research_agent,
    "writer": writer_agent,
    "editor": editor_agent,
    "fact_checker": fact_checker_agent
}

# Create a group chat for initial brainstorming
chat_tool = GroupChatTool(agent_registry=agents)
chat_id = chat_tool.create_chat(
    name="Content Planning",
    agent_ids=list(agents.keys())
)

# Run a planning conversation
planning_messages = chat_tool.run_conversation(
    chat_id=chat_id,
    initial_prompt="We need to create content about renewable energy. Let's plan our approach.",
    max_turns=8
)

# Extract the plan from the last message
plan = planning_messages[-1].content

# Create a sequential router for execution
execution_router = AgentRouter(
    name="ContentExecution",
    agents=[agents["researcher"], agents["writer"], agents["editor"], agents["fact_checker"]],
    router_type=RouterType.SEQUENTIAL
)

# Execute the plan
final_content = execution_router.run(f"Based on this plan: {plan}\nCreate content about renewable energy")
print(final_content["output"])
```

## Conclusion

The multi-agent communication tools in LG-ADK allow you to build sophisticated agent systems where agents can collaborate effectively. Whether you need agents to discuss a problem in a group chat or process tasks in a specific sequence, these tools provide the flexibility to create the right architecture for your needs.
