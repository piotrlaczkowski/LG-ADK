# Base Agent (lg_adk.agents.Agent)

The `Agent` class is the foundation for all agents in LG-ADK. It supports robust session management, memory, async workflows, and easy extension for collaborative and production-ready agents.

## Features
- **Session management**: Multi-user, persistent sessions via `SessionManager`.
- **Memory management**: Conversation history, memory update, and summarization via `MemoryManager`.
- **Async support**: Use `.arun()` for async model/memory operations.
- **Custom/collaborative nodes**: Plug in custom workflow steps for advanced or multi-agent behaviors.
- **Minimal API**: Create robust agents with just a config and a model.

## Usage
```python
from lg_adk.agents import Agent
from lg_adk.memory import MemoryManager
from lg_adk.sessions import SessionManager

agent = Agent(
    name="my_agent",
    llm="ollama/llama3",
    session_manager=SessionManager(),
    memory_manager=MemoryManager(),
    tools=[...],
    system_message="You are a helpful assistant.",
)

# Synchronous run
result = agent.run(user_input="Hello!", session_id="abc123")

# Asynchronous run
# import asyncio
# result = asyncio.run(agent.arun(user_input="Hello!", session_id="abc123"))
```

## Custom Nodes
You can add custom workflow steps (nodes) to extend agent behavior:
```python
def add_flag(state):
    state["custom_flag"] = True
    return state

agent = Agent(name="custom_agent", llm=DummyModel(), custom_nodes=[add_flag])
result = agent.run(user_input="Test")
assert result["custom_flag"] is True
```

## Advanced Features & Best Practices
- Use `session_manager` and `memory_manager` for production agents.
- Use `arun` for async models or memory backends.
- Extend with custom nodes for collaboration, analytics, or advanced workflows.
- The Agent class supports arbitrary types for managers and tools (no Pydantic schema errors).

## Troubleshooting
If you use custom managers or tools, the Agent class is configured to allow arbitrary types (no Pydantic schema errors). If you see errors about unknown types, ensure you are using the latest LG-ADK version.

## API Reference
- `run(user_input, session_id, metadata, state)`
- `arun(user_input, session_id, metadata, state)`
- `add_tool(tool)` / `add_tools(tools)`
- `get_or_create_session(state)`
- `get_history(session_id)`
- `update_memory(session_id, user_message, agent_message)`
- `summarize_history(history, max_tokens)`
