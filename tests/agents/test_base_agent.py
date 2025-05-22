import pytest

from lg_adk.agents.base import Agent
from lg_adk.memory import MemoryManager
from lg_adk.sessions import SessionManager


class DummyModel:
    def invoke(self, prompt):
        return f"Echo: {prompt.strip()}"

    async def ainvoke(self, prompt):
        return f"AsyncEcho: {prompt.strip()}"


def test_simple_run():
    agent = Agent(name="test_agent", llm=DummyModel())
    result = agent.run(user_input="Hello!")
    assert "Echo:" in result["output"]
    assert result["agent"] == "test_agent"


def test_session_and_memory():
    memory = MemoryManager()
    session = SessionManager()
    agent = Agent(name="mem_agent", llm=DummyModel(), memory_manager=memory, session_manager=session)
    result = agent.run(user_input="Hi", session_id="sess1")
    assert result["session_id"] == "sess1"
    history = memory.get_conversation_history("sess1")
    assert history == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": result["output"]},
    ]


def test_custom_node():
    def add_flag(state):
        state["custom_flag"] = True
        return state

    agent = Agent(name="custom_agent", llm=DummyModel(), custom_nodes=[add_flag])
    result = agent.run(user_input="Test")
    assert result["custom_flag"] is True


@pytest.mark.asyncio
async def test_async_run():
    agent = Agent(name="async_agent", llm=DummyModel())
    result = await agent.arun(user_input="Async!")
    assert "AsyncEcho:" in result["output"]
