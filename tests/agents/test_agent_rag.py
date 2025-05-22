import pytest
from pydantic import BaseModel

from lg_adk.agents.agent_rag import AgentRAG, AgentRAGConfig, RAGState


class DummyModel:
    def invoke(self, prompt):
        return f"Echo: {prompt.strip()}"

    async def ainvoke(self, prompt):
        return f"AsyncEcho: {prompt.strip()}"


class DummyVectorStore:
    def similarity_search(self, query, k=3):
        class Doc:
            def __init__(self, content):
                self.page_content = content

        return [Doc(f"Doc for {query}") for _ in range(k)]


@pytest.fixture
def rag_agent():
    config = AgentRAGConfig(
        model=DummyModel(),
        vectorstore=DummyVectorStore(),
        enable_human_in_loop=False,
        async_memory=False,
        debug=True,
    )
    return AgentRAG(config)


def test_sync_run(rag_agent):
    result = rag_agent.run("Test input", session_id="test-session")
    assert "output" in result
    assert result["output"].startswith("Echo:")
    assert result["session_id"] == "test-session"
    assert isinstance(result["trace"], list)


def test_trace_and_debug(rag_agent):
    result = rag_agent.run("Trace test", session_id="trace-session")
    assert any(step["step"] == "generate_response" for step in result["trace"])


def test_state_fields(rag_agent):
    result = rag_agent.run("Field test", session_id="field-session")
    assert result["original_query"] == "Field test"
    assert result["enhanced_query"] == "Field test"
    assert isinstance(result["context"], list)


@pytest.mark.asyncio
def test_async_run():
    config = AgentRAGConfig(
        model=DummyModel(),
        vectorstore=DummyVectorStore(),
        enable_human_in_loop=False,
        async_memory=True,
        debug=True,
    )
    rag_agent = AgentRAG(config)
    import asyncio

    result = asyncio.run(rag_agent.arun("Async input", session_id="async-session"))
    assert result["output"].startswith("AsyncEcho:")
    assert result["session_id"] == "async-session"
