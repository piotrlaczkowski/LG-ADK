from collections.abc import Callable
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from lg_adk.memory import MemoryManager
from lg_adk.sessions import SessionManager
from lg_adk.tools import HumanInLoopTool
from lg_adk.utils import condense_history


class RAGState(BaseModel):
    """State for AgentRAG workflows.
    Includes fields for debugging, collaboration, trace, and extensibility.
    """

    user_input: str = ""
    session_id: str = ""
    output: str = ""
    agent: str = ""
    memory: dict[str, Any] = Field(default_factory=dict)
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    original_query: str = ""
    enhanced_query: str = ""
    context: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    collaborators: list[str] = Field(
        default_factory=list,
        description="Other agents involved in the workflow.",
    )
    agent_actions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Actions taken by agents during the workflow.",
    )
    latency: float = 0.0
    token_usage: int = 0
    user_feedback: str | None = None
    debug: dict[str, Any] = Field(default_factory=dict)
    # Add more as needed for extensibility


class AgentRAGConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    """
    Configuration for AgentRAG.
    """
    model: str | Any
    vectorstore: Any  # Allow mocks and real vectorstores
    memory_manager: MemoryManager | None = None
    session_manager: SessionManager | None = None
    max_history_tokens: int = 8000
    enable_human_in_loop: bool = False
    async_memory: bool = False
    custom_nodes: list[Callable] | None = None
    debug: bool = False
    # Add more as needed


class AgentRAG:
    """High-level RAG agent for LG-ADK.

    Features:
    - Session management (multi-user, persistent)
    - Memory management (history, async, condensation)
    - Context retrieval from vector store
    - Human-in-the-loop support
    - Extensible with custom nodes
    - State trace/debugging/collaboration
    - Simple sync/async API
    """

    def __init__(self, config: AgentRAGConfig):
        """Initialize the AgentRAG with the given configuration."""
        self._config = config
        self.memory_manager = config.memory_manager or MemoryManager(
            memory_type="in_memory",
            max_tokens=config.max_history_tokens,
        )
        self.session_manager = config.session_manager or SessionManager()
        self.vectorstore = config.vectorstore
        self.model = config.model
        self.enable_human_in_loop = config.enable_human_in_loop
        self.async_memory = config.async_memory
        self.custom_nodes = config.custom_nodes or []
        self.debug = config.debug
        self._build_graph()

    def _build_graph(self) -> None:
        """Build the workflow graph for the RAG agent."""
        from lg_adk import GraphBuilder

        builder = GraphBuilder(state_class=RAGState)
        builder.add_memory(self.memory_manager)
        builder.configure_session_management(self.session_manager)
        builder.add_node("get_or_create_session", self.get_or_create_session)
        builder.add_node("retrieve_history", self.retrieve_history)
        builder.add_node("condense_history", self.condense_history)
        builder.add_node("enhance_query", self.enhance_query)
        builder.add_node("retrieve_context", self.retrieve_context)
        builder.add_node("advanced_summarizer", self.advanced_summarizer)
        builder.add_node("generate_response", self.generate_response)
        builder.add_node("followup_question_generator", self.followup_question_generator)
        builder.add_node("update_memory", self.update_memory)
        for node in self.custom_nodes:
            builder.add_node(node.__name__, node)
        if self.enable_human_in_loop:
            builder.add_node("human_in_loop", HumanInLoopTool())
        builder.add_edge("__start__", "get_or_create_session")
        builder.add_edge("get_or_create_session", "retrieve_history")
        builder.add_edge("retrieve_history", "condense_history")
        builder.add_edge("condense_history", "enhance_query")
        builder.add_edge("enhance_query", "retrieve_context")
        builder.add_edge("retrieve_context", "advanced_summarizer")
        builder.add_edge("advanced_summarizer", "generate_response")
        builder.add_edge("generate_response", "followup_question_generator")
        builder.add_edge("followup_question_generator", "update_memory")
        if self.enable_human_in_loop:
            builder.add_edge("update_memory", "human_in_loop")
            builder.add_edge("human_in_loop", "__end__")
        else:
            builder.add_edge("update_memory", "__end__")
        self.graph = builder.build()

    def get_or_create_session(self, state: RAGState) -> RAGState:
        """Get or create a session for the user."""
        session_id = state.session_id or self.session_manager.create_session()
        state.session_id = session_id
        if self.debug:
            logger.debug(f"[get_or_create_session] session_id: {session_id}")
        return state

    def retrieve_history(self, state: RAGState) -> RAGState:
        """Retrieve conversation history for the session."""
        history = self.memory_manager.get_conversation_history(state.session_id)
        state.conversation_history = history
        if self.debug:
            logger.debug(f"[retrieve_history] history: {history}")
        return state

    def condense_history(self, state: RAGState) -> RAGState:
        """Condense conversation history if it exceeds the token limit."""
        if self.memory_manager.token_count(state.conversation_history) > self._config.max_history_tokens:
            condensed = condense_history(
                state.conversation_history,
                self._config.max_history_tokens,
            )
            state.conversation_history = condensed
            if self.debug:
                logger.debug(f"[condense_history] condensed history: {condensed}")
        return state

    def enhance_query(self, state: RAGState) -> RAGState:
        """Enhance the user query using conversation history."""
        state.original_query = state.user_input
        state.enhanced_query = state.user_input  # Optionally call a model here
        if self.debug:
            logger.debug(
                f"[enhance_query] original_query: {state.original_query}, enhanced_query: {state.enhanced_query}",
            )
        return state

    def retrieve_context(self, state: RAGState) -> RAGState:
        """Retrieve relevant context from the vector store."""
        docs = self.vectorstore.similarity_search(state.enhanced_query, k=3)
        state.context = [doc.page_content for doc in docs]
        if self.debug:
            logger.debug(f"[retrieve_context] context: {state.context}")
        return state

    def advanced_summarizer(self, state: RAGState) -> RAGState:
        """Summarize long context or conversation history."""
        if len(" ".join(state.context).split()) > 500:
            prompt = f"Summarize the following context for the user:\n{state.context}"
            if self.async_memory:
                import asyncio

                summary = asyncio.run(self.model.ainvoke(prompt))
            else:
                summary = self.model.invoke(prompt)
            state.debug["context_summary"] = summary
            state.trace.append({"step": "advanced_summarizer", "output": summary})
        return state

    def generate_response(self, state: RAGState) -> RAGState:
        """Generate a response based on context and history."""
        prompt = (
            f"Context: {' '.join(state.context)}\n"
            f"History: {' '.join([msg['content'] for msg in state.conversation_history])}\n"
            f"User: {state.original_query}\n"
        )
        if self.async_memory:
            import asyncio

            state.output = asyncio.run(self.model.ainvoke(prompt))
        else:
            state.output = self.model.invoke(prompt)
        if self.debug:
            logger.debug(f"[generate_response] output: {state.output}")
        state.trace.append(
            {
                "step": "generate_response",
                "prompt": prompt,
                "output": state.output,
            },
        )
        return state

    def followup_question_generator(self, state: RAGState) -> RAGState:
        """Suggest a follow-up question if the context is insufficient."""
        if not state.context or "not enough information" in state.output.lower():
            prompt = (
                f"Given the user's question: {state.original_query}\n"
                f"and the context: {state.context}\n"
                "Suggest a clarifying follow-up question to help the user get a better answer."
            )
            if self.async_memory:
                import asyncio

                followup = asyncio.run(self.model.ainvoke(prompt))
            else:
                followup = self.model.invoke(prompt)
            state.debug["followup_question"] = followup
            state.trace.append({"step": "followup_question_generator", "output": followup})
        return state

    def update_memory(self, state: RAGState) -> RAGState:
        """Update memory with the latest user and assistant messages."""
        self.memory_manager.add_message(
            state.session_id,
            {"role": "user", "content": state.original_query},
        )
        self.memory_manager.add_message(
            state.session_id,
            {"role": "assistant", "content": state.output},
        )
        if self.debug:
            logger.debug(
                f"[update_memory] user: {state.original_query}, assistant: {state.output}",
            )
        return state

    def run(
        self,
        user_input: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the RAG agent synchronously.

        Args:
            user_input: User input string.
            session_id: Optional session ID.
            metadata: Optional metadata dict.

        Returns:
            Final state as dict.
        """
        initial_state = RAGState(
            user_input=user_input,
            session_id=session_id or "",
            metadata=metadata or {},
        )
        result = self.graph.invoke(initial_state)
        if self.debug:
            logger.debug(f"[run] final state: {result}")
        return result.model_dump() if hasattr(result, "model_dump") else dict(result)

    async def arun(
        self,
        user_input: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the RAG agent asynchronously.

        Args:
            user_input: User input string.
            session_id: Optional session ID.
            metadata: Optional metadata dict.

        Returns:
            Final state as dict.
        """
        initial_state = RAGState(
            user_input=user_input,
            session_id=session_id or "",
            metadata=metadata or {},
        )
        result = await self.graph.ainvoke(initial_state)
        if self.debug:
            logger.debug(f"[arun] final state: {result}")
        return result.model_dump() if hasattr(result, "model_dump") else dict(result)
