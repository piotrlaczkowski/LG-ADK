"""Base Agent class for LG-ADK."""

from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from lg_adk.memory import MemoryManager
from lg_adk.models import get_model
from lg_adk.sessions import SessionManager
from lg_adk.utils import condense_history


class Agent(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    """
    Base Agent class for LG-ADK with robust session, memory, async, and collaborative support.

    Attributes:
        name: The name of the agent.
        llm: The language model to use (can be a provider-prefixed name like "ollama/llama3").
        description: A description of the agent's purpose.
        tools: A list of tools the agent can use.
        system_message: System message to provide context to the agent.
        session_manager: Optional session manager for multi-user workflows.
        memory_manager: Optional memory manager for conversation history and memory.
        async_memory: Whether to use async memory operations.
        custom_nodes: Optional list of custom/collaborative workflow nodes.
    """

    name: str = Field(..., description="Name of the agent")
    llm: Any = Field(None, description="Language model to use")
    description: str = Field("A helpful AI assistant", description="Description of the agent's purpose")
    tools: list[BaseTool] = Field(default_factory=list, description="Tools available to the agent")
    system_message: str = Field("", description="System message to provide context to the agent")
    session_manager: SessionManager | None = None
    memory_manager: MemoryManager | None = None
    async_memory: bool = False
    custom_nodes: list[Callable] | None = None
    _model: Any = None

    def add_tool(self, tool: BaseTool) -> None:
        """Add a tool to the agent.

        Args:
            tool: The tool to add.
        """
        self.tools.append(tool)

    def add_tools(self, tools: list[BaseTool]) -> None:
        """Add multiple tools to the agent.

        Args:
            tools: List of tools to add.
        """
        self.tools.extend(tools)

    def create_prompt(self) -> str:
        """Create a prompt for the agent based on its configuration."""
        if self.system_message:
            return self.system_message

        prompt = f"You are {self.name}, {self.description}.\n\n"

        if self.tools:
            prompt += "You have access to the following tools:\n\n"
            for tool in self.tools:
                prompt += f"- {tool.name}: {tool.description}\n"

        return prompt

    def get_model(self) -> Any:
        """Get the underlying language model.

        Returns:
            The language model instance.
        """
        if self._model is None:
            if isinstance(self.llm, str):
                self._model = get_model(self.llm)
            else:
                # If it's already a model instance, use it directly
                self._model = self.llm

        return self._model

    def get_or_create_session(self, state: dict) -> str:
        """Get or create a session ID using the session manager."""
        if not self.session_manager:
            return state.get("session_id", "")
        session_id = state.get("session_id")
        if not session_id:
            session_id = self.session_manager.create_session()
        return session_id

    def get_history(self, session_id: str) -> list[dict]:
        """Retrieve conversation history for a session."""
        if self.memory_manager:
            return self.memory_manager.get_conversation_history(session_id)
        return []

    def update_memory(self, session_id: str, user_message: str, agent_message: str) -> None:
        """Update memory with the latest user and agent messages."""
        if self.memory_manager:
            self.memory_manager.add_message(session_id, {"role": "user", "content": user_message})
            self.memory_manager.add_message(session_id, {"role": "assistant", "content": agent_message})

    def summarize_history(self, history: list[dict], max_tokens: int = 8000) -> list[dict]:
        """Summarize or condense conversation history to fit within a token limit."""
        return condense_history(history, max_tokens)

    def run(
        self,
        user_input: str = "",
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        state: dict | None = None,
    ) -> dict:
        """Process the current state and return a new state, with session, memory, and extensible support.

        Args:
            user_input: The user input string.
            session_id: Optional session ID.
            metadata: Optional metadata dict.
            state: Optional full state dict (overrides user_input/session_id/metadata if provided).

        Returns:
            The updated state after the agent's processing.
        """
        if state is None:
            state = {
                "input": user_input,
                "session_id": session_id or "",
                "metadata": metadata or {},
            }
        # Session management
        session_id = self.get_or_create_session(state)
        state["session_id"] = session_id
        # Memory management
        history = self.get_history(session_id)
        state["conversation_history"] = history
        # Prompt creation
        prompt = self.create_prompt()
        prompt += f"\n\nUser: {state.get('input', '')}"
        # Model invocation
        model = self.get_model()
        try:
            response = model.invoke(prompt)
        except Exception as e:
            response = f"Error generating response: {str(e)}"
        state["output"] = response
        state["agent"] = self.name
        # Update memory
        self.update_memory(session_id, state.get("input", ""), response)
        # Custom/collaborative nodes
        if self.custom_nodes:
            for node in self.custom_nodes:
                state = node(state)
        return state

    async def arun(
        self,
        user_input: str = "",
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        state: dict | None = None,
    ) -> dict:
        """Async version of run(). Supports async model and memory operations."""
        if state is None:
            state = {
                "input": user_input,
                "session_id": session_id or "",
                "metadata": metadata or {},
            }
        # Session management
        session_id = self.get_or_create_session(state)
        state["session_id"] = session_id
        # Memory management
        history = self.get_history(session_id)
        state["conversation_history"] = history
        # Prompt creation
        prompt = self.create_prompt()
        prompt += f"\n\nUser: {state.get('input', '')}"
        # Model invocation
        model = self.get_model()
        try:
            if hasattr(model, "ainvoke"):
                response = await model.ainvoke(prompt)
            else:
                response = model.invoke(prompt)
        except Exception as e:
            response = f"Error generating response: {str(e)}"
        state["output"] = response
        state["agent"] = self.name
        # Update memory
        self.update_memory(session_id, state.get("input", ""), response)
        # Custom/collaborative nodes
        if self.custom_nodes:
            for node in self.custom_nodes:
                if callable(node):
                    state = await node(state) if callable(node) and hasattr(node, "__await__") else node(state)
        return state

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Make the agent callable directly.

        Args:
            state: The current state of the workflow.

        Returns:
            The updated state after the agent's processing.
        """
        return self.run(state=state)
