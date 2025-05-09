"""GraphBuilder for creating LangGraph workflows with proper session management."""

import asyncio
import datetime
import logging
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Generic, TypeVar

from langgraph.graph import END, Graph, StateGraph
from pydantic import BaseModel, Field, create_model

from lg_adk.agents.base import Agent
from lg_adk.memory.memory_manager import MemoryManager
from lg_adk.sessions.session_manager import Session, SessionManager

# Type variable for state
T = TypeVar("T")

# Configure logger
logger = logging.getLogger(__name__)


class GraphState(BaseModel):
    """Base state schema for graphs with session management."""

    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, Any]] = Field(default_factory=list)

    class Config:
        """Configuration for the model."""

        arbitrary_types_allowed = True
        extra = "allow"


class GraphBuilder(BaseModel, Generic[T]):
    """Builder for creating LangGraph workflows with proper session and state management."""

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",
    }

    name: str = Field("default_graph", description="Name of the graph")
    agents: list[Agent] = Field(default_factory=list, description="Agents in the graph")
    memory_manager: MemoryManager | None = Field(
        None,
        description="Memory manager for storing conversation history",
    )
    session_manager: SessionManager | None = Field(
        None,
        description="Session manager for managing sessions",
    )
    human_in_loop: bool = Field(False, description="Whether human intervention is enabled")
    nodes: dict[str, Any] = Field(default_factory=dict, description="Nodes in the graph")
    edges: list[dict[str, str]] = Field(default_factory=list, description="Edges between nodes")
    conditional_edges: list[dict[str, Any]] = Field(default_factory=list, description="Conditional edges")
    state_tracking: dict[str, bool] = Field(
        default_factory=lambda: {
            "include_session_id": True,
            "include_metadata": True,
            "include_messages": True,
        },
        description="Configuration for state tracking",
    )
    message_handlers: list[Callable] = Field(default_factory=list, description="Handlers for message processing")
    entry_point: str | None = Field(None, description="Entry point for the graph")
    exit_point: str | None = Field(None, description="Exit point for the graph")
    active_sessions: set[str] = Field(default_factory=set, description="Set of active session IDs")
    graph: Graph | None = Field(None, description="The built LangGraph graph")

    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the graph."""
        self.agents.append(agent)
        self.nodes[agent.name] = agent

    def add_memory(self, memory_manager: MemoryManager) -> None:
        """Add memory manager to the graph."""
        self.memory_manager = memory_manager

    def configure_session_management(self, session_manager: SessionManager) -> None:
        """Configure the session manager for the graph."""
        self.session_manager = session_manager

    def enable_human_in_loop(self, human_manager=None) -> None:
        """Enable human-in-the-loop for the graph."""
        self.human_in_loop = True
        # Store the human manager if provided
        if human_manager:
            self.human_manager = human_manager

    def enable_human_feedback(self, feedback_handler=None) -> None:
        """Enable human feedback for the graph."""
        self.human_feedback = True
        # Store the feedback handler if provided
        if feedback_handler:
            self.feedback_handler = feedback_handler

    def configure_state_tracking(
        self,
        include_session_id: bool = True,
        include_metadata: bool = True,
        include_messages: bool = True,
    ) -> None:
        """Configure state tracking options for the graph."""
        self.state_tracking = {
            "include_session_id": include_session_id,
            "include_metadata": include_metadata,
            "include_messages": include_messages,
        }

    def add_node(self, name: str, node_func: Any) -> None:
        """Add a node to the graph."""
        self.nodes[name] = node_func

    def add_edge(self, source: str, target: str) -> None:
        """Add an edge between nodes."""
        source_name = "__start__" if source is None or source == "START" else source
        target_name = target if target != "END" and target is not None else END
        self.edges.append({"source": source_name, "target": target_name})

    def add_conditional_edge(self, source: str, condition_function: Callable, targets: list[str]) -> None:
        """Add a conditional edge."""
        self.conditional_edges.append(
            {
                "source": source,
                "condition": condition_function,
                "targets": targets,
            },
        )

    def add_conditional_edges(self, source: str, condition_function: Callable, mapping: dict[str, str]) -> None:
        """Add conditional edges based on a named condition function."""
        self.conditional_edges.append(
            {
                "name": source,
                "function": condition_function,
                "mapping": mapping,
            },
        )

    def on_message(self, handler: Callable) -> None:
        """Register a message handler function."""
        self.message_handlers.append(handler)

    def set_entry_point(self, node_name: str) -> None:
        """Set the entry point for the graph."""
        self.entry_point = node_name

    def set_exit_point(self, node_name: str) -> None:
        """Set the exit point for the graph."""
        self.exit_point = node_name

    def add_human_node(self, human_node: Any, name: str | None = None) -> None:
        """Add a human node to the graph."""
        node_name = name or getattr(human_node, "name", "human")
        self.nodes[node_name] = human_node

    def create_session(
        self,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        timeout: int | None = 3600,
    ) -> str:
        """Create a new session and return its ID."""
        if not self.session_manager:
            raise ValueError("Session manager must be configured before creating sessions")

        session_id = self.session_manager.create_session(user_id=user_id, metadata=metadata, timeout=timeout)
        self.active_sessions.add(session_id)
        return session_id

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        if not self.session_manager:
            raise ValueError("Session manager not configured")
        return self.session_manager.get_session(session_id)

    def update_session_metadata(self, session_id: str, metadata: dict[str, Any], merge: bool = True) -> None:
        """Update session metadata."""
        if not self.session_manager:
            raise ValueError("Session manager not configured")
        self.session_manager.update_session_metadata(session_id, metadata, merge=merge)

    def end_session(self, session_id: str) -> None:
        """End a session and clean up resources."""
        if not self.session_manager:
            raise ValueError("Session manager not configured")
        self.session_manager.remove_session(session_id)
        self.active_sessions.discard(session_id)
        if self.memory_manager:
            try:
                self.memory_manager.clear_session_memories(session_id)
            except Exception as e:
                logger.warning(f"Error clearing memories for session {session_id}: {e}")

    def clear_expired_sessions(self) -> list[str]:
        """Clear expired sessions and return their IDs."""
        if not self.session_manager:
            raise ValueError("Session manager not configured")
        expired_sessions = self.session_manager.clear_expired_sessions()
        for session_id in expired_sessions:
            self.active_sessions.discard(session_id)
            if self.memory_manager:
                try:
                    self.memory_manager.clear_session_memories(session_id)
                except Exception as e:
                    logger.warning(f"Error clearing memories for expired session {session_id}: {e}")
        return expired_sessions

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        """Get the message history for a session."""
        if not self.memory_manager:
            raise ValueError("Memory manager not configured, cannot retrieve session history")
        return self.memory_manager.get_session_messages(session_id)

    def _create_state_schema(self) -> type:
        """Create the state schema for the graph with proper session tracking."""
        fields = {
            "input": (str, ...),
            "output": (str, ...),
            "agent": (str, ...),
            "memory": (dict, ...),
            "human_input": (str | None, None),
        }
        if self.state_tracking.get("include_session_id", True):
            fields["session_id"] = (str, None)
        if self.state_tracking.get("include_metadata", True):
            fields["metadata"] = (dict, {})
        if self.state_tracking.get("include_messages", True):
            fields["messages"] = (list, [])
        return create_model("DynamicGraphState", **fields)

    def _create_router(self) -> Callable[[dict[str, Any]], str | list[str]]:
        """Create a router function for the graph."""
        agent_names = [agent.name for agent in self.agents]

        def router(state: dict[str, Any]) -> str | list[str]:
            """Route to the next agent or end the graph."""
            current_agent = state.get("agent")
            if not current_agent:
                return self.entry_point if self.entry_point else (agent_names[0] if agent_names else END)
            if current_agent == self.exit_point:
                return END
            if current_agent in agent_names:
                current_idx = agent_names.index(current_agent)
                return (
                    self.exit_point
                    if current_idx == len(agent_names) - 1 and self.exit_point
                    else END
                    if current_idx == len(agent_names) - 1
                    else agent_names[current_idx + 1]
                )
            return END

        return router

    def _set_up_message_handling(self, workflow: StateGraph) -> None:
        """Set up message handling and session tracking."""
        if not self.message_handlers:
            return
        for handler in self.message_handlers:
            workflow.register_message_handler(handler)

    def _add_session_tracking(self, state: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
        """Add session tracking to state."""
        if not self.state_tracking.get("include_session_id", True):
            return state
        session_id = (
            session_id
            if session_id
            else (self.session_manager.create_session_id() if self.session_manager else str(uuid.uuid4()))
        )
        updated_state = {**state, "session_id": session_id}
        if self.state_tracking.get("include_metadata", True) and "metadata" not in updated_state:
            updated_state["metadata"] = {}
        if self.state_tracking.get("include_messages", True) and "messages" not in updated_state:
            updated_state["messages"] = []
        return updated_state

    def _update_session_from_state(self, session_id: str, state: dict[str, Any]) -> None:
        """Update session based on graph state."""
        if not self.session_manager:
            return
        self.session_manager.update_session(session_id)
        if "metadata" in state and self.state_tracking.get("include_metadata", True):
            existing_session = self.session_manager.get_session(session_id)
            if existing_session and existing_session.metadata != state["metadata"]:
                self.session_manager.update_session_metadata(
                    session_id,
                    state["metadata"],
                    merge=False,
                )

    def build(self) -> Graph:
        """Build and return a LangGraph workflow."""
        workflow = StateGraph(self._create_state_schema())
        for agent in self.agents:
            workflow.add_node(agent.name, agent)
        for name, node in self.nodes.items():
            if name not in [agent.name for agent in self.agents]:
                workflow.add_node(name, node)
        if self.human_in_loop and "human" not in self.nodes:

            def human_in_loop(state: dict[str, Any]) -> dict[str, Any]:
                """Placeholder for human-in-loop implementation."""
                return {**state, "human_input": "Approved"}

            workflow.add_node("human", human_in_loop)
        for edge in self.edges:
            source = edge["source"]
            target = edge["target"]
            workflow.add_edge(source, target)
        for cond_edge in self.conditional_edges:
            if "name" in cond_edge and "mapping" in cond_edge:
                condition_name = cond_edge["name"]
                routing_dict = cond_edge["mapping"]
                processed_routing = {
                    key: None if value == "None" or value is None else END if value == "END" else value
                    for key, value in routing_dict.items()
                }
                nodes_needing_routing = [edge["source"] for edge in self.edges if edge["target"] == condition_name]
                if not nodes_needing_routing:
                    workflow.add_conditional_edges(
                        "__start__",
                        self.nodes[condition_name],
                        processed_routing,
                    )
                else:
                    for node in nodes_needing_routing:
                        workflow.add_conditional_edges(
                            node,
                            self.nodes[condition_name],
                            processed_routing,
                        )
            else:
                source = cond_edge["source"]
                condition = cond_edge["condition"]
                targets = cond_edge["targets"]
                workflow.add_conditional_edges(
                    source,
                    condition,
                    {target: target for target in targets},
                )
        # Ensure at least one entry edge exists
        if not self.edges and not self.conditional_edges and self.agents:
            if len(self.agents) > 1:
                workflow.add_conditional_edges(
                    "__start__",
                    self._create_router(),
                )
                workflow.add_conditional_edges(
                    [agent.name for agent in self.agents],
                    self._create_router(),
                )
            elif len(self.agents) == 1:
                workflow.add_edge("__start__", self.agents[0].name)
                workflow.add_edge(self.agents[0].name, END)
        # If still no entry edge, add one from __start__ to the first agent
        if not any(edge["source"] == "__start__" for edge in self.edges) and self.agents:
            workflow.add_edge("__start__", self.agents[0].name)
        self._set_up_message_handling(workflow)
        self._configure_langgraph_session_handling(workflow)
        self.graph = workflow.compile()
        if self.memory_manager:
            self.graph.memory_manager = self.memory_manager
        self.graph.state_tracking = self.state_tracking
        return self.graph

    def _configure_langgraph_session_handling(self, workflow: StateGraph) -> None:
        """Configure LangGraph's native session handling if available."""
        try:
            # Check if LangGraph has the session configuration API
            if hasattr(workflow, "set_session_store"):
                # If we have our own session manager, wrap it as a LangGraph session store
                if self.session_manager and hasattr(self.session_manager, "_as_langgraph_store"):
                    # Use our session manager as a LangGraph session store adapter
                    langgraph_store = self.session_manager._as_langgraph_store()
                    workflow.set_session_store(langgraph_store)
                elif hasattr(workflow, "with_config"):
                    # Use LangGraph's default session store with our config
                    session_config = {
                        "session": {
                            "history": True,  # Track message history
                        },
                    }
                    workflow.with_config(configurable=session_config)
        except (AttributeError, ImportError):
            # LangGraph version might not support this feature
            # We'll fall back to our own session management
            pass

    def run(
        self,
        message: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the graph with a message in a session."""
        if not self.graph:
            self.graph = self.build()

        # Prepare config for LangGraph's native session management
        config = {}

        # Set up configurable section for LangGraph
        if "configurable" not in config:
            config["configurable"] = {}

        # Use LangGraph's native session management if available
        try:
            # Check if LangGraph supports the session config format
            if hasattr(self.graph, "add_session_history"):
                # LangGraph has native session support
                if not session_id:
                    # Let LangGraph create a new session
                    config["configurable"]["session"] = {"session_id": None}
                else:
                    # Use existing session ID with LangGraph's native system
                    config["configurable"]["session"] = {"session_id": session_id}
        except (AttributeError, ImportError):
            # LangGraph version might not support this feature
            # Fall back to our own session tracking
            pass

        # Initialize state with message
        initial_state = {"input": message}

        # Add metadata if provided
        if metadata and self.state_tracking.get("include_metadata", True):
            initial_state["metadata"] = metadata

        # Ensure required fields are present in initial_state
        if "output" not in initial_state:
            initial_state["output"] = ""
        if "agent" not in initial_state:
            initial_state["agent"] = ""
        if "memory" not in initial_state:
            initial_state["memory"] = {}

        # If LangGraph isn't handling sessions natively, use our system
        if "session" not in config.get("configurable", {}):
            # Create a new session if needed
            if not session_id and self.session_manager:
                session_id = self.create_session(metadata=metadata)
            elif not session_id:
                session_id = str(uuid.uuid4())

            # Add session tracking
            initial_state = self._add_session_tracking(initial_state, session_id)

            # Update session with new metadata if provided
            if metadata and self.session_manager:
                self.session_manager.update_session_metadata(session_id, metadata)

            # Mark session as active
            self.active_sessions.add(session_id)

        # Run the graph with or without LangGraph's native session support
        if config.get("configurable", {}).get("session"):
            # Using LangGraph's native session management
            final_state = self.graph.invoke(initial_state, config=config)

            # Extract session_id that LangGraph created if we didn't have one
            if not session_id and self.session_manager:
                # Get session ID from response if available
                if "session_id" in final_state:
                    # This is our field from GraphState
                    session_id = final_state["session_id"]
                elif hasattr(self.graph, "get_session_id"):
                    # Try to get from LangGraph's API if available
                    session_id = self.graph.get_session_id(final_state)

                # Register with our enhanced session system
                if session_id and self.session_manager:
                    self.session_manager.register_session(session_id, metadata=metadata)
                    self.active_sessions.add(session_id)
        else:
            # Using our own session management
            final_state = self.graph.invoke(initial_state)

            # Update session from final state
            if session_id and self.session_manager:
                self._update_session_from_state(session_id, final_state)

        # Track this interaction
        if session_id and self.session_manager and hasattr(self.session_manager, "track_interaction"):
            self.session_manager.track_interaction(
                session_id,
                "message",
                {
                    "input_length": len(message),
                    "has_output": "output" in final_state,
                },
            )

        return final_state

    async def arun(
        self,
        message: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the graph asynchronously with a message in a session."""
        if not self.graph:
            self.graph = self.build()

        # Prepare config for LangGraph's native session management
        config = {}

        # Set up configurable section for LangGraph
        if "configurable" not in config:
            config["configurable"] = {}

        # Use LangGraph's native session management if available
        try:
            # Check if LangGraph supports the session config format
            if hasattr(self.graph, "add_session_history"):
                # LangGraph has native session support
                if not session_id:
                    # Let LangGraph create a new session
                    config["configurable"]["session"] = {"session_id": None}
                else:
                    # Use existing session ID with LangGraph's native system
                    config["configurable"]["session"] = {"session_id": session_id}
        except (AttributeError, ImportError):
            # LangGraph version might not support this feature
            # Fall back to our own session tracking
            pass

        # Initialize state with message
        initial_state = {"input": message}

        # Add metadata if provided
        if metadata and self.state_tracking.get("include_metadata", True):
            initial_state["metadata"] = metadata

        # Ensure required fields are present in initial_state
        if "output" not in initial_state:
            initial_state["output"] = ""
        if "agent" not in initial_state:
            initial_state["agent"] = ""
        if "memory" not in initial_state:
            initial_state["memory"] = {}

        # If LangGraph isn't handling sessions natively, use our system
        if "session" not in config.get("configurable", {}):
            # Create a new session if needed
            if not session_id and self.session_manager:
                # Use create_session_id for async compatibility
                session_id = self.session_manager.create_session_id()
                # Actually create the session
                await self._async_create_session(session_id, metadata=metadata)
            elif not session_id:
                session_id = str(uuid.uuid4())

            # Add session tracking
            initial_state = self._add_session_tracking(initial_state, session_id)

            # Update session with new metadata if provided
            if metadata and self.session_manager and hasattr(self.session_manager, "update_session_metadata"):
                # Async update of session metadata
                import asyncio
                from concurrent.futures import ThreadPoolExecutor

                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    await loop.run_in_executor(
                        executor,
                        lambda: self.session_manager.update_session_metadata(session_id, metadata),
                    )

            # Mark session as active
            self.active_sessions.add(session_id)

        # Run the graph with or without LangGraph's native session support
        if config.get("configurable", {}).get("session"):
            # Using LangGraph's native session management
            final_state = await self.graph.ainvoke(initial_state, config=config)

            # Extract session_id that LangGraph created if we didn't have one
            if not session_id and self.session_manager:
                # Get session ID from response if available
                if "session_id" in final_state:
                    # This is our field from GraphState
                    session_id = final_state["session_id"]
                elif hasattr(self.graph, "get_session_id"):
                    # Try to get from LangGraph's API if available
                    session_id = self.graph.get_session_id(final_state)

                # Register with our enhanced session system
                if session_id and self.session_manager and hasattr(self.session_manager, "register_session_async"):
                    await self.session_manager.register_session_async(session_id, metadata=metadata)
                    self.active_sessions.add(session_id)
        else:
            # Using our own session management
            final_state = await self.graph.ainvoke(initial_state)

            # Update session from final state
            if session_id and self.session_manager:
                # Run this synchronously in a background thread
                import asyncio
                from concurrent.futures import ThreadPoolExecutor

                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    await loop.run_in_executor(
                        executor,
                        lambda: self._update_session_from_state(session_id, final_state),
                    )

        # Track this interaction
        if session_id and self.session_manager and hasattr(self.session_manager, "track_interaction_async"):
            await self.session_manager.track_interaction_async(
                session_id,
                "message",
                {
                    "input_length": len(message),
                    "has_output": "output" in final_state,
                },
            )

        return final_state

    async def _async_create_session(
        self,
        session_id: str,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        timeout: int | None = 3600,
    ) -> None:
        """Create a session asynchronously."""
        if not self.session_manager:
            return

        # Use ThreadPoolExecutor to run the synchronous operation in a separate thread
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                lambda: self.session_manager.create_session_with_id(
                    session_id=session_id,
                    user_id=user_id,
                    metadata=metadata,
                    timeout=timeout,
                ),
            )

    def run_multi_agent(
        self,
        message: str,
        agent_order: list[str],
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run multiple agents in sequence on a message."""
        if not self.graph:
            self.graph = self.build()

        # Validate agent order - all agents must exist
        for agent_name in agent_order:
            if agent_name not in [agent.name for agent in self.agents]:
                raise ValueError(f"Agent '{agent_name}' not found in graph")

        # Create a new session if needed
        if not session_id and self.session_manager:
            session_id = self.create_session(metadata=metadata)
        elif not session_id:
            session_id = str(uuid.uuid4())

        # Initialize state with session tracking
        state = {"input": message}

        # Add metadata if provided
        if metadata and self.state_tracking.get("include_metadata", True):
            state["metadata"] = metadata

        # Add session tracking
        state = self._add_session_tracking(state, session_id)

        # Update session with new metadata if provided
        if metadata and self.session_manager:
            self.session_manager.update_session_metadata(session_id, metadata)

        # Mark session as active
        self.active_sessions.add(session_id)

        # Run each agent in sequence
        for agent_name in agent_order:
            # Update agent field in state
            state["agent"] = agent_name

            # Get agent from nodes
            agent = self.nodes.get(agent_name)

            # Run agent
            state = agent(state)

            # Store intermediate output if messages tracking is enabled
            if self.state_tracking.get("include_messages", True):
                if "messages" not in state:
                    state["messages"] = []

                # Add agent's output to messages
                if "output" in state:
                    state["messages"].append(
                        {
                            "role": agent_name,
                            "content": state["output"],
                            "timestamp": datetime.datetime.now().isoformat(),
                        },
                    )

        # Update session from final state
        self._update_session_from_state(session_id, state)

        return state
