"""
GraphBuilder for creating LangGraph workflows with proper session management.
"""

from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, Generic, Type, Literal, cast, Set
from copy import deepcopy
import uuid
import datetime
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langgraph.graph import END, Graph, StateGraph
from pydantic import BaseModel, Field

from lg_adk.agents.base import Agent
from lg_adk.memory.memory_manager import MemoryManager
from lg_adk.sessions.session_manager import SessionManager, Session

# Type variable for state
T = TypeVar('T')

# Configure logger
logger = logging.getLogger(__name__)

class GraphState(BaseModel):
    """Base state schema for graphs with session management."""
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        """Configuration for the model."""
        arbitrary_types_allowed = True
        extra = "allow"

class GraphBuilder(BaseModel, Generic[T]):
    """
    Builder for creating LangGraph workflows with proper session and state management.
    
    Attributes:
        name: Name of the graph
        agents: List of agents in the graph
        memory_manager: Optional memory manager for storing conversation history
        session_manager: Optional session manager for managing sessions
        human_in_loop: Whether human intervention is enabled
        nodes: Dictionary of nodes in the graph
        edges: List of edges between nodes
        conditional_edges: List of conditional edges
        state_tracking: Configuration for state tracking
        message_handlers: Handlers for message processing
        entry_point: Entry point for the graph
        exit_point: Exit point for the graph
    """
    
    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"
    }
    
    name: str = Field("default_graph", description="Name of the graph")
    agents: List[Agent] = Field(default_factory=list, description="Agents in the graph")
    memory_manager: Optional[MemoryManager] = Field(
        None, description="Memory manager for storing conversation history"
    )
    session_manager: Optional[SessionManager] = Field(
        None, description="Session manager for managing sessions"
    )
    human_in_loop: bool = Field(False, description="Whether human intervention is enabled")
    nodes: Dict[str, Any] = Field(default_factory=dict, description="Nodes in the graph")
    edges: List[Dict[str, str]] = Field(default_factory=list, description="Edges between nodes")
    conditional_edges: List[Dict[str, Any]] = Field(default_factory=list, description="Conditional edges")
    state_tracking: Dict[str, bool] = Field(
        default_factory=lambda: {
            "include_session_id": True,
            "include_metadata": True,
            "include_messages": True
        }, 
        description="Configuration for state tracking"
    )
    message_handlers: List[Callable] = Field(default_factory=list, description="Handlers for message processing")
    entry_point: Optional[str] = Field(None, description="Entry point for the graph")
    exit_point: Optional[str] = Field(None, description="Exit point for the graph")
    active_sessions: Set[str] = Field(default_factory=set, description="Set of active session IDs")
    graph: Optional[Graph] = Field(None, description="The built LangGraph graph")
    
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
        """
        Enable human-in-the-loop for the graph.
        
        Args:
            human_manager: Optional human manager to use for human-in-the-loop functionality
        """
        self.human_in_loop = True
        # Store the human manager if provided
        if human_manager:
            self.human_manager = human_manager
    
    def enable_human_feedback(self, feedback_handler=None) -> None:
        """
        Enable human feedback for the graph.
        
        Args:
            feedback_handler: Optional handler for processing human feedback
        """
        self.human_feedback = True
        # Store the feedback handler if provided
        if feedback_handler:
            self.feedback_handler = feedback_handler
    
    def configure_state_tracking(
        self, 
        include_session_id: bool = True, 
        include_metadata: bool = True,
        include_messages: bool = True
    ) -> None:
        """
        Configure state tracking options for the graph.
        
        Args:
            include_session_id: Whether to include session ID in state
            include_metadata: Whether to include metadata in state
            include_messages: Whether to include message history in state
        """
        self.state_tracking = {
            "include_session_id": include_session_id,
            "include_metadata": include_metadata,
            "include_messages": include_messages
        }
    
    def add_node(self, name: str, node_func: Any) -> None:
        """
        Add a node to the graph.
        
        Args:
            name: Name of the node
            node_func: Function or Agent for the node
        """
        self.nodes[name] = node_func
    
    def add_edge(self, source: str, target: str) -> None:
        """
        Add an edge between nodes.
        
        Args:
            source: Name of the source node (or None for entry point)
            target: Name of the target node (or END for exit point)
        """
        source_name = source if source is not None else None
        target_name = target if target != "END" and target is not None else END
        
        self.edges.append({"source": source_name, "target": target_name})
    
    def add_conditional_edge(self, source: str, condition_function: Callable, targets: List[str]) -> None:
        """
        Add a conditional edge.
        
        Args:
            source: Name of the source node
            condition_function: Function that determines the target node
            targets: List of possible target nodes
        """
        self.conditional_edges.append({
            "source": source,
            "condition": condition_function,
            "targets": targets
        })
    
    def add_conditional_edges(self, condition_name: str, mapping: Dict[str, str]) -> None:
        """
        Add conditional edges based on a named condition function.
        
        Args:
            condition_name: Name of the condition function in the nodes dictionary
            mapping: Mapping from condition result to target node
        """
        condition_func = self.nodes.get(condition_name)
        if not condition_func:
            # Create a placeholder for the condition function
            # This allows adding the conditional edges before defining the function
            self.nodes[condition_name] = lambda state: state.get("_condition_result", "default")
            condition_func = self.nodes[condition_name]
        
        self.conditional_edges.append({
            "name": condition_name,
            "function": condition_func,
            "mapping": mapping
        })
    
    def on_message(self, handler: Callable) -> None:
        """
        Register a message handler function.
        
        Args:
            handler: Function to be called when a message is received
        """
        self.message_handlers.append(handler)
    
    def set_entry_point(self, node_name: str) -> None:
        """
        Set the entry point for the graph.
        
        Args:
            node_name: Name of the entry point node
        """
        self.entry_point = node_name
    
    def set_exit_point(self, node_name: str) -> None:
        """
        Set the exit point for the graph.
        
        Args:
            node_name: Name of the exit point node
        """
        self.exit_point = node_name
    
    def add_human_node(self, human_node: Any, name: Optional[str] = None) -> None:
        """
        Add a human-in-the-loop node.
        
        Args:
            human_node: Human node object or function
            name: Optional name for the node (defaults to human_node.name or 'human')
        """
        node_name = name or getattr(human_node, 'name', 'human')
        self.nodes[node_name] = human_node
    
    def create_session(self, user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, 
                      timeout: Optional[int] = 3600) -> str:
        """
        Create a new session for interacting with the graph.
        
        Args:
            user_id: Optional user ID associated with the session
            metadata: Optional metadata to store with the session
            timeout: Session timeout in seconds (default: 1 hour)
            
        Returns:
            Session ID for the new session
        """
        if not self.session_manager:
            raise ValueError("Session manager must be configured before creating sessions")
        
        session_id = self.session_manager.create_session(user_id=user_id, metadata=metadata, timeout=timeout)
        self.active_sessions.add(session_id)
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID.
        
        Args:
            session_id: Session ID to retrieve
            
        Returns:
            Session object or None if not found
        """
        if not self.session_manager:
            raise ValueError("Session manager not configured")
        
        return self.session_manager.get_session(session_id)
    
    def update_session_metadata(self, session_id: str, metadata: Dict[str, Any], merge: bool = True) -> None:
        """
        Update session metadata.
        
        Args:
            session_id: Session ID to update
            metadata: New metadata to store
            merge: Whether to merge with existing metadata (default: True)
        """
        if not self.session_manager:
            raise ValueError("Session manager not configured")
        
        self.session_manager.update_session_metadata(session_id, metadata, merge=merge)
    
    def end_session(self, session_id: str) -> None:
        """
        End a session and clean up resources.
        
        Args:
            session_id: Session ID to end
        """
        if not self.session_manager:
            raise ValueError("Session manager not configured")
        
        self.session_manager.remove_session(session_id)
        self.active_sessions.discard(session_id)
        
        # If memory manager exists, clean up session memories
        if self.memory_manager:
            try:
                self.memory_manager.clear_session_memories(session_id)
            except Exception as e:
                logger.warning(f"Error clearing memories for session {session_id}: {e}")
    
    def clear_expired_sessions(self) -> List[str]:
        """
        Clear expired sessions and return their IDs.
        
        Returns:
            List of expired session IDs that were cleared
        """
        if not self.session_manager:
            raise ValueError("Session manager not configured")
        
        expired_sessions = self.session_manager.clear_expired_sessions()
        
        # Remove expired sessions from active sessions
        for session_id in expired_sessions:
            self.active_sessions.discard(session_id)
            
            # Clean up memory if available
            if self.memory_manager:
                try:
                    self.memory_manager.clear_session_memories(session_id)
                except Exception as e:
                    logger.warning(f"Error clearing memories for expired session {session_id}: {e}")
        
        return expired_sessions
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get the message history for a session.
        
        Args:
            session_id: Session ID to get history for
            
        Returns:
            List of messages in the session
        """
        if not self.memory_manager:
            raise ValueError("Memory manager not configured, cannot retrieve session history")
        
        # Retrieve messages from memory
        return self.memory_manager.get_session_messages(session_id)
    
    def _create_state_schema(self) -> Dict[str, Any]:
        """Create the state schema for the graph with proper session tracking."""
        schema = {
            "input": str,
            "output": str,
            "agent": str,
            "memory": dict,
            "human_input": Optional[str],
        }
        
        # Add session tracking fields based on configuration
        if self.state_tracking.get("include_session_id", True):
            schema["session_id"] = str
        
        if self.state_tracking.get("include_metadata", True):
            schema["metadata"] = dict
        
        if self.state_tracking.get("include_messages", True):
            schema["messages"] = list
        
        return schema
    
    def _create_router(self) -> Callable:
        """Create a router function for the graph."""
        agent_names = [agent.name for agent in self.agents]
        
        def router(state: Dict[str, Any]) -> Union[str, List[str]]:
            """Route to the next agent or end the graph."""
            current_agent = state.get("agent")
            
            # If there's no current agent, route to the first one or the entry point
            if not current_agent:
                if self.entry_point:
                    return self.entry_point
                return agent_names[0] if agent_names else END
            
            # Use the exit point if specified
            if current_agent == self.exit_point:
                return END
            
            # In a more complex implementation, this would have logic to determine
            # which agent to route to next based on the state
            # For simple linear flow:
            if current_agent in agent_names:
                current_idx = agent_names.index(current_agent)
                
                # If we're at the last agent, go to exit point or end
                if current_idx == len(agent_names) - 1:
                    return self.exit_point if self.exit_point else END
                
                # Otherwise, route to the next agent
                return agent_names[current_idx + 1]
            
            return END
        
        return router
    
    def _set_up_message_handling(self, workflow: StateGraph) -> None:
        """
        Set up message handling and session tracking.
        
        Args:
            workflow: The StateGraph to configure
        """
        if not self.message_handlers:
            return
            
        # Register message handlers
        for handler in self.message_handlers:
            workflow.register_message_handler(handler)
    
    def _add_session_tracking(self, state: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Add session tracking to state.
        
        Args:
            state: Current state dictionary
            session_id: Optional session ID to use
            
        Returns:
            Updated state with session tracking
        """
        if not self.state_tracking.get("include_session_id", True):
            return state
            
        # If no session ID is provided, generate one
        if not session_id:
            if self.session_manager:
                session_id = self.session_manager.create_session_id()
            else:
                session_id = str(uuid.uuid4())
        
        updated_state = {**state, "session_id": session_id}
        
        # Add empty metadata if tracking is enabled
        if self.state_tracking.get("include_metadata", True) and "metadata" not in updated_state:
            updated_state["metadata"] = {}
            
        # Add empty messages list if tracking is enabled
        if self.state_tracking.get("include_messages", True) and "messages" not in updated_state:
            updated_state["messages"] = []
            
        return updated_state
    
    def _update_session_from_state(self, session_id: str, state: Dict[str, Any]) -> None:
        """
        Update session based on graph state.
        
        Args:
            session_id: Session ID to update
            state: Current state dictionary
        """
        if not self.session_manager:
            return
            
        # Update session last active timestamp
        self.session_manager.update_session(session_id)
        
        # Update session metadata if changed
        if "metadata" in state and self.state_tracking.get("include_metadata", True):
            existing_session = self.session_manager.get_session(session_id)
            if existing_session and existing_session.metadata != state["metadata"]:
                self.session_manager.update_session_metadata(
                    session_id, 
                    state["metadata"],
                    merge=False  # Replace with the current state
                )

    def build(self, state_type: Optional[Type[T]] = None) -> Graph:
        """
        Build and return a LangGraph workflow.
        
        Args:
            state_type: Optional type for the state, for better type checking
            
        Returns:
            A configured LangGraph graph.
        """
        # Create the state graph with the appropriate state schema
        workflow = StateGraph(self._create_state_schema())
        
        # Add nodes for each agent
        for agent in self.agents:
            workflow.add_node(agent.name, agent)
        
        # Add all other nodes
        for name, node in self.nodes.items():
            if name not in [agent.name for agent in self.agents]:
                workflow.add_node(name, node)
        
        # If human-in-loop is enabled but no node exists, add a default one
        if self.human_in_loop and "human" not in self.nodes:
            def human_in_loop(state: Dict[str, Any]) -> Dict[str, Any]:
                # This is a placeholder for actual human-in-loop implementation
                # In a real implementation, this would wait for human input
                return {**state, "human_input": "Approved"}
            
            workflow.add_node("human", human_in_loop)
        
        # Add explicit edges
        for edge in self.edges:
            source = edge["source"]
            target = edge["target"]
            
            workflow.add_edge(source, target)
        
        # Add conditional edges
        for cond_edge in self.conditional_edges:
            if "name" in cond_edge and "mapping" in cond_edge:
                # Map-based conditional routing
                condition_name = cond_edge["name"]
                routing_dict = cond_edge["mapping"]
                
                # Convert string "None" to None and string "END" to END constant
                processed_routing = {}
                for key, value in routing_dict.items():
                    if value == "None" or value is None:
                        processed_routing[key] = None
                    elif value == "END":
                        processed_routing[key] = END
                    else:
                        processed_routing[key] = value
                
                # Add the conditional edges to all nodes that need this routing
                nodes_needing_routing = []
                
                # Find nodes that have edges to the condition
                for edge in self.edges:
                    if edge["target"] == condition_name:
                        nodes_needing_routing.append(edge["source"])
                
                # If no incoming edges to condition, assume entry point
                if not nodes_needing_routing:
                    workflow.add_conditional_edges(
                        None,
                        self.nodes[condition_name],
                        processed_routing
                    )
                else:
                    # Add conditional edges for each source node
                    for node in nodes_needing_routing:
                        workflow.add_conditional_edges(
                            node,
                            self.nodes[condition_name],
                            processed_routing
                        )
            else:
                # Traditional conditional edge
                source = cond_edge["source"]
                condition = cond_edge["condition"]
                targets = cond_edge["targets"]
                
                workflow.add_conditional_edges(
                    source,
                    condition,
                    {target: target for target in targets}
                )
        
        # Set up default routing if no explicit routing is defined
        if not self.edges and not self.conditional_edges:
            if len(self.agents) > 1:
                workflow.add_conditional_edges(
                    None,  # This means this router routes from the starting node
                    self._create_router(),
                )
                
                workflow.add_conditional_edges(
                    [agent.name for agent in self.agents],
                    self._create_router(),
                )
            elif len(self.agents) == 1:
                # Simple flow for a single agent
                workflow.add_edge(None, self.agents[0].name)
                workflow.add_edge(self.agents[0].name, END)
        
        # Set up message handling
        self._set_up_message_handling(workflow)
        
        # Configure session handling in LangGraph if possible
        self._configure_langgraph_session_handling(workflow)
        
        # Return the compiled graph
        self.graph = workflow.compile()
        
        # Add our memory manager for the graph to use for additional features
        if self.memory_manager:
            self.graph.memory_manager = self.memory_manager
        
        # Store state tracking config on graph
        self.graph.state_tracking = self.state_tracking
        
        return self.graph
    
    def _configure_langgraph_session_handling(self, workflow: StateGraph) -> None:
        """
        Configure LangGraph's native session handling if available.
        
        This method tries to use LangGraph's native session features when they're available,
        falling back to our own implementation when they're not.
        
        Args:
            workflow: The LangGraph workflow to configure
        """
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
                        }
                    }
                    workflow.with_config(configurable=session_config)
        except (AttributeError, ImportError):
            # LangGraph version might not support this feature
            # We'll fall back to our own session management
            pass
    
    def run(
        self, 
        message: str, 
        session_id: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the graph with a message in a session.
        
        Args:
            message: Input message to process
            session_id: Optional session ID to use
            metadata: Optional metadata to include in state
            
        Returns:
            Final state after processing
        """
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
                    "has_output": "output" in final_state
                }
            )
        
        return final_state
    
    async def arun(
        self,
        message: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the graph asynchronously with a message in a session.
        
        Args:
            message: Input message to process
            session_id: Optional session ID to use
            metadata: Optional metadata to include in state
            
        Returns:
            Final state after processing
        """
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
                        lambda: self.session_manager.update_session_metadata(session_id, metadata)
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
                        lambda: self._update_session_from_state(session_id, final_state)
                    )
        
        # Track this interaction
        if session_id and self.session_manager and hasattr(self.session_manager, "track_interaction_async"):
            await self.session_manager.track_interaction_async(
                session_id, 
                "message", 
                {
                    "input_length": len(message),
                    "has_output": "output" in final_state
                }
            )
        
        return final_state
    
    async def _async_create_session(self, session_id: str, user_id: Optional[str] = None, 
                                   metadata: Optional[Dict[str, Any]] = None, 
                                   timeout: Optional[int] = 3600) -> None:
        """
        Create a session asynchronously.
        
        Args:
            session_id: Session ID to create
            user_id: Optional user ID to associate with the session
            metadata: Optional metadata to store with the session
            timeout: Session timeout in seconds
        """
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
                    timeout=timeout
                )
            )
    
    def run_multi_agent(
        self,
        message: str,
        agent_order: List[str],
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run multiple agents in sequence on a message.
        
        Args:
            message: Input message to process
            agent_order: List of agent names to execute in order
            session_id: Optional session ID to use
            metadata: Optional metadata to include in state
            
        Returns:
            Final state after processing
        """
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
                    state["messages"].append({
                        "role": agent_name,
                        "content": state["output"],
                        "timestamp": datetime.datetime.now().isoformat(),
                    })
        
        # Update session from final state
        self._update_session_from_state(session_id, state)
        
        return state
