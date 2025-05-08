"""Multi-agent system implementation for LG-ADK."""

from typing import Any

from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from lg_adk.agents.base import Agent


class MultiAgentSystem(BaseModel):
    """Multi-agent system for orchestrating multiple agents.

    This class allows defining a hierarchical structure of agents,
    similar to Google's ADK multi-agent system.

    Attributes:
        name: The name of the multi-agent system.
        coordinator: The coordinator agent.
        agents: A list of sub-agents managed by the coordinator.
        description: A description of the multi-agent system's purpose.
    """

    name: str = Field(..., description="Name of the multi-agent system")
    coordinator: Agent = Field(..., description="Coordinator agent")
    agents: list[Agent] = Field(default_factory=list, description="List of sub-agents")
    description: str = Field(
        "A multi-agent system",
        description="Description of the multi-agent system",
    )
    _graph: StateGraph | None = None

    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the multi-agent system."""
        self.agents.append(agent)

    def add_agents(self, agents: list[Agent]) -> None:
        """Add multiple agents to the multi-agent system."""
        self.agents.extend(agents)

    def _build_graph(self) -> StateGraph:
        """Build the multi-agent system graph.

        Returns:
            A StateGraph representing the multi-agent workflow.
        """
        from lg_adk.builders.graph_builder import GraphBuilder

        builder = GraphBuilder()

        # Add coordinator agent
        builder.add_agent(self.coordinator)

        # Add all sub-agents
        for agent in self.agents:
            builder.add_agent(agent)

        # Build the graph with coordinator as the entry point
        graph = builder.build(entry_point=self.coordinator.name)
        return graph

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process the current state using the multi-agent system.

        Args:
            state: The current state of the conversation.

        Returns:
            The updated state after processing.
        """
        if self._graph is None:
            self._graph = self._build_graph()

        # Modify the state to include coordinator instructions
        if "system_message" not in state:
            system_message = f"""
            You are the coordinator of a multi-agent system named '{self.name}'.
            {self.description}

            You have access to the following agents:
            """
            for agent in self.agents:
                system_message += f"\n- {agent.name}: {agent.description}"

            system_message += """

            Your job is to:
            1. Analyze the user query
            2. Decide which agent(s) should handle it
            3. Coordinate the workflow between agents
            4. Provide a final response to the user
            """

            state["system_message"] = system_message

        # Run the graph
        result = self._graph.invoke(state)
        return result

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Make the multi-agent system callable directly."""
        return self.run(state)


class Conversation(BaseModel):
    """Helper class for tracking a conversation with a multi-agent system.

    Attributes:
        multi_agent_system: The multi-agent system to use.
        conversation_history: The history of the conversation.
    """

    multi_agent_system: MultiAgentSystem
    conversation_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Conversation history",
    )

    def send_message(self, message: str) -> str:
        """Send a message to the multi-agent system and get the response.

        Args:
            message: The message to send.

        Returns:
            The response from the multi-agent system.
        """
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": message})

        # Prepare state with conversation history
        state = {
            "input": message,
            "conversation_history": self.conversation_history,
        }

        # Run the multi-agent system
        result = self.multi_agent_system.run(state)

        # Extract the response
        response = result.get("output", "")

        # Add system response to history
        self.conversation_history.append({"role": "system", "content": response})

        return response
