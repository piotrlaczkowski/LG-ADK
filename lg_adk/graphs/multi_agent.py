"""Multi-agent collaboration graph for LangGraph CLI.

This module implements a multi-agent collaboration system
with proper session handling for the LangGraph CLI.
"""

from typing import Any, Literal, TypedDict

from langgraph.graph import Graph
from pydantic import BaseModel, Field

from lg_adk.agents import Agent
from lg_adk.builders import GraphBuilder
from lg_adk.database import DatabaseManager
from lg_adk.memory import MemoryManager
from lg_adk.models import get_model
from lg_adk.tools import WebSearchTool


# Define message type
class Message(BaseModel):
    """Represents a message in the conversation."""

    role: str = Field(
        ...,
        description="Role of the message sender (system, user, assistant, researcher, writer, critic)",
    )
    content: str = Field(..., description="Content of the message")
    agent: str | None = Field(None, description="Agent that created this message")


# Define task type
class Task(BaseModel):
    """Represents a task in the workflow."""

    description: str = Field(..., description="Description of the task")
    status: str = Field(default="pending", description="Status of the task (pending, in_progress, completed)")
    assigned_to: str | None = Field(None, description="Agent assigned to this task")
    result: str | None = Field(None, description="Result of the task once completed")


# Define state type
class MultiAgentState(TypedDict):
    """Represents the state of a multi-agent conversation."""

    messages: list[Message]
    session_id: str
    current_agent: str
    tasks: list[Task]
    final_output: str | None
    metadata: dict[str, Any]


def create_multi_agent_graph() -> Graph:
    """Create a multi-agent collaboration graph with proper session handling.

    This function builds a graph with multiple agents that collaborate:
    - Researcher: Gathers information
    - Writer: Crafts content based on research
    - Critic: Reviews and improves the content

    Returns:
        Graph: A LangGraph graph instance for multi-agent collaboration.
    """
    # Create agents
    researcher = Agent(
        name="researcher",
        model=get_model("openai/gpt-4"),
        system_prompt=(
            "You are a skilled researcher who thoroughly investigates topics. "
            "Your job is to find relevant information and key facts about the given topic. "
            "Be thorough but focus on the most important and accurate information. "
            "Present your findings in a structured way."
        ),
        tools=[WebSearchTool()],
    )

    writer = Agent(
        name="writer",
        model=get_model("openai/gpt-4"),
        system_prompt=(
            "You are an expert writer who transforms research into engaging, clear content. "
            "Your job is to take the researcher's findings and create well-structured, "
            "informative content tailored to the user's request. "
            "Focus on clarity, accuracy, and engagement."
        ),
    )

    critic = Agent(
        name="critic",
        model=get_model("openai/gpt-4"),
        system_prompt=(
            "You are a discerning critic who reviews content for quality. "
            "Your job is to analyze the writer's draft and suggest specific improvements "
            "for clarity, accuracy, structure, and engagement. "
            "Be constructive and specific in your feedback."
        ),
    )

    # Create memory manager
    db_manager = DatabaseManager(connection_string="sqlite:///multi_agent_memory.db")
    memory_manager = MemoryManager(database_manager=db_manager)

    # Create graph builder
    builder = GraphBuilder(name="multi_agent")

    # Add agents to the builder
    builder.add_agent(researcher)
    builder.add_agent(writer)
    builder.add_agent(critic)

    # Add memory
    builder.add_memory(memory_manager)

    # Configure state tracking
    builder.configure_state_tracking(
        include_session_id=True,
        include_metadata=True,
    )

    # Process user request
    def process_request(state: MultiAgentState, request: str) -> MultiAgentState:
        """Process the initial user request and initialize the workflow."""
        messages = state.get("messages", [])
        user_message = Message(role="user", content=request)

        # Create a research task
        tasks = state.get("tasks", [])
        research_task = Task(
            description=f"Research the topic: {request}",
            status="pending",
            assigned_to="researcher",
        )

        return {
            **state,
            "messages": messages + [user_message],
            "tasks": tasks + [research_task],
            "current_agent": "researcher",
        }

    # Research phase
    def research_topic(state: MultiAgentState) -> MultiAgentState:
        """Perform research on the topic."""
        messages = state.get("messages", [])
        tasks = state.get("tasks", [])
        session_id = state.get("session_id")

        # Find the research task
        research_task = next((task for task in tasks if task["assigned_to"] == "researcher"), None)
        if not research_task:
            return state

        # Configure for proper checkpointing
        config = {
            "configurable": {
                "thread_id": session_id,
            },
        }

        # Prepare messages for the model
        model_messages = []
        # Add system message
        model_messages.append(
            {
                "role": "system",
                "content": researcher.system_prompt,
            },
        )

        # Add relevant messages
        for msg in messages:
            if msg.role in ["user", "assistant"]:
                model_messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                    },
                )

        # Generate research findings
        research_findings = researcher.model.generate(
            messages=model_messages,
            config=config,
        )

        # Update the research task
        updated_tasks = []
        for task in tasks:
            if task["assigned_to"] == "researcher":
                updated_task = {**task, "status": "completed", "result": research_findings}
                # Create next task for the writer
                writing_task = Task(
                    description="Write content based on the research findings",
                    status="pending",
                    assigned_to="writer",
                )
                updated_tasks.append(updated_task)
                updated_tasks.append(writing_task)
            else:
                updated_tasks.append(task)

        # Add researcher's message
        researcher_message = Message(
            role="assistant",
            content=research_findings,
            agent="researcher",
        )

        return {
            **state,
            "messages": messages + [researcher_message],
            "tasks": updated_tasks,
            "current_agent": "writer",
        }

    # Writing phase
    def write_content(state: MultiAgentState) -> MultiAgentState:
        """Write content based on research findings."""
        messages = state.get("messages", [])
        tasks = state.get("tasks", [])
        session_id = state.get("session_id")

        # Find the writing task and research results
        writing_task = next((task for task in tasks if task["assigned_to"] == "writer"), None)
        research_task = next(
            (task for task in tasks if task["assigned_to"] == "researcher" and task["status"] == "completed"),
            None,
        )

        if not writing_task or not research_task:
            return state

        # Configure for proper checkpointing
        config = {
            "configurable": {
                "thread_id": session_id,
            },
        }

        # Prepare messages for the model
        model_messages = []
        # Add system message
        model_messages.append(
            {
                "role": "system",
                "content": writer.system_prompt,
            },
        )

        # Add relevant messages including research findings
        for msg in messages:
            if msg.role == "user":
                model_messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                    },
                )
            elif msg.agent == "researcher":
                model_messages.append(
                    {
                        "role": "assistant",
                        "content": f"Research findings:\n\n{msg.content}",
                    },
                )

        # Generate written content
        written_content = writer.model.generate(
            messages=model_messages,
            config=config,
        )

        # Update the writing task
        updated_tasks = []
        for task in tasks:
            if task["assigned_to"] == "writer":
                updated_task = {**task, "status": "completed", "result": written_content}
                # Create next task for the critic
                critique_task = Task(
                    description="Review and improve the written content",
                    status="pending",
                    assigned_to="critic",
                )
                updated_tasks.append(updated_task)
                updated_tasks.append(critique_task)
            else:
                updated_tasks.append(task)

        # Add writer's message
        writer_message = Message(
            role="assistant",
            content=written_content,
            agent="writer",
        )

        return {
            **state,
            "messages": messages + [writer_message],
            "tasks": updated_tasks,
            "current_agent": "critic",
        }

    # Critique phase
    def critique_content(state: MultiAgentState) -> MultiAgentState:
        """Review and improve the content."""
        messages = state.get("messages", [])
        tasks = state.get("tasks", [])
        session_id = state.get("session_id")

        # Find the critique task and written content
        critique_task = next((task for task in tasks if task["assigned_to"] == "critic"), None)
        writing_task = next(
            (task for task in tasks if task["assigned_to"] == "writer" and task["status"] == "completed"),
            None,
        )

        if not critique_task or not writing_task:
            return state

        # Configure for proper checkpointing
        config = {
            "configurable": {
                "thread_id": session_id,
            },
        }

        # Prepare messages for the model
        model_messages = []
        # Add system message
        model_messages.append(
            {
                "role": "system",
                "content": critic.system_prompt,
            },
        )

        # Add relevant messages including written content
        for msg in messages:
            if msg.role == "user":
                model_messages.append(
                    {
                        "role": "user",
                        "content": f"Original request: {msg.content}",
                    },
                )
            elif msg.agent == "writer":
                model_messages.append(
                    {
                        "role": "user",
                        "content": f"Content to review:\n\n{msg.content}",
                    },
                )

        # Generate critique and improvements
        critique = critic.model.generate(
            messages=model_messages,
            config=config,
        )

        # Update the critique task
        updated_tasks = []
        for task in tasks:
            if task["assigned_to"] == "critic":
                updated_task = {**task, "status": "completed", "result": critique}
                updated_tasks.append(updated_task)
            else:
                updated_tasks.append(task)

        # Add critic's message
        critic_message = Message(
            role="assistant",
            content=critique,
            agent="critic",
        )

        # Prepare final output combining all contributions
        writer_content = next((msg.content for msg in messages if msg.agent == "writer"), "")
        final_output = f"""
# Final Result

## Original Content
{writer_content}

## Improvements
{critique}
"""

        return {
            **state,
            "messages": messages + [critic_message],
            "tasks": updated_tasks,
            "final_output": final_output,
            "current_agent": "complete",
        }

    # Routing logic
    def route_next_agent(state: MultiAgentState) -> Literal["researcher", "writer", "critic", "complete"]:
        """Determine which agent should act next based on the current state."""
        current_agent = state.get("current_agent", "")
        tasks = state.get("tasks", [])

        # Check if all tasks are completed
        all_completed = all(task["status"] == "completed" for task in tasks)

        if all_completed or current_agent == "complete":
            return "complete"

        # Otherwise, return the current agent
        return current_agent

    # Wire up the graph
    builder.add_node("process_request", process_request)
    builder.add_node("research_topic", research_topic)
    builder.add_node("write_content", write_content)
    builder.add_node("critique_content", critique_content)

    # Add conditional routing
    builder.add_conditional_edges(
        "route_next_agent",
        {
            "researcher": "research_topic",
            "writer": "write_content",
            "critic": "critique_content",
            "complete": None,  # End the graph
        },
    )

    # Connect the nodes
    builder.add_edge("process_request", "route_next_agent")
    builder.add_edge("research_topic", "route_next_agent")
    builder.add_edge("write_content", "route_next_agent")
    builder.add_edge("critique_content", "route_next_agent")

    # Set entry and exit points
    builder.set_entry_point("process_request")

    # Build the graph with proper typing
    typed_graph: Graph[MultiAgentState] = builder.build()
    return typed_graph


# Export the graph for the LangGraph CLI to discover
graph = create_multi_agent_graph()

# Example of how to use the graph directly
if __name__ == "__main__":
    import uuid

    # Create a session ID
    session_id = str(uuid.uuid4())

    # Configure for using with checkpointer
    config = {
        "configurable": {
            "thread_id": session_id,
        },
    }

    # Initialize state
    initial_state = {
        "messages": [],
        "session_id": session_id,
        "current_agent": "",
        "tasks": [],
        "final_output": None,
        "metadata": {"started_at": str(uuid.uuid4())},
    }

    # Get user request
    user_request = input("What would you like the agents to work on? ")

    # Process request through the graph
    result = graph.invoke(
        {"request": user_request},
        config=config,
        state=initial_state,
    )

    # Display agent contributions
    research_message = next(
        (msg.content for msg in result["messages"] if msg.get("agent") == "researcher"),
        "No research conducted",
    )

    written_content = next(
        (msg.content for msg in result["messages"] if msg.get("agent") == "writer"),
        "No content written",
    )

    critique = next((msg.content for msg in result["messages"] if msg.get("agent") == "critic"), "No critique provided")

    final_output = result.get("final_output", "No final output generated")
