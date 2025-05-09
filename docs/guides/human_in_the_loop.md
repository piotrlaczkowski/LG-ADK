# 🧑‍💻 Human-in-the-Loop in LG-ADK

---

## 🤔 Why Add Human-in-the-Loop?

Sometimes, only a human can make the right call! Add review, approval, or intervention steps to your agent workflows. 👀

---

## 🧩 Where to Use Human-in-the-Loop

- ✅ **Critical Decisions**: Approve or reject important actions
- 📝 **Content Review**: Check generated text before sending
- 🧑‍⚖️ **Escalation**: Route to a human when the agent is unsure
- 🛑 **Safety**: Prevent unsafe or unwanted outputs

---

## 🚦 Quick Example

!!! tip "Enable human-in-the-loop for a node"
    ```python
    builder.enable_human_in_the_loop(source="review", target="publish")
    ```

---

## 🛠️ How It Works

- Insert human-in-the-loop nodes anywhere in your graph
- The workflow pauses and waits for human input at these nodes
- You can customize prompts and review logic

---

## 🚨 Common Pitfalls

!!! warning "Workflow stalls"
    Make sure you have a way to notify or alert humans when their input is needed, or the workflow may pause indefinitely.

---

## 🌟 Next Steps

- [Building Graphs](building_graphs.md) 🏗️
- [Session Management](session_management.md) 🗂️
- [Examples](../examples/) 💡

# Human-in-the-Loop with LG-ADK

This guide explains how to implement human-in-the-loop interactions in the LangGraph Agent Development Kit, allowing agents to request human input during their processing.

## Understanding Human-in-the-Loop

Human-in-the-loop (HITL) enables:

1. Agents to request human input or clarification
2. Humans to review and approve agent actions
3. Collaborative problem-solving between humans and agents
4. Building systems with appropriate human oversight

## Core Components

The HITL system in LG-ADK consists of:

- **HumanInputNode**: A specialized graph node that pauses for human input
- **HumanManager**: Manages the interaction between agents and humans
- **GraphBuilder HITL methods**: Helper methods to enable HITL in your graphs

## Basic Implementation

To enable human-in-the-loop in your graph:

```python
from lg_adk.builders import GraphBuilder
from lg_adk.agents import Agent
from lg_adk.models import get_model
from lg_adk.human import HumanManager

# Create an agent
agent = Agent(
    name="assistant",
    model=get_model("openai/gpt-4"),
    system_prompt="You are a helpful assistant that knows when to ask humans for help."
)

# Create a human manager
human_manager = HumanManager()

# Create a graph with human-in-the-loop
builder = GraphBuilder()
builder.add_agent(agent)
builder.enable_human_in_the_loop(human_manager)

# Build the graph
graph = builder.build()
```

## Running a Graph with Human Input

When running a graph with HITL enabled:

```python
# Start a conversation that might require human input
result = graph.run("What's the best approach for our new marketing campaign?")

# Check if human input is required
if result.state.get("requires_human_input", False):
    # Provide human input
    human_response = input("Human response: ")

    # Continue the conversation with human input
    updated_result = graph.continue_run(
        state=result.state,
        human_input=human_response
    )
```

## Agent-Initiated Human Input

You can configure your agent to recognize when it should request human input:

```python
agent = Agent(
    name="customer_service",
    model=get_model("openai/gpt-4"),
    system_prompt="""You are a customer service agent.

    If the customer request involves:
    - High-value refunds (over $100)
    - Account closures
    - Complaints requiring escalation

    Ask for human supervisor input by saying "I need to consult with a supervisor on this."
    Otherwise, handle the request yourself."""
)
```

The HumanManager will automatically detect this pattern and pause for human input.

## Configuring Human Intervention Triggers

You can customize when the system should pause for human input:

```python
from lg_adk.human import HumanManager

# Create a human manager with custom triggers
human_manager = HumanManager(
    trigger_phrases=[
        "I need human input",
        "Please ask the user",
        "Human assistance required"
    ],
    # Also trigger on specific actions or keywords
    trigger_keywords=["refund", "escalate", "override"],
    # Require approval for high-risk actions
    require_approval_for=["database_update", "payment_processing"]
)
```

## Approval Workflows

You can implement approval workflows for critical agent actions:

```python
# Create an agent that proposes actions requiring approval
agent = Agent(
    name="finance_assistant",
    model=get_model("openai/gpt-4"),
    system_prompt="""You are a finance assistant.

    For any transaction over $1000, create a PROPOSED_ACTION that
    requires human approval before proceeding."""
)

# Configure human manager for approvals
human_manager = HumanManager(
    approval_required=True,
    approval_prompt="A financial action requires your approval. Do you approve? (yes/no)"
)

# In the graph runner
result = graph.run("Please transfer $5000 to account #12345")

if result.state.get("requires_approval", False):
    # Display the proposed action to the human
    print(f"Proposed action: {result.state['proposed_action']}")

    # Get approval decision
    approval = input("Do you approve? (yes/no): ")

    # Continue with the approval decision
    updated_result = graph.continue_run(
        state=result.state,
        human_input={"approval": approval == "yes"}
    )
```

## Multiple Human Input Points

You can configure a graph to pause for human input at multiple points:

```python
from lg_adk.builders import GraphBuilder
from lg_adk.agents import Agent
from lg_adk.human import HumanManager, HumanInputNode

# Create agents
planner = Agent(
    name="planner",
    model=get_model("openai/gpt-4"),
    system_prompt="You create plans based on user requests."
)

executor = Agent(
    name="executor",
    model=get_model("openai/gpt-4"),
    system_prompt="You execute plans after they've been approved."
)

# Create human input nodes
plan_approval = HumanInputNode(
    name="plan_approval",
    prompt="Please review the proposed plan. Do you approve it or want changes?"
)

execution_confirmation = HumanInputNode(
    name="execution_confirmation",
    prompt="Execution is about to begin. Final confirmation?"
)

# Build graph with multiple human input points
builder = GraphBuilder()
builder.add_agent(planner)
builder.add_agent(executor)
builder.add_human_node(plan_approval)
builder.add_human_node(execution_confirmation)

# Connect the components
builder.add_edge(planner, plan_approval)
builder.add_edge(plan_approval, executor)
builder.add_edge(executor, execution_confirmation)

# Build the graph
graph = builder.build()
```

## Human-in-the-Loop with Streaming

For a more interactive experience, you can use streaming with HITL:

```python
# Start a streaming run that might require human input
for chunk in graph.stream("What's your recommendation for our product roadmap?"):
    if chunk.get("requires_human_input", False):
        # Get human input
        human_response = input("Human input required: ")

        # Continue the stream with human input
        for updated_chunk in graph.continue_stream(
            state=chunk,
            human_input=human_response
        ):
            print(updated_chunk.get("agent_response", ""))
    else:
        print(chunk.get("agent_response", ""), end="", flush=True)
```

## Configuring Human Input Timeouts

You can configure timeouts for human input:

```python
from lg_adk.human import HumanManager

# Configure timeouts
human_manager = HumanManager(
    input_timeout=300,  # 5 minutes in seconds
    fallback_response="No human input received within the timeout period. I'll proceed with the best course of action."
)
```

If no human input is received within the timeout, the system will use the fallback response.

## Custom Input and Output Interfaces

You can customize how human input is collected and presented:

```python
from lg_adk.human import HumanManager, InputInterface, OutputInterface

# Custom input interface (e.g., from a web form)
class WebFormInput(InputInterface):
    def get_input(self, prompt, state):
        # Code to display prompt in web UI and collect input
        # This is a placeholder implementation
        return web_form_submit_value

# Custom output interface (e.g., to a chat UI)
class ChatUIOutput(OutputInterface):
    def display_output(self, output, state):
        # Code to display output in chat UI
        # This is a placeholder implementation
        chat_ui.add_message(output)

# Use custom interfaces
human_manager = HumanManager(
    input_interface=WebFormInput(),
    output_interface=ChatUIOutput()
)
```

## Asynchronous Human Input

For web applications or services, you can use asynchronous human input:

```python
import asyncio
from lg_adk.human import AsyncHumanManager

# Create an async human manager
async_human_manager = AsyncHumanManager()

# In an async context
async def process_request(user_input, user_id):
    # Start a graph run
    result = await graph.arun(user_input, session_id=user_id)

    if result.state.get("requires_human_input", False):
        # Store the state waiting for human input
        await async_human_manager.store_pending_state(user_id, result.state)
        # Return indication that human input is needed
        return {"status": "waiting_for_human_input"}

    return {"status": "complete", "response": result.response}

# When human input arrives later
async def process_human_input(user_id, human_input):
    # Retrieve the pending state
    pending_state = await async_human_manager.get_pending_state(user_id)

    if pending_state:
        # Continue the run with the human input
        updated_result = await graph.acontinue_run(
            state=pending_state,
            human_input=human_input
        )

        # Clear the pending state
        await async_human_manager.clear_pending_state(user_id)

        return {"status": "complete", "response": updated_result.response}

    return {"status": "error", "message": "No pending state found"}
```

## Human Feedback Collection

You can use HITL to collect and incorporate human feedback:

```python
from lg_adk.human import HumanFeedbackNode

# Create a feedback node
feedback_node = HumanFeedbackNode(
    name="result_feedback",
    prompt="How satisfied are you with this response? (1-5)",
    feedback_options=["1", "2", "3", "4", "5"]
)

# Add to graph
builder = GraphBuilder()
builder.add_agent(agent)
builder.add_human_node(feedback_node)
builder.add_edge(agent, feedback_node)

# Process feedback
def handle_feedback(feedback_value, session_id):
    # Store feedback for later analysis
    feedback_store.add({
        "session_id": session_id,
        "rating": feedback_value,
        "timestamp": datetime.now()
    })

    # For low ratings, trigger review
    if int(feedback_value) <= 2:
        trigger_human_review(session_id)
```

## Complete Example: Customer Support with Human Escalation

Here's a complete example of a customer support system with HITL:

```python
from lg_adk.agents import Agent
from lg_adk.builders import GraphBuilder
from lg_adk.models import get_model
from lg_adk.human import HumanManager, HumanInputNode
from lg_adk.tools import MemoryTool
from lg_adk.memory import MemoryManager
from lg_adk.database import DatabaseManager

# Set up memory
db_manager = DatabaseManager(connection_string="sqlite:///customer_support.db")
memory_manager = MemoryManager(database_manager=db_manager)
memory_tool = MemoryTool(memory_manager=memory_manager)

# Create tier-1 support agent
tier1_agent = Agent(
    name="tier1_support",
    model=get_model("openai/gpt-4"),
    system_prompt="""You are a Tier 1 customer support agent.

    Handle basic customer inquiries about:
    - Account information
    - Basic troubleshooting
    - Product information

    If the customer issue involves:
    - Technical problems you can't solve
    - Billing disputes
    - Complaints about service

    Say "This requires escalation to a specialist." to trigger escalation.""",
    tools=[memory_tool]
)

# Create tier-2 support agent
tier2_agent = Agent(
    name="tier2_support",
    model=get_model("openai/gpt-4"),
    system_prompt="""You are a Tier 2 specialist with advanced knowledge.

    Review the conversation history and provide expert assistance.
    If you need additional customer information, ask for it.

    If the issue is beyond your scope, say "This requires human manager review.".""",
    tools=[memory_tool]
)

# Create human input nodes
human_specialist = HumanInputNode(
    name="human_specialist",
    prompt="This customer issue requires human specialist review. Please provide guidance:"
)

# Create human manager
human_manager = HumanManager(
    trigger_phrases=["This requires escalation to a specialist.",
                     "This requires human manager review."]
)

# Build the graph
builder = GraphBuilder()
builder.add_agent(tier1_agent)
builder.add_agent(tier2_agent)
builder.add_human_node(human_specialist)
builder.enable_human_in_the_loop(human_manager)

# Connect the components
builder.add_edge(tier1_agent, tier2_agent,
                 condition=lambda state: "requires escalation" in state.get("agent_response", "").lower())
builder.add_edge(tier2_agent, human_specialist,
                 condition=lambda state: "requires human manager review" in state.get("agent_response", "").lower())

# Build the graph
support_graph = builder.build()

# Example usage
session_id = "customer_123"

# Initial customer inquiry
result = support_graph.run("I'm having trouble with my billing. My last invoice shows charges for services I didn't use.",
                        session_id=session_id)

# If human input is needed at any point
if result.state.get("requires_human_input", False):
    # Get input from human specialist
    specialist_input = input("Specialist input: ")

    # Continue with the human input
    updated_result = support_graph.continue_run(
        state=result.state,
        human_input=specialist_input
    )

    print("Final response:", updated_result.response)
else:
    print("Response:", result.response)
```

## Best Practices for Human-in-the-Loop

1. **Clear Prompting**: Ensure agents have clear instructions about when to request human input
2. **Meaningful Context**: Provide humans with sufficient context to make informed decisions
3. **Appropriate Timeouts**: Set reasonable timeouts based on the urgency of the task
4. **Fallback Mechanisms**: Implement fallback behaviors when human input isn't available
5. **User Experience**: Design the human interaction to be intuitive and unobtrusive
6. **Feedback Loop**: Collect data on which interactions require human intervention to improve the system
7. **Progressive Autonomy**: Start with more human oversight and gradually reduce it as the system proves reliable

By implementing human-in-the-loop functionality with LG-ADK, you can build AI systems that combine the strengths of automated agents with human judgment and expertise. For more information on related topics, see the [Building Graphs](building_graphs.md), [Creating Agents](creating_agents.md), and [Tool Integration](tool_integration.md) guides.
