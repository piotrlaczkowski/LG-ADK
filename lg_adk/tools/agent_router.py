"""Agent Router Tool for LG-ADK. Provides tools for routing tasks between agents."""

import time
from collections.abc import Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from lg_adk import Agent
from lg_adk.utils.logging import get_logger

logger = get_logger(__name__)


class RouterType(Enum):
    """Type of routing strategy to use."""

    SEQUENTIAL = "sequential"  # Run agents in sequence
    CONCURRENT = "concurrent"  # Run agents concurrently
    SELECTOR = "selector"  # Select the best agent for the task
    MIXTURE = "mixture"  # Combine results from multiple agents


class TaskLog(BaseModel):
    """Log entry for a task processed by the router."""

    timestamp: int = Field(default_factory=lambda: int(time.time()))
    level: str
    agent_id: str | None = None
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class AgentRouter:
    """Tool for routing tasks to the most appropriate agent.

    This tool allows:
    1. Registering multiple agents with different capabilities
    2. Routing tasks to the most appropriate agent
    3. Sequential or concurrent execution of agents
    4. Mixing results from multiple agents
    """

    def __init__(
        self,
        name: str,
        agents: list[Agent] | None = None,
        router_type: RouterType = RouterType.SELECTOR,
        agent_selector: Callable[[str, list[Agent]], Agent] | None = None,
        max_loops: int = 1,
    ) -> None:
        """Initialize the agent router.

        Args:
            name: Name of the router.
            agents: List of agents to register.
            router_type: Type of routing strategy to use.
            agent_selector: Optional function to select agents for tasks.
            max_loops: Maximum number of routing loops.
        """
        self.name = name
        self.agents = agents or []
        self.router_type = router_type
        self.agent_selector = agent_selector
        self.max_loops = max_loops
        self.logs: list[TaskLog] = []

    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the router.

        Args:
            agent: Agent to add.
        """
        self.agents.append(agent)

    def _log(
        self,
        level: str,
        message: str,
        agent_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Add a log entry.

        Args:
            level: Log level.
            message: Log message.
            agent_id: ID of the agent (if applicable).
            details: Additional details.
        """
        log = TaskLog(
            level=level,
            message=message,
            agent_id=agent_id,
            details=details or {},
        )
        self.logs.append(log)

        # Also log to the logger
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "debug":
            logger.debug(message)

    def get_logs(self) -> list[TaskLog]:
        """Get all logs for the router.

        Returns:
            List of log entries.
        """
        return self.logs

    def _select_agent(self, task: str) -> Agent:
        """Select the best agent for a task.

        Args:
            task: The task description.

        Returns:
            The selected agent.

        Raises:
            ValueError: If no agents are available.
        """
        if not self.agents:
            raise ValueError("No agents available for routing")

        if self.agent_selector:
            return self.agent_selector(task, self.agents)

        # Default selection: use the first agent with a matching description or system prompt
        lower_task = task.lower()
        for agent in self.agents:
            if (
                hasattr(agent, "description")
                and agent.description
                and any(kw in lower_task for kw in agent.description.lower().split())
            ):
                return agent

            if (
                hasattr(agent, "system_prompt")
                and agent.system_prompt
                and any(kw in lower_task for kw in agent.system_prompt.lower().split())
            ):
                return agent

        # If no match found, return the first agent
        return self.agents[0]

    def route_sequential(self, task: str) -> dict[str, Any]:
        """Route a task through agents sequentially.

        Args:
            task: The task description.

        Returns:
            The final result.
        """
        self._log("info", f"Starting sequential routing for task: {task}")

        result = {"input": task}

        for i, agent in enumerate(self.agents):
            agent_name = getattr(agent, "agent_name", f"Agent_{i}")
            self._log("info", f"Routing to agent: {agent_name}", agent_id=agent_name)

            try:
                # Update the input with the previous result if available
                if "output" in result:
                    result["input"] = result["output"]

                # Run the agent
                agent_result = agent.run(result)
                result.update(agent_result)

                self._log("info", f"Agent {agent_name} completed successfully")
            except Exception as e:
                self._log(
                    "error",
                    f"Agent {agent_name} failed: {str(e)}",
                    agent_id=agent_name,
                    details={"error": str(e)},
                )
                # Continue with the next agent

        self._log("info", "Sequential routing completed")
        return result

    async def route_concurrent(self, task: str) -> dict[str, Any]:
        """Route a task to multiple agents concurrently.

        Args:
            task: The task description.

        Returns:
            Combined results from all agents.
        """
        import asyncio

        self._log("info", f"Starting concurrent routing for task: {task}")

        input_data = {"input": task}
        tasks = []

        for i, agent in enumerate(self.agents):
            agent_name = getattr(agent, "agent_name", f"Agent_{i}")
            self._log("info", f"Adding agent to concurrent pool: {agent_name}", agent_id=agent_name)

            # Create task for this agent
            tasks.append(self._run_agent_async(agent, input_data, agent_name))

        # Run all agents concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        combined_result = {"input": task, "agent_results": {}}

        for i, result in enumerate(results):
            agent_name = getattr(self.agents[i], "agent_name", f"Agent_{i}")

            if isinstance(result, Exception):
                self._log(
                    "error",
                    f"Agent {agent_name} failed: {str(result)}",
                    agent_id=agent_name,
                    details={"error": str(result)},
                )
                combined_result["agent_results"][agent_name] = {"error": str(result)}
            else:
                combined_result["agent_results"][agent_name] = result
                if "output" in result:
                    combined_result[f"output_{agent_name}"] = result["output"]

        # Set the main output as the concatenation of all outputs
        outputs = [
            r.get("output", "")
            for r in combined_result["agent_results"].values()
            if isinstance(r, dict) and "output" in r
        ]
        combined_result["output"] = "\n\n".join(outputs)

        self._log("info", "Concurrent routing completed")
        return combined_result

    async def _run_agent_async(self, agent: Agent, input_data: dict[str, Any], agent_name: str) -> dict[str, Any]:
        """Run an agent asynchronously.

        Args:
            agent: The agent to run.
            input_data: The input data.
            agent_name: The name of the agent.

        Returns:
            The agent's result.
        """
        self._log("info", f"Starting agent: {agent_name}", agent_id=agent_name)

        try:
            if hasattr(agent, "arun") and callable(agent.arun):
                result = await agent.arun(input_data)
            else:
                # Use loop.run_in_executor for synchronous agents
                import asyncio

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, agent.run, input_data)

            self._log("info", f"Agent {agent_name} completed successfully", agent_id=agent_name)
            return result
        except Exception as e:
            self._log("error", f"Agent {agent_name} failed: {str(e)}", agent_id=agent_name, details={"error": str(e)})
            raise

    def route_selector(self, task: str) -> dict[str, Any]:
        """Route a task to the most appropriate agent.

        Args:
            task: The task description.

        Returns:
            The agent's result.
        """
        self._log("info", f"Starting selector routing for task: {task}")

        # Select the best agent for the task
        agent = self._select_agent(task)
        agent_name = getattr(agent, "agent_name", "Selected_Agent")

        self._log("info", f"Selected agent: {agent_name}", agent_id=agent_name)

        try:
            # Run the agent
            input_data = {"input": task}
            result = agent.run(input_data)

            self._log("info", f"Agent {agent_name} completed successfully", agent_id=agent_name)

            # Make sure the result has an output field
            if "output" not in result:
                result["output"] = str(result)

            return result
        except Exception as e:
            self._log("error", f"Agent {agent_name} failed: {str(e)}", agent_id=agent_name, details={"error": str(e)})
            return {"error": str(e), "agent": agent_name, "input": task}

    def route_mixture(self, task: str) -> dict[str, Any]:
        """Route a task to multiple agents and combine their results.

        Args:
            task: The task description.

        Returns:
            Combined results from all agents.
        """
        self._log("info", f"Starting mixture routing for task: {task}")

        input_data = {"input": task}
        results = []

        for i, agent in enumerate(self.agents):
            agent_name = getattr(agent, "agent_name", f"Agent_{i}")
            self._log("info", f"Routing to agent: {agent_name}", agent_id=agent_name)

            try:
                # Run the agent
                agent_result = agent.run(input_data)
                results.append((agent_name, agent_result))

                self._log("info", f"Agent {agent_name} completed successfully", agent_id=agent_name)
            except Exception as e:
                self._log(
                    "error",
                    f"Agent {agent_name} failed: {str(e)}",
                    agent_id=agent_name,
                    details={"error": str(e)},
                )
                # Continue with the next agent

        # Combine results
        final_result = self._combine_agent_results(task, results)

        self._log("info", "Mixture routing completed")
        return final_result

    def _combine_agent_results(self, task: str, results: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
        """Combine results from multiple agents.

        Args:
            task: The original task.
            results: List of tuples (agent_name, result).

        Returns:
            Combined result.
        """
        combined_result = {"input": task, "agent_results": {}}

        for agent_name, result in results:
            combined_result["agent_results"][agent_name] = result
            if "output" in result:
                combined_result[f"output_{agent_name}"] = result["output"]

        # Set the main output as the concatenation of all outputs
        [r.get("output", "") for _, r in results if "output" in r]
        combined_result["output"] = "\n\n".join(
            [f"Agent {name}: {r.get('output', '')}" for name, r in results if "output" in r],
        )

        return combined_result

    def run(self, task: str) -> dict[str, Any]:
        """Route a task using the configured routing strategy.

        Args:
            task: The task description.

        Returns:
            The result based on the routing strategy.
        """
        # Clear logs for this run
        self.logs = []

        try:
            if self.router_type == RouterType.SEQUENTIAL:
                return self.route_sequential(task)
            elif self.router_type == RouterType.SELECTOR:
                return self.route_selector(task)
            elif self.router_type == RouterType.MIXTURE:
                return self.route_mixture(task)
            elif self.router_type == RouterType.CONCURRENT:
                import asyncio

                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self.route_concurrent(task))
            else:
                raise ValueError(f"Unknown router type: {self.router_type}")
        except Exception as e:
            self._log("error", f"Routing failed: {str(e)}", details={"error": str(e)})
            return {"error": str(e), "input": task}
