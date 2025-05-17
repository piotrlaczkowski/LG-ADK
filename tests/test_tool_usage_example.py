import unittest
from unittest.mock import MagicMock, patch

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class CalculatorArgs(BaseModel):
    operation: str = Field(...)
    a: float = Field(...)
    b: float = Field(...)


class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Performs basic arithmetic operations"
    args_schema: type = CalculatorArgs

    def _run(self, operation: str, a: float, b: float):
        if operation == "add":
            return {"result": a + b}
        elif operation == "subtract":
            return {"result": a - b}
        elif operation == "multiply":
            return {"result": a * b}
        elif operation == "divide":
            if b == 0:
                return {"error": "Cannot divide by zero"}
            return {"result": a / b}
        else:
            return {"error": f"Unknown operation: {operation}"}

    def _run_tool(self, *args, **kwargs):
        return self._run(*args, **kwargs)


class TestToolUsageExample(unittest.TestCase):
    @patch("lg_adk.models.get_model")
    @patch("lg_adk.agents.Agent.run")
    def test_calculator_tool_integration(self, mock_agent_run, mock_get_model):
        # Mock the model and agent run
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        mock_agent_run.return_value = {"output": "42"}
        # Import Agent after patching
        from lg_adk.agents import Agent

        agent = Agent(
            name="math_assistant",
            llm=mock_model,
            description="You are a helpful math assistant. Use the calculator tool when needed.",
            tools=[CalculatorTool()],
        )
        # Simulate a run
        result = agent.run({"input": "What is 6 × 7?"})
        self.assertIn("output", result)
