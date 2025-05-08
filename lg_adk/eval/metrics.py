"""Evaluation metrics for LG-ADK."""

import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from lg_adk.eval.dataset import EvalExample


class Metric(ABC):
    """Base class for evaluation metrics."""

    @abstractmethod
    def evaluate(
        self,
        example: EvalExample,
        actual_output: Any,
        **kwargs: Any,
    ) -> float:
        """Evaluate a single example.

        Args:
            example: The evaluation example.
            actual_output: The actual output from the agent.
            **kwargs: Additional arguments.

        Returns:
            A score for the example.
        """
        pass

    @abstractmethod
    def aggregate(self, scores: list[float]) -> float:
        """Aggregate scores across examples.

        Args:
            scores: List of scores from individual examples.

        Returns:
            An aggregated score.
        """
        pass


class AccuracyMetric(Metric):
    """Accuracy metric for evaluating agent responses.

    This metric compares the actual output with the expected output
    and calculates the accuracy score.
    """

    def evaluate(
        self,
        example: EvalExample,
        actual_output: Any,
        **kwargs: Any,
    ) -> float:
        """Evaluate a single example for accuracy.

        Args:
            example: The evaluation example.
            actual_output: The actual output from the agent.
            **kwargs: Additional arguments including:
                - model: Optional LLM to use for evaluating responses

        Returns:
            A score between 0 and 1.
        """
        expected = example.expected_output

        # If expected output is None, we can't evaluate
        if expected is None:
            return 0.0

        # If a model was provided, use it to evaluate
        model = kwargs.get("model")
        if model:
            prompt = f"""
            Compare the following expected output with the actual output and
            rate how well the actual output matches the expected output on a scale from 0 to 1.

            Expected output: {expected}
            Actual output: {actual_output}

            Score (0-1):
            """
            try:
                result = model.invoke(prompt).strip()
                # Extract just the score
                score = float(result.split("\n")[-1].strip())
                return min(max(score, 0.0), 1.0)  # Ensure it's between 0 and 1
            except Exception as e:
                # Fall back to direct comparison if model evaluation fails
                import logging

                logging.warning(f"Model evaluation failed in AccuracyMetric: {e}")

        # Direct comparison
        if isinstance(expected, str) and isinstance(actual_output, str):
            return 1.0 if expected.strip() == actual_output.strip() else 0.0
        elif expected == actual_output:
            return 1.0

        return 0.0

    def aggregate(self, scores: list[float]) -> float:
        """Aggregate accuracy scores by averaging them.

        Args:
            scores: List of accuracy scores.

        Returns:
            The average accuracy score.
        """
        if not scores:
            return 0.0
        return sum(scores) / len(scores)


class LatencyMetric(Metric):
    """Latency metric for measuring response time."""

    def __init__(self) -> None:
        """Initialize the LatencyMetric."""
        self.start_times: dict[str, float] = {}

    def start_timer(self, example_id: str) -> None:
        """Start the timer for an example.

        Args:
            example_id: The example ID to track.
        """
        self.start_times[example_id] = time.time()

    def evaluate(
        self,
        example: EvalExample,
    ) -> float:
        """Evaluate latency for a single example.

        Args:
            example: The evaluation example.

        Returns:
            The response time in seconds.
        """
        end_time = time.time()
        start_time = self.start_times.get(example.id)

        if start_time is None:
            return 0.0

        return end_time - start_time

    def aggregate(self, scores: list[float]) -> float:
        """Aggregate latency scores by averaging them.

        Args:
            scores: List of latency scores.

        Returns:
            The average latency.
        """
        if not scores:
            return 0.0
        return sum(scores) / len(scores)


class EvalResults(BaseModel):
    """Results from an evaluation run.

    Attributes:
        dataset_name: Name of the dataset used.
        metric_scores: Aggregated scores for each metric.
        example_scores: Individual scores for each example and metric.
    """

    dataset_name: str = Field(..., description="Name of the dataset used")
    metric_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Aggregated scores for each metric",
    )
    example_scores: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Individual scores for each example and metric",
    )
