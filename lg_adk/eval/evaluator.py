"""Evaluator for LG-ADK agents."""

import json
from typing import Any

from rich.console import Console
from rich.table import Table

from lg_adk.agents.base import Agent
from lg_adk.eval.dataset import EvalDataset
from lg_adk.eval.metrics import AccuracyMetric, EvalResults, LatencyMetric, Metric


class Evaluator:
    """Evaluator for LG-ADK agents.

    This class provides methods for evaluating agents against datasets
    using different metrics.
    """

    def __init__(self, metrics: list[Metric] | None = None):
        """Initialize the evaluator.

        Args:
            metrics: List of metrics to use for evaluation.
        """
        self.metrics = metrics or [AccuracyMetric(), LatencyMetric()]
        self.console = Console()

    def evaluate(
        self,
        agent: Agent,
        dataset: EvalDataset,
        **kwargs: Any,
    ) -> EvalResults:
        """Evaluate an agent against a dataset.

        Args:
            agent: The agent to evaluate.
            dataset: The dataset to evaluate against.
            **kwargs: Additional arguments to pass to the agent.

        Returns:
            Evaluation results.
        """
        # Initialize results
        results = EvalResults(
            dataset_name=dataset.name,
            metric_scores={},
            example_scores={},
        )

        # Track scores for each metric
        metric_scores: dict[str, list[float]] = {metric.__class__.__name__: [] for metric in self.metrics}

        # Print evaluation header
        self.console.print(f"\n[bold]Evaluating agent on {dataset.name}[/bold]\n")

        # Create a progress table
        table = Table(title=f"Evaluation Progress ({len(dataset.examples)} examples)")
        table.add_column("Example ID")
        table.add_column("Input")
        for metric in self.metrics:
            table.add_column(metric.__class__.__name__)

        # Initialize latency metric if present
        latency_metric = None
        for metric in self.metrics:
            if isinstance(metric, LatencyMetric):
                latency_metric = metric
                break

        # Evaluate each example
        for example in dataset.examples:
            example_id = example.id

            # Start timer if latency metric is used
            if latency_metric:
                latency_metric.start_timer(example_id)

            # Run the agent
            state = {"input": example.input}
            try:
                result = agent.run(state)
                output = result.get("output", "")
            except Exception as e:
                self.console.print(f"[red]Error running agent on example {example_id}: {str(e)}[/red]")
                output = f"ERROR: {str(e)}"

            # Evaluate with each metric
            example_scores = {}
            table_row = [example_id, str(example.input)[:50]]

            for metric in self.metrics:
                metric_name = metric.__class__.__name__
                if isinstance(metric, LatencyMetric):
                    score = metric.evaluate(example)
                else:
                    score = metric.evaluate(example, output, **kwargs)
                example_scores[metric_name] = score
                metric_scores[metric_name].append(score)
                table_row.append(f"{score:.4f}")

            # Add row to progress table
            table.add_row(*table_row)

            # Store example scores
            results.example_scores[example_id] = example_scores

        # Print progress table
        self.console.print(table)

        # Calculate aggregate metrics
        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            scores = metric_scores[metric_name]
            aggregate_score = metric.aggregate(scores)
            results.metric_scores[metric_name] = aggregate_score

        # Print summary
        self.print_summary(results)

        return results

    def print_summary(self, results: EvalResults) -> None:
        """Print a summary of evaluation results.

        Args:
            results: The evaluation results to summarize.
        """
        summary_table = Table(title="Evaluation Summary")
        summary_table.add_column("Metric")
        summary_table.add_column("Score")

        for metric_name, score in results.metric_scores.items():
            # Format score based on metric type
            formatted_score = f"{score:.4f} seconds" if metric_name == "LatencyMetric" else f"{score:.4f}"

            summary_table.add_row(metric_name, formatted_score)

        self.console.print("\n")
        self.console.print(summary_table)

    def save_results(self, results: EvalResults, path: str) -> None:
        """Save evaluation results to a JSON file.

        Args:
            results: The evaluation results to save.
            path: Path to save the results to.
        """
        from pathlib import Path

        with Path(path).open("w") as f:
            json.dump(results.model_dump(), f, indent=2)

        self.console.print(f"\n[green]Results saved to {path}[/green]")
