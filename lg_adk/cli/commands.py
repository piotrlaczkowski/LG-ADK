"""Command-line interface commands for LG-ADK."""

import importlib
import sys
from pathlib import Path

import click
from rich.console import Console

from lg_adk.eval.dataset import EvalDataset
from lg_adk.eval.evaluator import Evaluator
from lg_adk.eval.metrics import AccuracyMetric, LatencyMetric

console = Console()


@click.group()
def main() -> None:
    """LG-ADK Command Line Interface."""
    pass


@main.command("eval")
@click.argument("agent_path", type=click.Path(exists=True))
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Path to save evaluation results",
)
@click.option(
    "--metrics",
    "-m",
    multiple=True,
    help="Metrics to use for evaluation (e.g., accuracy, latency)",
)
def evaluate(
    agent_path: str,
    dataset_path: str,
    output: str | None,
    metrics: list[str],
) -> None:
    """Evaluate an agent against a dataset.

    AGENT_PATH is the path to the python file containing the agent.
    DATASET_PATH is the path to the JSON file containing the evaluation dataset.
    """
    console.print("[bold]LG-ADK Evaluation[/bold]")

    # Load the agent
    console.print(f"Loading agent from {agent_path}...")
    try:
        agent_path_obj = Path(agent_path).resolve()
        agent_dir = str(agent_path_obj.parent)

        # Add the agent directory to sys.path
        sys.path.insert(0, agent_dir)

        # Import the agent module
        module_name = agent_path_obj.stem
        agent_module = importlib.import_module(module_name)

        # Look for an attribute named 'agent', 'root_agent', or similar
        agent = None
        for attr_name in ["agent", "root_agent", "main_agent"]:
            if hasattr(agent_module, attr_name):
                agent = getattr(agent_module, attr_name)
                break

        if agent is None:
            console.print("[red]Error: Could not find agent in the module.[/red]")
            return

    except Exception as e:
        console.print(f"[red]Error loading agent: {str(e)}[/red]")
        return

    # Load the dataset
    console.print(f"Loading dataset from {dataset_path}...")
    try:
        dataset = EvalDataset.from_json(dataset_path)
    except Exception as e:
        console.print(f"[red]Error loading dataset: {str(e)}[/red]")
        return

    # Configure metrics
    eval_metrics = []
    if not metrics:
        # Default metrics
        eval_metrics = [AccuracyMetric(), LatencyMetric()]
    else:
        for metric_name in metrics:
            if metric_name.lower() == "accuracy":
                eval_metrics.append(AccuracyMetric())
            elif metric_name.lower() == "latency":
                eval_metrics.append(LatencyMetric())
            else:
                console.print(f"[yellow]Warning: Unknown metric '{metric_name}'[/yellow]")

    # Create evaluator
    evaluator = Evaluator(metrics=eval_metrics)

    # Run evaluation
    console.print("Running evaluation...")
    results = evaluator.evaluate(agent, dataset)

    # Save results if output path is provided
    if output:
        evaluator.save_results(results, output)


@main.command("run")
@click.argument("agent_path", type=click.Path(exists=True))
def run_agent(agent_path: str) -> None:
    """Run an agent in interactive mode.

    AGENT_PATH is the path to the python file containing the agent.
    """
    console.print("[bold]LG-ADK Interactive Agent[/bold]")

    # Load the agent
    console.print(f"Loading agent from {agent_path}...")
    try:
        agent_path_obj = Path(agent_path).resolve()
        agent_dir = str(agent_path_obj.parent)

        # Add the agent directory to sys.path
        sys.path.insert(0, agent_dir)

        # Import the agent module
        module_name = agent_path_obj.stem
        agent_module = importlib.import_module(module_name)

        # Look for an attribute named 'agent', 'root_agent', or similar
        agent = None
        for attr_name in ["agent", "root_agent", "main_agent"]:
            if hasattr(agent_module, attr_name):
                agent = getattr(agent_module, attr_name)
                break

        if agent is None:
            console.print("[red]Error: Could not find agent in the module.[/red]")
            return

    except Exception as e:
        console.print(f"[red]Error loading agent: {str(e)}[/red]")
        return

    # Interactive chat loop
    console.print("[bold green]Agent loaded. Type 'exit' to quit.[/bold green]")
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                break

            state = {"input": user_input}
            result = agent.run(state)
            output = result.get("output", "")

            console.print(f"\n[bold cyan]Agent:[/bold cyan] {output}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")

    console.print("\n[bold]Goodbye![/bold]")


@main.command("debug")
@click.argument("agent_path", type=click.Path(exists=True))
def debug_agent(agent_path: str) -> None:
    """Debug an agent using langgraph-cli.

    AGENT_PATH is the path to the python file containing the agent.
    """
    console.print("[bold]LG-ADK Agent Debugger[/bold]")
    console.print(f"Launching langgraph-cli debugger for {agent_path}...")

    try:
        import shutil
        import subprocess

        # Find the absolute path to the 'langgraph' executable
        langgraph_path = shutil.which("langgraph")
        if not langgraph_path:
            console.print("[red]Error: 'langgraph' executable not found in PATH.[/red]")
            return

        # Validate agent_path to avoid untrusted input
        agent_path_obj = Path(agent_path).resolve()
        agent_path_str = str(agent_path_obj)
        # Only allow safe, absolute paths
        if not agent_path_obj.is_file():
            console.print("[red]Error: Provided agent_path is not a valid file.[/red]")
            return

        # S603: subprocess call is safe; langgraph_path and agent_path_str are validated absolute paths.
        subprocess.run(
            [
                langgraph_path,
                "server",
                "--file",
                agent_path_str,
            ],
            check=True,
        )  # noqa: S603
    except ImportError:
        console.print("[red]Error: langgraph-cli is not installed.[/red]")
        console.print("Install it with: pip install langgraph-cli")
    except Exception as e:
        console.print(f"[red]Error launching debugger: {str(e)}[/red]")


if __name__ == "__main__":
    main()
