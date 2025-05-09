# 🧪 Agent Evaluation Example with LG-ADK

This example demonstrates how to evaluate an agent using the evaluation tools provided by LG-ADK.

---

!!! tip "You can copy and run this example as a script."

```python
import json
import os

from lg_adk import Agent, EvalDataset, Evaluator
from lg_adk.eval.metrics import AccuracyMetric, LatencyMetric

# Create a simple agent to evaluate
agent = Agent(name="qa_agent", llm="ollama/llama3", description="Answers factual questions")

# Create or load an evaluation dataset
# First, define a dataset
dataset_data = {
    "name": "General Knowledge Questions",
    "description": "Basic factual questions to test agent knowledge",
    "examples": [
        {"id": "q1", "input": "What is the capital of France?", "expected_output": "The capital of France is Paris."},
        {
            "id": "q2",
            "input": "Who wrote the play Romeo and Juliet?",
            "expected_output": "William Shakespeare wrote Romeo and Juliet.",
        },
        {
            "id": "q3",
            "input": "What is the largest planet in our solar system?",
            "expected_output": "Jupiter is the largest planet in our solar system.",
        },
        {
            "id": "q4",
            "input": "What is the chemical symbol for gold?",
            "expected_output": "The chemical symbol for gold is Au.",
        },
        {
            "id": "q5",
            "input": "Who painted the Mona Lisa?",
            "expected_output": "Leonardo da Vinci painted the Mona Lisa.",
        },
    ],
}

# Save the dataset to a file
dataset_path = "general_knowledge_dataset.json"
with open(dataset_path, "w") as f:
    json.dump(dataset_data, f, indent=2)

# Load the dataset
dataset = EvalDataset.from_json(dataset_path)

# Create metrics for evaluation
metrics = [
    AccuracyMetric(),  # Measures how accurate the answers are
    LatencyMetric(),  # Measures response time
]

# Create an evaluator
evaluator = Evaluator(metrics=metrics)

# Run the evaluation
print("Starting evaluation...")
results = evaluator.evaluate(agent, dataset)

# Save the results
results_path = "evaluation_results.json"
evaluator.save_results(results, results_path)

print(f"\nEvaluation complete. Results saved to {results_path}")
print(f"Accuracy: {results.metric_scores.get('AccuracyMetric', 0):.4f}")
print(f"Average Latency: {results.metric_scores.get('LatencyMetric', 0):.4f} seconds")

# Clean up the dataset file if needed
# os.remove(dataset_path)
```
