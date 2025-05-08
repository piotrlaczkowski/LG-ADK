"""Evaluation dataset for LG-ADK."""

import json

from pydantic import BaseModel, Field


class EvalExample(BaseModel):
    """A single evaluation example.

    Attributes:
        example_id: Unique identifier for the example.
        agent_input: Input to the agent.
        expected_output: Expected output from the agent.
        metadata: Additional metadata for the example.
    """

    example_id: str = Field(..., alias="id", description="Unique identifier for the example")
    agent_input: str | dict = Field(..., alias="input", description="Input to the agent")
    expected_output: str | dict | None = Field(
        None,
        description="Expected output from the agent",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata for the example",
    )


class EvalDataset(BaseModel):
    """A dataset for evaluating agents.

    Attributes:
        name: Name of the dataset.
        description: Description of the dataset.
        examples: List of evaluation examples.
    """

    name: str = Field(..., description="Name of the dataset")
    description: str = Field("", description="Description of the dataset")
    examples: list[EvalExample] = Field(
        default_factory=list,
        description="List of evaluation examples",
    )

    @classmethod
    def from_json(cls, path: str) -> "EvalDataset":
        """Load an evaluation dataset from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            An EvalDataset instance.
        """
        from pathlib import Path

        with Path(path).open() as f:
            data = json.load(f)

        examples = [EvalExample(**ex) for ex in data.get("examples", [])]
        return cls(
            name=data.get("name", "Unnamed Dataset"),
            description=data.get("description", ""),
            examples=examples,
        )

    def to_json(self, path: str) -> None:
        """Save the evaluation dataset to a JSON file.

        Args:
            path: Path to save the JSON file.
        """
        from pathlib import Path

        with Path(path).open("w") as f:
            json.dump(self.model_dump(), f, indent=2)

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)
