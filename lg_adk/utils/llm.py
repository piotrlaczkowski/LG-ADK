"""Utility functions for working with LLMs."""

from typing import Any

from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI

from lg_adk.config.settings import Settings


def get_llm_from_name(
    llm_name: str,
    settings: Settings | None = None,
    **kwargs: Any,
) -> Any:
    """Get an LLM instance from a name.

    Args:
        llm_name: Name of the LLM to use.
        settings: Settings object.
        **kwargs: Additional arguments to pass to the LLM constructor.

    Returns:
        An LLM instance.
    """
    if settings is None:
        settings = Settings.from_env()

    # Handle Ollama models
    if llm_name.startswith("ollama/"):
        model_name = llm_name.split("/")[1]
        return Ollama(
            model=model_name,
            base_url=settings.ollama_base_url,
            **kwargs,
        )

    # Handle OpenAI models
    if llm_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]:
        if settings.openai_api_key is None:
            raise ValueError("OpenAI API key not provided")

        return ChatOpenAI(
            model_name=llm_name,
            api_key=settings.openai_api_key.get_secret_value(),
            **kwargs,
        )

    # Unknown model
    raise ValueError(f"Unknown LLM: {llm_name}")
