"""Configuration settings for LG-ADK."""

import os
from typing import Any

from pydantic import BaseModel, Field, SecretStr


class Settings(BaseModel):
    """Settings for LG-ADK.

    Attributes:
        openai_api_key: OpenAI API key for using OpenAI models.
        google_api_key: Google API key for using Gemini models.
        ollama_base_url: Base URL for Ollama API.
        default_llm: Default language model to use.
        debug: Whether to enable debug mode.
        db_url: Database URL for persistent storage.
        vector_store_path: Path to the vector store.
    """

    openai_api_key: SecretStr | None = Field(
        None,
        description="OpenAI API key",
    )
    google_api_key: SecretStr | None = Field(
        None,
        description="Google API key for Gemini models",
    )
    ollama_base_url: str = Field(
        "http://localhost:11434",
        description="Ollama API base URL",
    )
    default_llm: str = Field(
        "ollama/llama3",
        description="Default LLM to use",
    )
    debug: bool = Field(False, description="Debug mode")
    db_url: str | None = Field(
        None,
        description="Database URL for persistent storage",
    )
    vector_store_path: str = Field(
        "./.vector_store",
        description="Path to vector store",
    )

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables.

        Returns:
            Settings instance populated from environment variables.
        """
        return cls(
            openai_api_key=SecretStr(os.environ.get("OPENAI_API_KEY", ""))
            if os.environ.get("OPENAI_API_KEY")
            else None,
            google_api_key=SecretStr(os.environ.get("GOOGLE_API_KEY", ""))
            if os.environ.get("GOOGLE_API_KEY")
            else None,
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            default_llm=os.environ.get("DEFAULT_LLM", "ollama/llama3"),
            debug=os.environ.get("DEBUG", "false").lower() == "true",
            db_url=os.environ.get("DB_URL"),
            vector_store_path=os.environ.get("VECTOR_STORE_PATH", "./.vector_store"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to a dictionary.

        Returns:
            Dictionary of settings.
        """
        result = {}
        for key, value in self.model_dump().items():
            if key in ["openai_api_key", "google_api_key"] and value is not None:
                # Handle SecretStr
                result[key] = value.get_secret_value()
            else:
                result[key] = value

        return result
