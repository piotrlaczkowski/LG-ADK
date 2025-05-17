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
        morphik_host: Hostname for Morphik DB instance.
        morphik_port: Port for Morphik DB instance.
        morphik_api_key: API key for Morphik DB.
        morphik_default_user: Default user ID for Morphik DB.
        morphik_default_folder: Default folder for Morphik DB.
        use_morphik_as_default: Whether to use Morphik as the default database.
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
    morphik_host: str = Field(
        "localhost",
        description="Hostname for Morphik DB",
    )
    morphik_port: int = Field(
        11434,
        description="Port for Morphik DB",
    )
    morphik_api_key: SecretStr | None = Field(
        None,
        description="API key for Morphik DB",
    )
    morphik_default_user: str = Field(
        "default",
        description="Default user ID for Morphik DB",
    )
    morphik_default_folder: str = Field(
        "lg-adk",
        description="Default folder for Morphik DB",
    )
    use_morphik_as_default: bool = Field(
        False,
        description="Whether to use Morphik as the default database",
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
            morphik_host=os.environ.get("MORPHIK_HOST", "localhost"),
            morphik_port=int(os.environ.get("MORPHIK_PORT", "11434")),
            morphik_api_key=SecretStr(os.environ.get("MORPHIK_API_KEY", ""))
            if os.environ.get("MORPHIK_API_KEY")
            else None,
            morphik_default_user=os.environ.get("MORPHIK_DEFAULT_USER", "default"),
            morphik_default_folder=os.environ.get("MORPHIK_DEFAULT_FOLDER", "lg-adk"),
            use_morphik_as_default=os.environ.get("USE_MORPHIK_AS_DEFAULT", "false").lower() == "true",
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to a dictionary.

        Returns:
            Dictionary of settings.
        """
        result = {}
        for key, value in self.model_dump().items():
            if key in ["openai_api_key", "google_api_key", "morphik_api_key"] and value is not None:
                # Handle SecretStr
                result[key] = value.get_secret_value()
            else:
                result[key] = value

        return result
