"""Base model registry for unified model access."""

from typing import Any

from pydantic import BaseModel, Field

from lg_adk.config.settings import Settings


class ModelProvider(BaseModel):
    """Base class for model providers.

    Attributes:
        name: Name of the provider.
    """

    name: str = Field(..., description="Provider name")

    def get_model(self, model_name: str, **kwargs: Any) -> Any:
        """Get a model instance from this provider.

        Args:
            model_name: Name of the model.
            **kwargs: Additional arguments for the model.

        Returns:
            The model instance.
        """
        raise NotImplementedError("Subclass must implement get_model")

    def is_supported_model(self, model_name: str) -> bool:
        """Check if this provider supports a given model.

        Args:
            model_name: Name of the model.

        Returns:
            True if supported, False otherwise.
        """
        raise NotImplementedError("Subclass must implement is_supported_model")

    def generate(self, model_name: str, prompt: str, **kwargs: Any) -> str:
        """Generate text using the model.

        Args:
            model_name: Name of the model.
            prompt: Prompt to generate from.
            **kwargs: Additional arguments for generation.

        Returns:
            Generated text.
        """
        raise NotImplementedError("Subclass must implement generate")

    async def agenerate(self, model_name: str, prompt: str, **kwargs: Any) -> str:
        """Generate text asynchronously using the model.

        Args:
            model_name: Name of the model.
            prompt: Prompt to generate from.
            **kwargs: Additional arguments for generation.

        Returns:
            Generated text.
        """
        raise NotImplementedError("Subclass must implement agenerate")


class ModelRegistry:
    """Registry for model providers.

    Acts as a factory for LLM instances, providing a unified interface
    regardless of the underlying model provider (Ollama, Gemini, OpenAI, etc).

    Attributes:
        _providers: Dictionary of registered model providers.
    """

    _providers: dict[str, ModelProvider] = {}

    @classmethod
    def register_provider(cls, provider: ModelProvider) -> None:
        """Register a model provider.

        Args:
            provider: The model provider to register.
        """
        cls._providers[provider.name] = provider

    @classmethod
    def get_provider(cls, name: str) -> ModelProvider | None:
        """Get a provider by name.

        Args:
            name: Name of the provider.

        Returns:
            The provider instance if found, else None.
        """
        return cls._providers.get(name)

    @classmethod
    def get_providers(cls) -> dict[str, ModelProvider]:
        """Get all registered providers.

        Returns:
            Dictionary of provider name to provider instance.
        """
        return cls._providers

    @classmethod
    def get_model(cls, model_name: str, **kwargs: Any) -> Any:
        """Get a model instance based on model name.

        Args:
            model_name: Name of the model, optionally prefixed with provider.
            **kwargs: Additional arguments to pass to the model.

        Returns:
            An instance of the requested model.

        Raises:
            ValueError: If no provider supports the requested model.
        """
        # Check if the model name has a provider prefix
        if "/" in model_name:
            provider_name, actual_model = model_name.split("/", 1)
            provider = cls.get_provider(provider_name)
            if provider:
                return provider.get_model(actual_model, **kwargs)

        # No provider prefix or specific provider not found, try each provider
        for provider in cls._providers.values():
            if provider.is_supported_model(model_name):
                return provider.get_model(model_name, **kwargs)

        raise ValueError(f"No provider supports model: {model_name}")


def get_model(model_name: str, settings: Settings | None = None, **kwargs: Any) -> Any:
    """Convenience function to get a model instance.

    Args:
        model_name: Name of the model (e.g., "ollama/llama3", "gemini/gemini-pro").
        settings: Optional settings instance.
        **kwargs: Additional arguments to pass to the model.

    Returns:
        An instance of the requested model.
    """
    if settings is None:
        settings = Settings.from_env()

    # Import providers here to avoid circular imports
    from lg_adk.models.providers import GeminiProvider, OllamaProvider

    # Register providers if not already registered
    if not ModelRegistry.get_providers():
        ModelRegistry.register_provider(OllamaProvider(settings=settings))
        ModelRegistry.register_provider(GeminiProvider(settings=settings))

    return ModelRegistry.get_model(model_name, **kwargs)
