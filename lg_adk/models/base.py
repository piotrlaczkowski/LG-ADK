"""
Base model registry for unified model access.
"""

from typing import Any, Dict, Optional, Type, Union

import litellm
from pydantic import BaseModel, Field

from lg_adk.config.settings import Settings


class ModelProvider(BaseModel):
    """Base class for model providers."""
    
    name: str = Field(..., description="Provider name")
    
    def get_model(self, model_name: str, **kwargs: Any) -> Any:
        """Get a model instance from this provider."""
        raise NotImplementedError("Subclass must implement get_model")
    
    def is_supported_model(self, model_name: str) -> bool:
        """Check if this provider supports a given model."""
        raise NotImplementedError("Subclass must implement is_supported_model")
    
    def generate(self, model_name: str, prompt: str, **kwargs: Any) -> str:
        """Generate text using the model."""
        raise NotImplementedError("Subclass must implement generate")
    
    async def agenerate(self, model_name: str, prompt: str, **kwargs: Any) -> str:
        """Generate text asynchronously using the model."""
        raise NotImplementedError("Subclass must implement agenerate")


class ModelRegistry:
    """
    Registry for model providers.
    
    Acts as a factory for LLM instances, providing a unified interface
    regardless of the underlying model provider (Ollama, Gemini, OpenAI, etc).
    """
    
    _providers: Dict[str, ModelProvider] = {}
    
    @classmethod
    def register_provider(cls, provider: ModelProvider) -> None:
        """Register a model provider."""
        cls._providers[provider.name] = provider
    
    @classmethod
    def get_provider(cls, name: str) -> Optional[ModelProvider]:
        """Get a provider by name."""
        return cls._providers.get(name)
    
    @classmethod
    def get_providers(cls) -> Dict[str, ModelProvider]:
        """Get all registered providers."""
        return cls._providers
    
    @classmethod
    def get_model(cls, model_name: str, **kwargs: Any) -> Any:
        """
        Get a model instance based on model name.
        
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


def get_model(model_name: str, settings: Optional[Settings] = None, **kwargs: Any) -> Any:
    """
    Convenience function to get a model instance.
    
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
    from lg_adk.models.providers import (
        GeminiProvider,
        OllamaProvider,
    )
    
    # Register providers if not already registered
    if not ModelRegistry.get_providers():
        ModelRegistry.register_provider(OllamaProvider(settings=settings))
        ModelRegistry.register_provider(GeminiProvider(settings=settings))
    
    return ModelRegistry.get_model(model_name, **kwargs) 