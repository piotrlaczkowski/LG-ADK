"""
Model providers for different LLM backends.
"""

import os
from typing import Any, Dict, List, Optional, Union

import google.generativeai as genai
import litellm
from langchain_community.llms.ollama import Ollama
from pydantic import SecretStr

from lg_adk.config.settings import Settings
from lg_adk.models.base import ModelProvider


class OllamaProvider(ModelProvider):
    """Ollama model provider."""
    
    name: str = "ollama"
    settings: Settings
    
    def __init__(self, settings: Optional[Settings] = None, **data: Any):
        super().__init__(name=self.name, **data)
        self.settings = settings or Settings.from_env()
    
    def get_model(self, model_name: str, **kwargs: Any) -> Any:
        """Get an Ollama model instance."""
        return Ollama(
            model=model_name,
            base_url=self.settings.ollama_base_url,
            **kwargs,
        )
    
    def is_supported_model(self, model_name: str) -> bool:
        """Check if a model is supported by Ollama."""
        # Ollama supports any model that's been pulled
        return True
    
    def generate(self, model_name: str, prompt: str, **kwargs: Any) -> str:
        """Generate text using Ollama model."""
        model = self.get_model(model_name, **kwargs)
        return model.invoke(prompt)
    
    async def agenerate(self, model_name: str, prompt: str, **kwargs: Any) -> str:
        """Generate text asynchronously using Ollama model."""
        model = self.get_model(model_name, **kwargs)
        return await model.ainvoke(prompt)


class GeminiProvider(ModelProvider):
    """Google Gemini model provider."""
    
    name: str = "gemini"
    settings: Settings
    _initialized: bool = False
    
    def __init__(self, settings: Optional[Settings] = None, **data: Any):
        super().__init__(name=self.name, **data)
        self.settings = settings or Settings.from_env()
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the Gemini API."""
        if not self._initialized and self.settings.google_api_key:
            genai.configure(api_key=self.settings.google_api_key.get_secret_value())
            self._initialized = True
    
    def get_model(self, model_name: str, **kwargs: Any) -> Any:
        """Get a Gemini model instance."""
        self._initialize()
        
        # Map model names to actual Gemini models
        model_mapping = {
            "gemini-pro": "gemini-pro",
            "gemini-pro-vision": "gemini-pro-vision",
            "gemini-ultra": "gemini-ultra",
        }
        
        actual_model = model_mapping.get(model_name, model_name)
        
        # Use litellm to provide a more standard interface
        return litellm.Gemini(model=actual_model, **kwargs)
    
    def is_supported_model(self, model_name: str) -> bool:
        """Check if a model is supported by Gemini."""
        supported_models = [
            "gemini-pro", 
            "gemini-pro-vision", 
            "gemini-ultra",
        ]
        return model_name in supported_models
    
    def generate(self, model_name: str, prompt: str, **kwargs: Any) -> str:
        """Generate text using Gemini model."""
        model = self.get_model(model_name, **kwargs)
        return model.invoke(prompt)
    
    async def agenerate(self, model_name: str, prompt: str, **kwargs: Any) -> str:
        """Generate text asynchronously using Gemini model."""
        model = self.get_model(model_name, **kwargs)
        return await model.ainvoke(prompt) 