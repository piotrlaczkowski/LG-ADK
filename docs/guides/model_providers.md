# Model Providers in LG-ADK

This guide explains how to work with different model providers in the LangGraph Agent Development Kit.

## Understanding Model Providers

LG-ADK abstracts away the complexity of working with different language model APIs through a unified provider system. This allows you to:

1. **Switch Models Easily**: Change between different models with minimal code changes
2. **Support Multiple Providers**: Work with OpenAI, Google Gemini, Anthropic, and other models
3. **Use Local Models**: Integrate with Ollama for local model inference
4. **Custom Providers**: Create your own providers for specialized needs

## Getting Started with Models

The simplest way to get a model is through the `get_model` function:

```python
from lg_adk.models import get_model

# Get an OpenAI model
openai_model = get_model("openai/gpt-4")

# Get a Google Gemini model
gemini_model = get_model("google/gemini-pro")

# Get an Anthropic model
claude_model = get_model("anthropic/claude-3-opus")

# Get an Ollama model (local)
local_model = get_model("ollama/llama3")
```

## Available Model Providers

LG-ADK supports these model providers out of the box:

### OpenAI

```python
# OpenAI GPT-4 model
model = get_model("openai/gpt-4")

# OpenAI GPT-4o model
model = get_model("openai/gpt-4o")

# OpenAI GPT-3.5 Turbo model
model = get_model("openai/gpt-3.5-turbo")
```

### Google Gemini

```python
# Google Gemini Pro model
model = get_model("google/gemini-pro")

# Google Gemini Ultra
model = get_model("google/gemini-1.5-pro")
```

### Anthropic

```python
# Anthropic Claude 3 Opus
model = get_model("anthropic/claude-3-opus")

# Anthropic Claude 3 Sonnet
model = get_model("anthropic/claude-3-sonnet")

# Anthropic Claude 3 Haiku
model = get_model("anthropic/claude-3-haiku")
```

### Ollama (Local Models)

```python
# Local Llama 3 model via Ollama
model = get_model("ollama/llama3")

# Local Mixtral model
model = get_model("ollama/mixtral")
```

## Model Configuration

Each model can be configured with provider-specific parameters:

```python
from lg_adk.models import get_model

# Configure an OpenAI model
openai_model = get_model(
    "openai/gpt-4",
    temperature=0.7,
    max_tokens=2000,
    streaming=True
)

# Configure a Google model
google_model = get_model(
    "google/gemini-pro",
    temperature=0.2,
    top_p=0.95,
    top_k=40
)
```

## Working with the ModelRegistry

For more advanced usage, you can interact directly with the `ModelRegistry`:

```python
from lg_adk.models import ModelRegistry

# Get the registry singleton
registry = ModelRegistry.get_instance()

# Register a custom model configuration
registry.register(
    "openai/gpt-4-custom",
    provider="openai",
    model_name="gpt-4",
    temperature=0.5,
    top_p=0.9
)

# Get the custom model
custom_model = registry.get_model("openai/gpt-4-custom")
```

## Provider-Specific Authentication

Different providers require different authentication mechanisms:

### OpenAI Authentication

```python
import os
from lg_adk.models import ModelRegistry

# Set OpenAI API key in environment variable
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Or configure explicitly
registry = ModelRegistry.get_instance()
registry.configure_provider(
    "openai",
    api_key="your-api-key",
    organization="your-org-id"  # Optional
)
```

### Google Authentication

```python
import os
from lg_adk.models import ModelRegistry

# Set Google API key in environment variable
os.environ["GOOGLE_API_KEY"] = "your-api-key"

# Or configure explicitly
registry = ModelRegistry.get_instance()
registry.configure_provider(
    "google",
    api_key="your-api-key"
)
```

### Anthropic Authentication

```python
import os
from lg_adk.models import ModelRegistry

# Set Anthropic API key in environment variable
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

# Or configure explicitly
registry = ModelRegistry.get_instance()
registry.configure_provider(
    "anthropic",
    api_key="your-api-key"
)
```

### Ollama Authentication

```python
from lg_adk.models import ModelRegistry

# Configure Ollama endpoint (defaults to http://localhost:11434)
registry = ModelRegistry.get_instance()
registry.configure_provider(
    "ollama",
    base_url="http://localhost:11434"
)
```

## Making Direct Model Calls

You can use the model objects directly for generation:

```python
from lg_adk.models import get_model

# Get a model
model = get_model("openai/gpt-4")

# Generate a response
response = model.generate(
    "Explain the theory of relativity briefly.",
    system_prompt="You are a helpful physics tutor."
)

print(response)
```

## Streaming Responses

Enable streaming for real-time responses:

```python
from lg_adk.models import get_model

# Get a model with streaming enabled
model = get_model("openai/gpt-4", streaming=True)

# Stream a response
for chunk in model.generate_stream(
    "Write a short poem about the ocean.",
    system_prompt="You are a poet."
):
    print(chunk, end="", flush=True)
```

## Tool Calling with Models

Enable tool calling for models that support it:

```python
from lg_adk.models import get_model
from typing import List, Dict, Any

# Define your tools
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # In a real app, you would call a weather API
    return f"Sunny and 75°F in {location}"

tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        }
    }
]

# Tool calling functions
def execute_tool_call(tool_call: Dict[str, Any]) -> str:
    if tool_call["name"] == "get_weather":
        location = tool_call["parameters"]["location"]
        return get_weather(location)
    return "Unknown tool"

# Get a model
model = get_model("openai/gpt-4", tools=tools)

# Generate with tool calling
response = model.generate(
    "What's the weather like in Miami?",
    system_prompt="You can use tools to answer questions."
)

# Check for tool calls in the response
if hasattr(response, "tool_calls") and response.tool_calls:
    for tool_call in response.tool_calls:
        tool_result = execute_tool_call(tool_call)

        # Send the result back to the model
        follow_up = model.generate(
            f"Tool result: {tool_result}",
            system_prompt="You can use tools to answer questions.",
            previous_messages=[
                {"role": "user", "content": "What's the weather like in Miami?"},
                {"role": "assistant", "content": response.content}
            ]
        )
        print(follow_up)
else:
    print(response)
```

## Async Model Usage

For asynchronous applications:

```python
import asyncio
from lg_adk.models import get_model

async def generate_async():
    # Get a model
    model = get_model("openai/gpt-4")

    # Generate asynchronously
    response = await model.agenerate(
        "What are the benefits of asynchronous programming?",
        system_prompt="You are a programming instructor."
    )

    print(response)

    # Stream asynchronously
    async for chunk in model.agenerate_stream(
        "Explain coroutines in Python.",
        system_prompt="You are a Python expert."
    ):
        print(chunk, end="", flush=True)

# Run the async function
asyncio.run(generate_async())
```

## Custom Model Providers

You can create your own model provider:

```python
from lg_adk.models import ModelProvider, ModelRegistry
from typing import Dict, Any, AsyncGenerator, Optional, List

class CustomModelProvider(ModelProvider):
    """Custom model provider implementation."""

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        # Initialize your custom API client here

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        previous_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """Generate text using your custom API."""
        # Implement synchronous generation
        # ...
        return "Generated response"

    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        previous_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """Generate text asynchronously using your custom API."""
        # Implement asynchronous generation
        # ...
        return "Generated response"

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        previous_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ):
        """Stream generated text from your custom API."""
        # Implement synchronous streaming
        yield "Generated "
        yield "response "
        yield "in chunks"

    async def agenerate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        previous_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream generated text asynchronously from your custom API."""
        # Implement asynchronous streaming
        yield "Generated "
        yield "response "
        yield "in chunks"

# Register your custom provider
registry = ModelRegistry.get_instance()
registry.register_provider("custom", CustomModelProvider)

# Configure your provider
registry.configure_provider("custom", api_key="your-api-key")

# Register a model using your provider
registry.register(
    "custom/my-model",
    provider="custom",
    model_name="my-model-name",
    temperature=0.5
)

# Get your custom model
model = registry.get_model("custom/my-model")
```

## Model Caching

Improve performance with model response caching:

```python
from lg_adk.models import ModelRegistry, get_model
from lg_adk.utils.caching import enable_model_caching

# Enable caching for all models
enable_model_caching()

# Or enable caching for specific models
model = get_model("openai/gpt-4")
model.enable_caching(cache_ttl=3600)  # Cache for 1 hour

# First call (will make API request)
response1 = model.generate(
    "What is the capital of France?",
    system_prompt="You are a geography expert."
)

# Second call with same inputs (will use cache)
response2 = model.generate(
    "What is the capital of France?",
    system_prompt="You are a geography expert."
)

# Different prompt (will make new API request)
response3 = model.generate(
    "What is the capital of Italy?",
    system_prompt="You are a geography expert."
)
```

## Model Fallbacks

Create fallback chains for reliability:

```python
from lg_adk.models import ModelRegistry, get_model

# Create a fallback chain
registry = ModelRegistry.get_instance()
registry.register_fallback_chain(
    "reliable-completion",
    [
        "openai/gpt-4",
        "anthropic/claude-3-sonnet",
        "ollama/llama3"  # Local fallback
    ]
)

# Use the fallback chain
model = registry.get_model("reliable-completion")
response = model.generate("Explain quantum computing.")
```

## Model Authentication from Config Files

Load provider configurations from a file:

```python
from lg_adk.models import ModelRegistry
from lg_adk.config import load_config

# Load config from a YAML file
config = load_config("config.yaml")

# Initialize the registry with config
registry = ModelRegistry.get_instance()
registry.configure_from_config(config.model_providers)

# Now you can use get_model without explicit configuration
model = get_model("openai/gpt-4")
```

Example `config.yaml`:

```yaml
model_providers:
  openai:
    api_key: ${OPENAI_API_KEY}  # Environment variable
    organization: "org-123"
  google:
    api_key: ${GOOGLE_API_KEY}
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
  ollama:
    base_url: "http://localhost:11434"
```

## Complete Example: Multi-Provider Application

Here's a complete example using multiple model providers:

```python
import os
import asyncio
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from lg_adk.models import get_model, ModelRegistry

# Set up environment variables for API keys
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

# Model configuration
class ModelConfig(BaseModel):
    provider: str
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1000

# Application class
class MultiModelApp:
    def __init__(self):
        self.registry = ModelRegistry.get_instance()

        # Configure the OpenAI provider
        self.registry.configure_provider(
            "openai",
            api_key=os.environ.get("OPENAI_API_KEY")
        )

        # Configure the Anthropic provider
        self.registry.configure_provider(
            "anthropic",
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

        # Configure Ollama for local models
        self.registry.configure_provider(
            "ollama",
            base_url="http://localhost:11434"
        )

        # Register models
        self.models = {
            "creative": get_model(
                "openai/gpt-4",
                temperature=0.8,
                max_tokens=2000
            ),
            "precise": get_model(
                "anthropic/claude-3-opus",
                temperature=0.2,
                max_tokens=1000
            ),
            "fast": get_model(
                "anthropic/claude-3-haiku",
                temperature=0.7,
                max_tokens=500
            ),
            "local": get_model(
                "ollama/llama3",
                temperature=0.7
            )
        }

    def generate(self, prompt: str, model_type: str) -> str:
        """Generate using the specified model type."""
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")

        model = self.models[model_type]
        return model.generate(prompt)

    async def generate_from_all(self, prompt: str) -> Dict[str, str]:
        """Generate responses from all models asynchronously."""
        tasks = {}

        for model_type, model in self.models.items():
            tasks[model_type] = asyncio.create_task(
                model.agenerate(prompt)
            )

        # Await all tasks
        results = {}
        for model_type, task in tasks.items():
            try:
                results[model_type] = await task
            except Exception as e:
                results[model_type] = f"Error: {str(e)}"

        return results

# Example usage
async def main():
    app = MultiModelApp()

    # Single model generation
    creative_response = app.generate(
        "Write a story about a robot learning to paint.",
        "creative"
    )
    print(f"Creative model response:\n{creative_response}\n")

    precise_response = app.generate(
        "Explain the process of photosynthesis.",
        "precise"
    )
    print(f"Precise model response:\n{precise_response}\n")

    # Generate from all models
    prompt = "What are the ethical implications of artificial intelligence?"
    all_responses = await app.generate_from_all(prompt)

    print("\nResponses from all models:")
    for model_type, response in all_responses.items():
        print(f"\n--- {model_type.upper()} MODEL ---")
        print(response[:200] + "..." if len(response) > 200 else response)

if __name__ == "__main__":
    asyncio.run(main())
```

## Best Practices for Working with Model Providers

1. **Environment Variables**: Store API keys in environment variables for security
2. **Model Selection**: Use the right model for the task (creative vs. precise)
3. **Fallbacks**: Set up fallback chains for mission-critical applications
4. **Caching**: Enable caching for frequently used prompts
5. **Provider Abstraction**: Use the provider abstraction to make your code provider-agnostic
6. **Local Testing**: Use Ollama models for local testing before deploying
7. **Monitoring**: Track usage and performance of different providers
8. **Rate Limiting**: Be aware of rate limits for different providers
9. **Cost Management**: Use cheaper models for less critical tasks
10. **Error Handling**: Implement robust error handling for API failures

By leveraging LG-ADK's model provider system, you can build applications that are not tied to any specific model provider, allowing for flexibility, reliability, and optimal performance.
