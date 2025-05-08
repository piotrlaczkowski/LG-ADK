# Using Multiple Model Providers

This example demonstrates how to use LG-ADK's model provider system to create an application that can switch between different language model providers.

## Multi-Provider Agent

In this example, we'll create a simple agent that can use different model providers (OpenAI, Google, Anthropic, or Ollama) based on user preferences:

```python
import os
from typing import Dict, Optional
from lg_adk import Agent, GraphBuilder
from lg_adk.models import get_model, ModelRegistry
from lg_adk.human import HumanInputTool
from lg_adk.tools import WebSearchTool
from lg_adk.memory import MemoryManager

# Set up API keys (in a real application, these would be in environment variables)
os.environ["OPENAI_API_KEY"] = "your-openai-key"  # Replace with your actual key
os.environ["GOOGLE_API_KEY"] = "your-google-key"  # Replace with your actual key
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"  # Replace with your actual key

class ModelSwitchingAgent:
    """An agent that can switch between different model providers."""

    def __init__(self):
        # Available models configuration
        self.models = {
            "openai": "openai/gpt-4",
            "google": "google/gemini-pro",
            "anthropic": "anthropic/claude-3-sonnet",
            "ollama": "ollama/llama3"  # Local model
        }

        # Start with OpenAI as default
        self.current_model = "openai"

        # Initialize the agent with the default model
        self.setup_agent()

    def setup_agent(self):
        """Set up the agent with the current model."""
        model = get_model(self.models[self.current_model])

        # Create a new agent with the selected model
        self.agent = Agent(
            agent_name="model_switcher",
            llm=model,
            system_prompt=(
                "You are a helpful assistant powered by a language model. "
                "You can search the web to answer questions and can switch between "
                "different model providers based on user requests."
            )
        )

        # Add tools
        self.agent.add_tool(WebSearchTool())
        self.agent.add_tool(HumanInputTool())

        # Create a graph with the agent
        self.builder = GraphBuilder()
        self.builder.add_agent(self.agent)
        self.builder.add_memory(MemoryManager())
        self.builder.enable_human_feedback()

        # Build the graph
        self.graph = self.builder.build()

    def switch_model(self, provider: str) -> str:
        """Switch to a different model provider."""
        if provider not in self.models:
            return f"Unknown provider: {provider}. Available providers: {', '.join(self.models.keys())}"

        # Switch the model
        self.current_model = provider
        self.setup_agent()

        return f"Switched to {provider} model: {self.models[provider]}"

    def process_message(self, message: str, session_id: Optional[str] = None) -> Dict:
        """Process a user message, handling model switching commands."""
        # Check for model switching command
        if message.lower().startswith("switch to "):
            provider = message.lower().replace("switch to ", "").strip()
            result = self.switch_model(provider)
            return {"output": result}

        # Regular message processing with the current model
        return self.graph.invoke({"input": message}, {"session_id": session_id})

# Usage example
if __name__ == "__main__":
    agent = ModelSwitchingAgent()

    # Create a unique session ID for this conversation
    import uuid
    session_id = str(uuid.uuid4())

    # Interactive chat loop
    print("Multi-Provider Agent Chat (type 'exit' to quit)")
    print("Current model: " + agent.current_model)
    print("Available commands: 'switch to openai', 'switch to google', 'switch to anthropic', 'switch to ollama'")
    print("---------------------------------------------------")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        response = agent.process_message(user_input, session_id)
        print(f"Agent ({agent.current_model}): {response['output']}")
```

## Comparing Models Example

This example demonstrates how to compare responses from different model providers for the same prompt:

```python
import asyncio
import os
from typing import Dict, List
from lg_adk.models import get_model, ModelRegistry

# Set up API keys (in a real application, these would be in environment variables)
os.environ["OPENAI_API_KEY"] = "your-openai-key"  # Replace with your actual key
os.environ["GOOGLE_API_KEY"] = "your-google-key"  # Replace with your actual key
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"  # Replace with your actual key

async def compare_models(prompt: str, system_prompt: str = None) -> Dict[str, str]:
    """Compare responses from different model providers for the same prompt."""
    # Configure models to use
    models = {
        "OpenAI GPT-4": get_model("openai/gpt-4"),
        "Google Gemini Pro": get_model("google/gemini-pro"),
        "Anthropic Claude": get_model("anthropic/claude-3-sonnet"),
        "Ollama Llama3": get_model("ollama/llama3", temperature=0.7)
    }

    # Create tasks for each model
    tasks = {}
    for name, model in models.items():
        tasks[name] = asyncio.create_task(
            model.agenerate(prompt, system_prompt=system_prompt)
        )

    # Wait for all tasks to complete
    results = {}
    for name, task in tasks.items():
        try:
            results[name] = await task
        except Exception as e:
            results[name] = f"Error: {str(e)}"

    return results

# Example usage
async def main():
    prompt = "Explain the concept of quantum entanglement in simple terms."
    system_prompt = "You are a physics teacher explaining concepts to high school students."

    print(f"Prompt: {prompt}")
    print("Comparing responses from different models...")
    print("-" * 50)

    results = await compare_models(prompt, system_prompt)

    for model_name, response in results.items():
        print(f"\n--- {model_name} ---")
        print(response[:300] + "..." if len(str(response)) > 300 else response)
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
```

## Fallback Chain Example

This example shows how to use model fallback chains for reliability:

```python
from lg_adk.models import ModelRegistry, get_model
import os

# Set up API keys (in a real application, these would be in environment variables)
os.environ["OPENAI_API_KEY"] = "your-openai-key"  # Replace with your actual key
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"  # Replace with your actual key

def create_reliable_agent():
    """Create an agent with a fallback chain of models."""
    # Get the model registry
    registry = ModelRegistry.get_instance()

    # Register a fallback chain named "reliable-chain"
    registry.register_fallback_chain(
        "reliable-chain",
        [
            "openai/gpt-4",  # Try OpenAI first
            "anthropic/claude-3-sonnet",  # If OpenAI fails, try Anthropic
            "ollama/llama3"  # Local fallback as last resort
        ]
    )

    # Use the fallback chain
    fallback_model = registry.get_model("reliable-chain")

    # Generate a response (will fall back if primary model fails)
    response = fallback_model.generate(
        "Explain why reliability is important in AI systems.",
        system_prompt="You are a helpful AI expert."
    )

    return response

# Example usage
if __name__ == "__main__":
    result = create_reliable_agent()
    print("Response from fallback chain:")
    print(result)
```

## Running the Examples

To run these examples, you'll need to:

1. Install LG-ADK with all optional dependencies:
   ```bash
   pip install "lg-adk[all]"
   ```

2. Set up API keys for the model providers you want to use:
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export GOOGLE_API_KEY="your-google-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   ```

3. For local models, install Ollama from https://ollama.ai/ and run:
   ```bash
   ollama pull llama3
   ```

4. Run any of the example scripts:
   ```bash
   python model_switching_agent.py
   ```

These examples demonstrate how LG-ADK's model provider system makes it easy to work with different LLM providers, switch between them, or create fallback chains for reliability.
