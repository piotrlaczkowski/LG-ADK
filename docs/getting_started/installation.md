# 🛠️ Installation

LG-ADK can be installed using pip or Poetry.

## ⚡ Prerequisites

- 🐍 Python 3.11 or higher
- 🦙 (Optional) Ollama for local model support
- 🤖 (Optional) Access to Google AI services for Gemini models

## 📦 Using pip

!!! tip "Quick install with pip"
    ```bash
    pip install lg-adk
    ```

## 📦 Using Poetry

!!! tip "Install with Poetry"
    ```bash
    poetry add lg-adk
    ```

## 👩‍💻 Development Installation

If you want to contribute to LG-ADK, you can install it in development mode:

```bash
# Clone the repository
git clone https://github.com/yourusername/lg-adk.git
cd lg-adk

# Install dependencies
poetry install

# Alternatively with pip
pip install -e ".[dev]"
```

## ⚙️ Setting Up Environment Variables

LG-ADK uses environment variables for configuration. You can create a `.env` file in your project directory:

```bash
# .env file
OPENAI_API_KEY=your_openai_api_key  # If using OpenAI models
GOOGLE_API_KEY=your_google_api_key  # If using Gemini models
OLLAMA_BASE_URL=http://localhost:11434  # For local Ollama models
DEFAULT_LLM=ollama/llama3  # Default model to use
```

## ✅ Verifying Installation

You can verify your installation by running a simple example:

```python
from lg_adk import Agent

# This should print the version number
print(f"LG-ADK version: {lg_adk.__version__}")

# Create a simple agent
agent = Agent(
    name="test_agent",
    llm="ollama/llama3",
    description="Test agent"
)
```

## 🛑 Troubleshooting

!!! warning "Common Issues"
    - **Import Errors**: Make sure you have installed all the required dependencies.
    - **Model Connection Errors**:
      - For Ollama: Ensure Ollama is running locally (`ollama serve`).
      - For Gemini/OpenAI: Check your API keys are set correctly.

### 💡 Getting Help

If you encounter issues, please:

1. Check the [Troubleshooting Guide](../guides/troubleshooting.md)
2. Search existing [GitHub Issues](https://github.com/piotrlaczkowski/lg-adk/issues)
3. Open a new issue if needed
