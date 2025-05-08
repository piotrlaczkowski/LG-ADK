"""
Self-Correcting RAG Example with LG-ADK

This example demonstrates how to build a self-correcting RAG system that:
1. Processes a user query
2. Retrieves relevant documents from a vector store
3. Generates a preliminary response
4. Evaluates and critiques the response
5. Corrects and improves the response if needed
"""

import os
from typing import Any, Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from lg_adk import Agent, GraphBuilder
from lg_adk.memory import MemoryManager

# --- 1. Create sample documents ---
# Create a sample text file for our knowledge base
sample_text = """
# Quantum Computing

Quantum computing is a type of computing that uses quantum mechanics to process information. While traditional computers use bits (0s and 1s), quantum computers use quantum bits or qubits. The fundamental difference is that qubits can exist in multiple states simultaneously due to superposition.

## Key Concepts

### Superposition
Superposition allows qubits to exist in multiple states at once. While a classical bit can be either 0 or 1, a qubit can be in a state that is a combination of both 0 and 1 at the same time.

### Entanglement
Quantum entanglement is a phenomenon where pairs or groups of particles are generated or interact in ways such that the quantum state of each particle cannot be described independently of the others. Measuring one particle instantly influences the state of the entangled particle, regardless of the distance between them.

### Quantum Gates
Quantum gates are the building blocks of quantum circuits, similar to logic gates in classical computing. They manipulate the state of qubits according to the principles of quantum mechanics.

## Applications

### Cryptography
Quantum computers could break many of the encryption systems currently in use, as they can efficiently solve certain mathematical problems that classical computers cannot.

### Drug Discovery
Quantum computers can simulate molecular interactions more accurately than classical computers, potentially revolutionizing drug discovery and material science.

### Optimization Problems
Quantum computing can efficiently solve complex optimization problems in logistics, finance, and artificial intelligence.

## Current State

Despite significant progress, quantum computers are still in their early stages. They face challenges such as error correction, maintaining quantum coherence, and scaling to a sufficient number of qubits for practical applications.
"""

# Create a documents directory if it doesn't exist
os.makedirs("documents", exist_ok=True)

# Write the sample text to a file
with open("documents/quantum_computing.txt", "w") as f:
    f.write(sample_text)

# --- 2. Create the vector store ---
# Load the document
documents = TextLoader("documents/quantum_computing.txt").load()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

# Create embeddings
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a vector store
vector_store = Chroma.from_documents(documents=splits, embedding=embedding_function)

# --- 3. Define the agents ---
# Query processing agent
query_processor = Agent(
    name="query_processor",
    llm="ollama/llama3",
    description="Processes and reformulates user queries for optimal retrieval",
    system_message="""You are a query processing specialist. Your job is to:
    1. Understand the user's query
    2. Reformulate it to make it more effective for retrieval
    3. Extract key terms and concepts

    Output only the reformulated query without any explanations or additional text.
    """,
)

# Response generation agent
response_generator = Agent(
    name="response_generator",
    llm="ollama/llama3",
    description="Generates responses based on retrieved context and user query",
    system_message="""You are a response generator. Your job is to:
    1. Read the retrieved context carefully
    2. Understand the user's original question
    3. Generate a comprehensive and accurate response based on the context
    4. If the context doesn't contain relevant information, acknowledge the limitations

    Always base your answers on the provided context only.
    """,
)

# Response critic agent
response_critic = Agent(
    name="response_critic",
    llm="ollama/llama3",
    description="Evaluates and critiques responses for accuracy and completeness",
    system_message="""You are a response critic. Your job is to:
    1. Carefully review the generated response
    2. Compare it with the retrieved context
    3. Identify any factual errors, inconsistencies, or missing information
    4. Provide specific critiques and suggestions for improvement

    Output format:
    {
        "score": <rating from 0-10>,
        "critiques": ["critique 1", "critique 2", ...],
        "missing_information": ["missing info 1", "missing info 2", ...],
        "factual_errors": ["error 1", "error 2", ...],
        "needs_correction": <true/false>
    }
    """,
)

# Response corrector agent
response_corrector = Agent(
    name="response_corrector",
    llm="ollama/llama3",
    description="Corrects and improves responses based on critique",
    system_message="""You are a response corrector. Your job is to:
    1. Review the original response
    2. Consider the critique carefully
    3. Rewrite and improve the response to address all identified issues
    4. Ensure the corrected response is accurate, complete, and well-structured

    Output only the corrected response without any explanations or additional text.
    """,
)


# --- 4. Define the workflow functions ---
def process_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process the user query for retrieval."""
    user_input = state.get("input", "")

    # Use the query processor agent to reformulate the query
    result = query_processor.run({"input": user_input})
    processed_query = result.get("output", user_input)

    return {
        **state,
        "original_query": user_input,
        "processed_query": processed_query,
    }


def retrieve_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve relevant context from the vector store."""
    processed_query = state.get("processed_query", "")

    # Search the vector store
    docs = vector_store.similarity_search(processed_query, k=3)
    context = [doc.page_content for doc in docs]

    return {
        **state,
        "context": context,
    }


def generate_initial_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate an initial response based on the query and retrieved context."""
    original_query = state.get("original_query", "")
    context = state.get("context", [])

    # Format the context
    formatted_context = "\n\n".join([f"Document chunk {i+1}:\n{chunk}" for i, chunk in enumerate(context)])

    # Generate response with the response agent
    prompt = f"""
    Context information:
    {formatted_context}

    User question: {original_query}

    Please answer the question based on the context provided.
    """

    result = response_generator.run({"input": prompt})
    initial_response = result.get("output", "")

    return {
        **state,
        "initial_response": initial_response,
    }


def critique_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Critique the initial response for accuracy and completeness."""
    original_query = state.get("original_query", "")
    context = state.get("context", [])
    initial_response = state.get("initial_response", "")

    # Format the context
    formatted_context = "\n\n".join([f"Document chunk {i+1}:\n{chunk}" for i, chunk in enumerate(context)])

    # Generate critique with the critic agent
    prompt = f"""
    Context information:
    {formatted_context}

    User question: {original_query}

    Generated response:
    {initial_response}

    Please evaluate this response for accuracy, completeness, and relevance.
    """

    result = response_critic.run({"input": prompt})
    critique = result.get("output", "")

    # Try to parse the critique to determine if correction is needed
    needs_correction = False
    try:
        import json

        critique_json = json.loads(critique.replace("```json", "").replace("```", "").strip())
        score = critique_json.get("score", 0)
        needs_correction = critique_json.get("needs_correction", score < 7)
    except:
        # If parsing fails, check for negative keywords
        negative_keywords = ["incorrect", "error", "missing", "improve", "incomplete", "wrong"]
        needs_correction = any(keyword in critique.lower() for keyword in negative_keywords)

    return {
        **state,
        "critique": critique,
        "needs_correction": needs_correction,
    }


def decide_path(state: Dict[str, Any]) -> str:
    """Decide whether to correct the response or return it as is."""
    needs_correction = state.get("needs_correction", False)
    return "correct_response" if needs_correction else "finalize_response"


def correct_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Correct and improve the response based on critique."""
    original_query = state.get("original_query", "")
    initial_response = state.get("initial_response", "")
    critique = state.get("critique", "")
    context = state.get("context", [])

    # Format the context
    formatted_context = "\n\n".join([f"Document chunk {i+1}:\n{chunk}" for i, chunk in enumerate(context)])

    # Generate corrected response with the corrector agent
    prompt = f"""
    Context information:
    {formatted_context}

    User question: {original_query}

    Initial response:
    {initial_response}

    Critique:
    {critique}

    Please correct and improve the response based on the critique.
    """

    result = response_corrector.run({"input": prompt})
    corrected_response = result.get("output", "")

    return {
        **state,
        "corrected_response": corrected_response,
    }


def finalize_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Finalize the response."""
    needs_correction = state.get("needs_correction", False)
    initial_response = state.get("initial_response", "")
    corrected_response = state.get("corrected_response", "")

    # Use the corrected response if available, otherwise use the initial response
    final_response = corrected_response if needs_correction else initial_response

    return {
        **state,
        "output": final_response,
    }


# --- 5. Build the self-correcting RAG graph ---
builder = GraphBuilder()

# Create a memory manager for session persistence
memory_manager = MemoryManager()
builder.add_memory(memory_manager)

# Add the nodes
builder.add_node("process_query", process_query)
builder.add_node("retrieve_context", retrieve_context)
builder.add_node("generate_initial_response", generate_initial_response)
builder.add_node("critique_response", critique_response)
builder.add_node("correct_response", correct_response)
builder.add_node("finalize_response", finalize_response)

# Build the graph with conditional flow
flow = [
    (None, "process_query"),
    ("process_query", "retrieve_context"),
    ("retrieve_context", "generate_initial_response"),
    ("generate_initial_response", "critique_response"),
    ("critique_response", decide_path),
    ("correct_response", "finalize_response"),
    ("finalize_response", None),
]

# Build the graph
graph = builder.build(flow=flow)

# --- 6. Run the self-correcting RAG system ---
if __name__ == "__main__":
    print("Self-Correcting RAG Example")
    print("===========================")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break

        # Run the self-correcting RAG graph
        result = graph.invoke({"input": user_input})

        # Print the response
        print(f"\nRAG System: {result.get('output', '')}\n")

        # Optionally show the critique and correction process
        if result.get("needs_correction", False):
            print("\n--- Debugging Information ---")
            print("Initial response was critiqued and corrected.")
            print(f"Critique: {result.get('critique', 'No critique available')}")
            print("----------------------------\n")
