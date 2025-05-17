"""
Basic RAG (Retrieval-Augmented Generation) Example with LG-ADK

This example demonstrates how to build a simple RAG system that:
1. Processes a user query
2. Retrieves relevant documents from a vector store
3. Generates a response based on the retrieved context
"""

import os
from typing import Any, Dict, List

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lg_adk import Agent, GraphBuilder
from lg_adk.memory import MemoryManager

# --- 1. Create sample documents ---
# Create a sample text file for our knowledge base
sample_text = """
# Artificial Intelligence

Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence displayed by humans or animals.
AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.

The term "artificial intelligence" was first used by John McCarthy in 1956. The field has gone through multiple cycles of optimism followed by disappointment and loss of funding, followed by new approaches and renewed optimism.

## Machine Learning

Machine learning (ML) is a subset of AI that focuses on the development of algorithms that can access data and use it to learn for themselves.
The primary goal is to allow computers to learn automatically without human intervention.

### Types of Machine Learning:
- Supervised Learning: The algorithm is trained on labeled data.
- Unsupervised Learning: The algorithm finds patterns in unlabeled data.
- Reinforcement Learning: The algorithm learns through trial and error.

## Natural Language Processing

Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language.
NLP is used in many applications including:
- Voice assistants like Siri and Alexa
- Translation services like Google Translate
- Customer service chatbots

## Deep Learning

Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers.
These neural networks attempt to simulate the behavior of the human brain to solve complex problems.
"""

# Create a documents directory if it doesn't exist
os.makedirs("documents", exist_ok=True)

# Write the sample text to a file
with open("documents/ai_overview.txt", "w") as f:
    f.write(sample_text)

# --- 2. Create the vector store ---
# Load the document
documents = TextLoader("documents/ai_overview.txt").load()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

# Create embeddings
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a vector store
vector_store = Chroma.from_documents(documents=splits, embedding=embedding_function)

# --- 3. Define the RAG agents ---
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


# --- 4. Define the RAG workflow functions ---
def process_query(state: object) -> dict:
    """Process the user query for retrieval."""
    user_input = getattr(state, "input", "")

    # Use the query processor agent to reformulate the query
    result = query_processor.run({"input": user_input})
    processed_query = result.get("output", user_input)

    return {
        **state.__dict__,
        "original_query": user_input,
        "processed_query": processed_query,
    }


def retrieve_context(state: object) -> dict:
    """Retrieve relevant context from the vector store."""
    processed_query = getattr(state, "processed_query", "")

    # Search the vector store
    docs = vector_store.similarity_search(processed_query, k=3)
    context = [doc.page_content for doc in docs]

    return {
        **state.__dict__,
        "context": context,
    }


def generate_response(state: object) -> dict:
    """Generate a response based on the query and retrieved context."""
    original_query = getattr(state, "original_query", "")
    context = getattr(state, "context", [])

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
    response = result.get("output", "")

    return {
        **state.__dict__,
        "output": response,
    }


# --- 5. Build the RAG graph ---
builder = GraphBuilder()

# Create a memory manager for session persistence
memory_manager = MemoryManager()
builder.add_memory(memory_manager)

# Build the graph with the defined workflow
flow = [
    (None, "process_query"),
    ("process_query", "retrieve_context"),
    ("retrieve_context", "generate_response"),
    ("generate_response", None),
]

# Add the nodes
builder.add_node("process_query", process_query)
builder.add_node("retrieve_context", retrieve_context)
builder.add_node("generate_response", generate_response)

# Build the graph with the flow
builder.add_edge(None, "process_query")
builder.add_edge("process_query", "retrieve_context")
builder.add_edge("retrieve_context", "generate_response")
builder.add_edge("generate_response", None)
graph = builder.build()

# --- 6. Run the RAG system ---
if __name__ == "__main__":
    print("RAG Example")
    print("===========")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break

        # Run the RAG graph
        result = graph.invoke({"input": user_input})

        # Print the response
        print(f"\nRAG System: {result.get('output', '')}\n")
