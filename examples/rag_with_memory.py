"""
RAG with Persistent Memory Example

This example demonstrates a RAG system with:
1. Persistent memory for tracking conversation history
2. Session management for multiple users
3. Query context enhancement using conversation history
"""

import os
import uuid
from typing import Dict, Any, List, Optional

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from lg_adk import Agent, GraphBuilder
from lg_adk.memory import MemoryManager
from lg_adk.sessions import SessionManager

# --- 1. Create sample documents ---
# Create a sample text file for our knowledge base
sample_text = """
# Climate Change

Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels, which produces heat-trapping gases.

## Causes

### Greenhouse Gases
Greenhouse gases act like a blanket around Earth, trapping the sun's heat. The main greenhouse gases include:
- Carbon dioxide (CO2): From burning fossil fuels and deforestation
- Methane (CH4): From livestock, agriculture, and fossil fuel extraction
- Nitrous oxide (N2O): From fertilizers and industrial processes
- Fluorinated gases: From industrial processes

### Deforestation
Trees absorb CO2, helping to regulate the climate. When forests are cut down, the carbon stored in the trees is released into the atmosphere, contributing to the greenhouse effect.

## Effects

### Rising Temperatures
Global temperatures have increased by about 1°C since pre-industrial times. This warming is causing:
- More frequent and intense heat waves
- Changes in precipitation patterns
- Melting ice and rising sea levels

### Extreme Weather Events
Climate change is making extreme weather events more frequent and severe, including:
- Hurricanes and cyclones
- Droughts and wildfires
- Floods and heavy rainfall

### Impacts on Ecosystems
Many plant and animal species are struggling to adapt to rapid climate change:
- Coral reef bleaching due to ocean warming
- Species migration to cooler regions
- Timing changes in seasonal activities

## Mitigation Strategies

### Renewable Energy
Transitioning from fossil fuels to renewable energy sources like solar, wind, and hydropower is essential for reducing greenhouse gas emissions.

### Energy Efficiency
Improving energy efficiency in buildings, transportation, and industry can significantly reduce energy consumption and emissions.

### Carbon Pricing
Implementing carbon taxes or cap-and-trade systems creates economic incentives for reducing emissions.

### Sustainable Land Use
Protecting and restoring forests, improving agricultural practices, and promoting sustainable land use can help absorb carbon and reduce emissions.
"""

# Create a documents directory if it doesn't exist
os.makedirs("documents", exist_ok=True)

# Write the sample text to a file
with open("documents/climate_change.txt", "w") as f:
    f.write(sample_text)

# --- 2. Create the vector store ---
# Load the document
documents = TextLoader("documents/climate_change.txt").load()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

# Create embeddings
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a vector store
vector_store = Chroma.from_documents(documents=splits, embedding=embedding_function)

# --- 3. Create memory and session managers ---
# Create a memory manager
memory_manager = MemoryManager(
    memory_type="in_memory",
    max_tokens=8000
)

# Create a session manager
session_manager = SessionManager()

# --- 4. Define the agents ---
# Context enhancement agent
context_enhancer = Agent(
    name="context_enhancer",
    llm="ollama/llama3",
    description="Enhances queries with conversation context",
    system_message="""You are a context enhancement specialist. Your job is to:
    1. Analyze the current user query
    2. Review the conversation history
    3. Create an enhanced query that includes relevant context from the history
    4. Ensure the enhanced query will retrieve the most relevant information
    
    Output only the enhanced query without any explanations or additional text.
    """
)

# Response generation agent
response_generator = Agent(
    name="response_generator",
    llm="ollama/llama3",
    description="Generates responses based on retrieved context and conversation history",
    system_message="""You are a knowledgeable assistant. Your job is to:
    1. Read the retrieved context carefully
    2. Review the conversation history to understand the context
    3. Answer the user's question comprehensively based on the retrieved information
    4. If the context doesn't contain relevant information, acknowledge the limitations
    5. Make references to previous parts of the conversation when appropriate
    
    Always base your answers on the provided context and conversation history.
    """
)

# --- 5. Define the workflow functions ---
def get_or_create_session(state: Dict[str, Any]) -> Dict[str, Any]:
    """Get an existing session or create a new one."""
    session_id = state.get("session_id")
    
    # If no session ID was provided, create a new one
    if not session_id:
        session_id = str(uuid.uuid4())
        session_manager.create_session(session_id)
    
    # Get the session data
    session_data = session_manager.get_session(session_id)
    
    return {
        **state,
        "session_id": session_id,
        "session_data": session_data,
    }

def retrieve_history(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve conversation history for the current session."""
    session_id = state.get("session_id")
    
    # Get conversation history
    conversation_history = memory_manager.get_conversation_history(session_id)
    
    return {
        **state,
        "conversation_history": conversation_history,
    }

def enhance_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance the query with conversation context."""
    user_input = state.get("input", "")
    conversation_history = state.get("conversation_history", [])
    
    # If no conversation history, use the original query
    if not conversation_history:
        return {
            **state,
            "enhanced_query": user_input,
        }
    
    # Format the conversation history
    formatted_history = "\n".join([
        f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
        for msg in conversation_history
    ])
    
    # Use the context enhancer agent to create an enhanced query
    prompt = f"""
    User's current question: {user_input}
    
    Conversation history:
    {formatted_history}
    
    Based on the conversation history and the current question, 
    create an enhanced query that will help retrieve the most relevant information.
    """
    
    result = context_enhancer.run({"input": prompt})
    enhanced_query = result.get("output", user_input)
    
    return {
        **state,
        "original_query": user_input,
        "enhanced_query": enhanced_query,
    }

def retrieve_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve relevant context from the vector store."""
    enhanced_query = state.get("enhanced_query", "")
    
    # Search the vector store
    docs = vector_store.similarity_search(enhanced_query, k=3)
    context = [doc.page_content for doc in docs]
    
    return {
        **state,
        "context": context,
    }

def generate_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a response based on context and conversation history."""
    original_query = state.get("original_query", "")
    context = state.get("context", [])
    conversation_history = state.get("conversation_history", [])
    
    # Format the context
    formatted_context = "\n\n".join([f"Document chunk {i+1}:\n{chunk}" for i, chunk in enumerate(context)])
    
    # Format the conversation history
    formatted_history = "\n".join([
        f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
        for msg in conversation_history
    ])
    
    # Generate response with the response agent
    prompt = f"""
    Context information:
    {formatted_context}
    
    Conversation history:
    {formatted_history}
    
    User question: {original_query}
    
    Please answer the question based on the context provided, taking the conversation history into account.
    """
    
    result = response_generator.run({"input": prompt})
    response = result.get("output", "")
    
    return {
        **state,
        "output": response,
    }

def update_memory(state: Dict[str, Any]) -> Dict[str, Any]:
    """Update the memory with the latest exchange."""
    session_id = state.get("session_id", "")
    original_query = state.get("original_query", "")
    response = state.get("output", "")
    
    # Add user message to memory
    memory_manager.add_message(
        session_id,
        {"role": "user", "content": original_query}
    )
    
    # Add assistant response to memory
    memory_manager.add_message(
        session_id,
        {"role": "assistant", "content": response}
    )
    
    return state

# --- 6. Build the RAG with memory graph ---
builder = GraphBuilder()

# Add the memory manager
builder.add_memory(memory_manager)

# Add the nodes
builder.add_node("get_or_create_session", get_or_create_session)
builder.add_node("retrieve_history", retrieve_history)
builder.add_node("enhance_query", enhance_query)
builder.add_node("retrieve_context", retrieve_context)
builder.add_node("generate_response", generate_response)
builder.add_node("update_memory", update_memory)

# Define the flow
flow = [
    (None, "get_or_create_session"),
    ("get_or_create_session", "retrieve_history"),
    ("retrieve_history", "enhance_query"),
    ("enhance_query", "retrieve_context"),
    ("retrieve_context", "generate_response"),
    ("generate_response", "update_memory"),
    ("update_memory", None)
]

# Build the graph
graph = builder.build(flow=flow)

# --- 7. Run the RAG with memory system ---
if __name__ == "__main__":
    print("RAG with Memory Example")
    print("======================")
    print("Type 'exit' to quit, or 'new session' to start a new conversation.\n")
    
    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit", "q"]:
            break
        
        if user_input.lower() == "new session":
            session_id = str(uuid.uuid4())
            print(f"\nStarting new session with ID: {session_id}\n")
            continue
        
        # Run the RAG with memory graph
        result = graph.invoke({
            "input": user_input,
            "session_id": session_id
        })
        
        # Print the response
        print(f"\nRAG System: {result.get('output', '')}\n")
        
        # Optionally show the enhanced query
        # print(f"Enhanced query: {result.get('enhanced_query', '')}") 