"""
Tests for RAG with memory components in LG-ADK.
"""

import unittest
from unittest.mock import MagicMock, patch
import uuid

import pytest

from lg_adk import Agent, GraphBuilder
from lg_adk.memory import MemoryManager
from lg_adk.sessions import SessionManager


@pytest.fixture
def mock_memory_manager():
    """Create a mock memory manager."""
    memory_manager = MagicMock(spec=MemoryManager)
    memory_manager.get_conversation_history.return_value = [
        {"role": "user", "content": "What are the causes of climate change?"},
        {"role": "assistant", "content": "Climate change is primarily caused by greenhouse gases."}
    ]
    return memory_manager


@pytest.fixture
def mock_session_manager():
    """Create a mock session manager."""
    session_manager = MagicMock(spec=SessionManager)
    session_manager.get_session.return_value = {"user_id": "test_user"}
    return session_manager


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    vector_store = MagicMock()
    vector_store.similarity_search.return_value = [
        MagicMock(page_content="Document about climate change causes."),
        MagicMock(page_content="Document about greenhouse gases.")
    ]
    return vector_store


class TestRAGWithMemory(unittest.TestCase):
    """Tests for RAG with memory functionality."""
    
    def test_get_or_create_session(self):
        """Test that get_or_create_session works correctly."""
        from docs.examples.rag_with_memory import get_or_create_session
        
        # Mock SessionManager
        session_manager = MagicMock(spec=SessionManager)
        session_manager.get_session.return_value = {"user_id": "test_user"}
        
        # Patch the session manager
        with patch("docs.examples.rag_with_memory.session_manager", session_manager):
            # Test with existing session ID
            state = {"session_id": "existing_id"}
            result = get_or_create_session(state)
            
            # Verify results
            session_manager.create_session.assert_not_called()
            session_manager.get_session.assert_called_once_with("existing_id")
            self.assertEqual(result["session_id"], "existing_id")
            self.assertEqual(result["session_data"], {"user_id": "test_user"})
            
            # Reset mock
            session_manager.reset_mock()
            
            # Test without session ID
            with patch("docs.examples.rag_with_memory.uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")):
                state = {}
                result = get_or_create_session(state)
                
                # Verify results
                session_manager.create_session.assert_called_once_with("12345678-1234-5678-1234-567812345678")
                session_manager.get_session.assert_called_once_with("12345678-1234-5678-1234-567812345678")
                self.assertEqual(result["session_id"], "12345678-1234-5678-1234-567812345678")
                self.assertEqual(result["session_data"], {"user_id": "test_user"})
    
    def test_retrieve_history(self):
        """Test that retrieve_history retrieves conversation history."""
        from docs.examples.rag_with_memory import retrieve_history
        
        # Mock MemoryManager
        memory_manager = MagicMock(spec=MemoryManager)
        memory_manager.get_conversation_history.return_value = [
            {"role": "user", "content": "What are the causes of climate change?"},
            {"role": "assistant", "content": "Climate change is primarily caused by greenhouse gases."}
        ]
        
        # Patch the memory manager
        with patch("docs.examples.rag_with_memory.memory_manager", memory_manager):
            # Test with session ID
            state = {"session_id": "test_session"}
            result = retrieve_history(state)
            
            # Verify results
            memory_manager.get_conversation_history.assert_called_once_with("test_session")
            self.assertEqual(len(result["conversation_history"]), 2)
            self.assertEqual(result["conversation_history"][0]["role"], "user")
            self.assertEqual(result["conversation_history"][1]["role"], "assistant")
    
    def test_enhance_query(self):
        """Test that enhance_query uses conversation context to improve queries."""
        from docs.examples.rag_with_memory import enhance_query
        
        # Mock context_enhancer Agent
        context_enhancer = MagicMock(spec=Agent)
        context_enhancer.run.return_value = {"output": "enhanced query about climate change"}
        
        # Patch the context enhancer
        with patch("docs.examples.rag_with_memory.context_enhancer", context_enhancer):
            # Test with conversation history
            state = {
                "input": "Tell me more",
                "conversation_history": [
                    {"role": "user", "content": "What are the causes of climate change?"},
                    {"role": "assistant", "content": "Climate change is primarily caused by greenhouse gases."}
                ]
            }
            result = enhance_query(state)
            
            # Verify results
            context_enhancer.run.assert_called_once()
            self.assertEqual(result["original_query"], "Tell me more")
            self.assertEqual(result["enhanced_query"], "enhanced query about climate change")
            
            # Test without conversation history
            context_enhancer.reset_mock()
            state = {"input": "Tell me more", "conversation_history": []}
            result = enhance_query(state)
            
            # Verify results for empty history
            context_enhancer.run.assert_not_called()
            self.assertEqual(result["enhanced_query"], "Tell me more")
    
    def test_retrieve_context(self):
        """Test that retrieve_context retrieves relevant documents."""
        from docs.examples.rag_with_memory import retrieve_context
        
        # Mock vector store
        vector_store = MagicMock()
        vector_store.similarity_search.return_value = [
            MagicMock(page_content="Document about climate change causes."),
            MagicMock(page_content="Document about greenhouse gases.")
        ]
        
        # Patch the vector store
        with patch("docs.examples.rag_with_memory.vector_store", vector_store):
            # Test with an enhanced query
            state = {"enhanced_query": "climate change causes"}
            result = retrieve_context(state)
            
            # Verify results
            vector_store.similarity_search.assert_called_once_with("climate change causes", k=3)
            self.assertEqual(len(result["context"]), 2)
            self.assertEqual(result["context"][0], "Document about climate change causes.")
    
    def test_generate_response(self):
        """Test that generate_response uses context and history to generate a response."""
        from docs.examples.rag_with_memory import generate_response
        
        # Mock response_generator Agent
        response_generator = MagicMock(spec=Agent)
        response_generator.run.return_value = {"output": "Climate change is caused by various factors."}
        
        # Patch the response generator
        with patch("docs.examples.rag_with_memory.response_generator", response_generator):
            # Test with context and history
            state = {
                "original_query": "Tell me more about climate change causes",
                "context": ["Document about climate change causes.", "Document about greenhouse gases."],
                "conversation_history": [
                    {"role": "user", "content": "What is climate change?"},
                    {"role": "assistant", "content": "Climate change refers to long-term shifts in temperatures."}
                ]
            }
            result = generate_response(state)
            
            # Verify results
            response_generator.run.assert_called_once()
            self.assertEqual(result["output"], "Climate change is caused by various factors.")
    
    def test_update_memory(self):
        """Test that update_memory adds messages to memory manager."""
        from docs.examples.rag_with_memory import update_memory
        
        # Mock MemoryManager
        memory_manager = MagicMock(spec=MemoryManager)
        
        # Patch the memory manager
        with patch("docs.examples.rag_with_memory.memory_manager", memory_manager):
            # Test with session ID and messages
            state = {
                "session_id": "test_session",
                "original_query": "What causes climate change?",
                "output": "Climate change is caused by greenhouse gases."
            }
            update_memory(state)
            
            # Verify results
            self.assertEqual(memory_manager.add_message.call_count, 2)
            memory_manager.add_message.assert_any_call(
                "test_session",
                {"role": "user", "content": "What causes climate change?"}
            )
            memory_manager.add_message.assert_any_call(
                "test_session",
                {"role": "assistant", "content": "Climate change is caused by greenhouse gases."}
            )
    
    @patch("docs.examples.rag_with_memory.get_or_create_session")
    @patch("docs.examples.rag_with_memory.retrieve_history")
    @patch("docs.examples.rag_with_memory.enhance_query")
    @patch("docs.examples.rag_with_memory.retrieve_context")
    @patch("docs.examples.rag_with_memory.generate_response")
    @patch("docs.examples.rag_with_memory.update_memory")
    def test_rag_with_memory_flow(self, mock_update, mock_generate, mock_retrieve, 
                                 mock_enhance, mock_history, mock_session):
        """Test the flow of the RAG with memory graph."""
        # Setup mocks
        mock_session.return_value = {"session_id": "test_session", "session_data": {"user_id": "test_user"}}
        mock_history.return_value = {"conversation_history": [{"role": "user", "content": "Previous question"}]}
        mock_enhance.return_value = {"enhanced_query": "enhanced query", "original_query": "original query"}
        mock_retrieve.return_value = {"context": ["Document content"]}
        mock_generate.return_value = {"output": "Generated response"}
        mock_update.return_value = {"output": "Generated response"}
        
        # Create the graph builder
        builder = GraphBuilder()
        
        # Add the nodes
        builder.add_node("get_or_create_session", mock_session)
        builder.add_node("retrieve_history", mock_history)
        builder.add_node("enhance_query", mock_enhance)
        builder.add_node("retrieve_context", mock_retrieve)
        builder.add_node("generate_response", mock_generate)
        builder.add_node("update_memory", mock_update)
        
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
        
        # Test with a sample input
        result = graph.invoke({"input": "What causes climate change?"})
        
        # Verify that all nodes were called in order
        mock_session.assert_called_once()
        mock_history.assert_called_once()
        mock_enhance.assert_called_once()
        mock_retrieve.assert_called_once()
        mock_generate.assert_called_once()
        mock_update.assert_called_once()
        
        # Check the result
        self.assertEqual(result["output"], "Generated response")


if __name__ == "__main__":
    unittest.main() 