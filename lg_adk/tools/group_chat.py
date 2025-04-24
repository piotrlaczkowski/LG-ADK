"""
Group Chat Tool for LG-ADK

This module provides a tool for enabling multi-agent conversations:
1. GroupChatTool: For facilitating conversations between multiple agents
"""

from typing import Dict, List, Any, Optional, Callable, Union
import time
import uuid
from pydantic import BaseModel, Field

from lg_adk import Agent
from lg_adk.memory import MemoryManager
from lg_adk.utils.logging import get_logger

logger = get_logger(__name__)


class Message(BaseModel):
    """Message in a group chat."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    content: str
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GroupChat(BaseModel):
    """Group chat session."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    agents: List[str]
    messages: List[Message] = Field(default_factory=list)
    created_at: int = Field(default_factory=lambda: int(time.time()))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GroupChatTool:
    """Tool for facilitating conversations between multiple agents.
    
    This tool allows:
    1. Creating group chats with multiple agents
    2. Sending messages from one agent to the group
    3. Managing conversation history
    4. Implementing different conversation patterns
    """
    
    def __init__(self, agent_registry: Dict[str, Agent] = None, 
                memory_manager: Optional[MemoryManager] = None):
        """Initialize the group chat tool.
        
        Args:
            agent_registry: A dictionary mapping agent names to Agent instances
            memory_manager: Optional memory manager for persistence
        """
        self.agent_registry = agent_registry or {}
        self.memory_manager = memory_manager
        self.chats: Dict[str, GroupChat] = {}
        
    def register_agent(self, name: str, agent: Agent) -> None:
        """Register an agent for group chat.
        
        Args:
            name: A unique name for the agent
            agent: The Agent instance
        """
        self.agent_registry[name] = agent
    
    def create_chat(self, name: str, agent_ids: List[str], 
                   metadata: Dict[str, Any] = None) -> str:
        """Create a new group chat.
        
        Args:
            name: Name of the chat
            agent_ids: List of agent IDs to include
            metadata: Additional metadata for the chat
            
        Returns:
            The ID of the created chat
            
        Raises:
            KeyError: If any agent ID is not found
        """
        # Validate all agent IDs
        for agent_id in agent_ids:
            if agent_id not in self.agent_registry:
                raise KeyError(f"Agent '{agent_id}' not found in registry")
        
        # Create the chat
        chat = GroupChat(
            name=name,
            agents=agent_ids,
            metadata=metadata or {}
        )
        
        # Store the chat
        self.chats[chat.id] = chat
        
        return chat.id
    
    def send_message(self, chat_id: str, agent_id: str, 
                    content: str, metadata: Dict[str, Any] = None) -> Message:
        """Send a message to a group chat.
        
        Args:
            chat_id: The ID of the chat
            agent_id: The ID of the sending agent
            content: The message content
            metadata: Additional metadata for the message
            
        Returns:
            The created message
            
        Raises:
            KeyError: If the chat or agent is not found
            ValueError: If the agent is not in the chat
        """
        if chat_id not in self.chats:
            raise KeyError(f"Chat '{chat_id}' not found")
        
        chat = self.chats[chat_id]
        
        if agent_id not in self.agent_registry:
            raise KeyError(f"Agent '{agent_id}' not found in registry")
        
        if agent_id not in chat.agents:
            raise ValueError(f"Agent '{agent_id}' is not in chat '{chat_id}'")
        
        # Create the message
        message = Message(
            agent_id=agent_id,
            content=content,
            metadata=metadata or {}
        )
        
        # Add the message to the chat
        chat.messages.append(message)
        
        return message
    
    def get_chat_history(self, chat_id: str, limit: int = None) -> List[Message]:
        """Get the history of a group chat.
        
        Args:
            chat_id: The ID of the chat
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of messages in the chat
            
        Raises:
            KeyError: If the chat is not found
        """
        if chat_id not in self.chats:
            raise KeyError(f"Chat '{chat_id}' not found")
        
        chat = self.chats[chat_id]
        
        if limit is not None:
            return chat.messages[-limit:]
        
        return chat.messages
    
    def get_chat(self, chat_id: str) -> GroupChat:
        """Get a group chat.
        
        Args:
            chat_id: The ID of the chat
            
        Returns:
            The group chat
            
        Raises:
            KeyError: If the chat is not found
        """
        if chat_id not in self.chats:
            raise KeyError(f"Chat '{chat_id}' not found")
        
        return self.chats[chat_id]
    
    def run_conversation(self, chat_id: str, initial_prompt: str, 
                       max_turns: int = 5, 
                       speaker_selection: Callable[[GroupChat, List[Message]], str] = None) -> List[Message]:
        """Run a conversation between agents in a chat.
        
        Args:
            chat_id: The ID of the chat
            initial_prompt: The initial prompt to start the conversation
            max_turns: Maximum number of conversation turns
            speaker_selection: Function to select the next speaker
            
        Returns:
            The messages generated in the conversation
            
        Raises:
            KeyError: If the chat is not found
        """
        chat = self.get_chat(chat_id)
        
        # Default speaker selection: round-robin
        if speaker_selection is None:
            def round_robin(chat: GroupChat, history: List[Message]) -> str:
                if not history:
                    return chat.agents[0]
                last_speaker_idx = chat.agents.index(history[-1].agent_id)
                next_speaker_idx = (last_speaker_idx + 1) % len(chat.agents)
                return chat.agents[next_speaker_idx]
            
            speaker_selection = round_robin
        
        # Start with the initial prompt (from the first agent)
        current_messages = []
        first_agent_id = speaker_selection(chat, [])
        first_message = self.send_message(chat_id, first_agent_id, initial_prompt)
        current_messages.append(first_message)
        
        # Run the conversation
        for _ in range(max_turns):
            # Get the conversation history
            history = self.get_chat_history(chat_id)
            
            # Select the next speaker
            next_agent_id = speaker_selection(chat, history)
            agent = self.agent_registry[next_agent_id]
            
            # Format conversation history for the agent
            formatted_history = [
                {"role": "user" if msg.agent_id != next_agent_id else "assistant", 
                 "content": msg.content}
                for msg in history
            ]
            
            # Run the agent to get the response
            result = agent.run({
                "input": initial_prompt,
                "conversation_history": formatted_history
            })
            
            # Extract the response
            response = result.get("output", "")
            
            # Send the message
            message = self.send_message(chat_id, next_agent_id, response)
            current_messages.append(message)
        
        return current_messages 