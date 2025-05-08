"""Group chat tools for LG-ADK agents."""

import time
import uuid
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from lg_adk import Agent
from lg_adk.utils.logging import get_logger

logger = get_logger(__name__)


class Message(BaseModel):
    """Message in a group chat."""

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    content: str
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    metadata: dict[str, Any] = Field(default_factory=dict)


class GroupChat(BaseModel):
    """Group chat session."""

    chat_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    agents: list[str]
    messages: list[Message] = Field(default_factory=list)
    created_at: int = Field(default_factory=lambda: int(time.time()))
    metadata: dict[str, Any] = Field(default_factory=dict)


class GroupChatTool(BaseModel):
    """Tool for managing group chat sessions.

    Attributes:
        chat_id: Unique identifier for the group chat.
        participants: List of participant user IDs.
        messages: List of messages in the chat.
    """

    chat_id: str = Field(..., description="Unique identifier for the group chat.")
    participants: list[str] = Field(default_factory=list, description="List of participant user IDs.")
    messages: list[dict[str, Any]] = Field(default_factory=list, description="List of messages in the chat.")

    def add_participant(self, user_id: str) -> None:
        """Add a participant to the group chat.

        Args:
            user_id: The user ID to add.
        """
        if user_id not in self.participants:
            self.participants.append(user_id)

    def remove_participant(self, user_id: str) -> None:
        """Remove a participant from the group chat.

        Args:
            user_id: The user ID to remove.
        """
        if user_id in self.participants:
            self.participants.remove(user_id)

    def add_message(self, sender_id: str, content: str) -> None:
        """Add a message to the group chat.

        Args:
            sender_id: The ID of the sender.
            content: The message content.
        """
        self.messages.append({"sender_id": sender_id, "content": content})

    def get_messages(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get messages from the group chat.

        Args:
            limit: The maximum number of messages to return.

        Returns:
            A list of messages.
        """
        return self.messages[-limit:] if limit else self.messages

    def __init__(self, **data: Any) -> None:
        """Initialize GroupChatTool.

        Args:
            **data: Arbitrary keyword arguments for initialization.
        """
        super().__init__(**data)
        self.chats: dict[str, GroupChat] = {}

    def register_agent(self, name: str, _agent: Agent) -> None:
        """Register an agent for group chat.

        Args:
            name: A unique name for the agent.
            agent: The Agent instance.
        """
        if name not in self.participants:
            self.participants.append(name)

    def create_chat(self, name: str, agent_ids: list[str], metadata: dict[str, Any] = None) -> str:
        """Create a new group chat.

        Args:
            name: Name of the chat.
            agent_ids: List of agent IDs to include.
            metadata: Additional metadata for the chat.

        Returns:
            The ID of the created chat.

        Raises:
            KeyError: If any agent ID is not found.
        """
        for agent_id in agent_ids:
            if agent_id not in self.participants:
                raise KeyError(f"Agent '{agent_id}' not found in registry")
        chat = GroupChat(
            name=name,
            agents=agent_ids,
            metadata=metadata or {},
        )
        self.chats[chat.chat_id] = chat
        return chat.chat_id

    def send_message(
        self,
        chat_id: str,
        agent_id: str,
        content: str,
        metadata: dict[str, Any] = None,
    ) -> Message:
        """Send a message to a group chat.

        Args:
            chat_id: The ID of the chat.
            agent_id: The ID of the sending agent.
            content: The message content.
            metadata: Additional metadata for the message.

        Returns:
            The created message.

        Raises:
            KeyError: If the chat or agent is not found.
            ValueError: If the agent is not in the chat.
        """
        if chat_id not in self.chats:
            raise KeyError(f"Chat '{chat_id}' not found")
        chat = self.chats[chat_id]
        if agent_id not in self.participants:
            raise KeyError(f"Agent '{agent_id}' not found in registry")
        if agent_id not in chat.agents:
            raise ValueError(f"Agent '{agent_id}' is not in chat '{chat_id}'")
        message = Message(
            agent_id=agent_id,
            content=content,
            metadata=metadata or {},
        )
        chat.messages.append(message)
        return message

    def get_chat_history(self, chat_id: str, limit: int = None) -> list[Message]:
        """Get the history of a group chat.

        Args:
            chat_id: The ID of the chat.
            limit: Maximum number of messages to retrieve.

        Returns:
            List of messages in the chat.

        Raises:
            KeyError: If the chat is not found.
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
            chat_id: The ID of the chat.

        Returns:
            The group chat.

        Raises:
            KeyError: If the chat is not found.
        """
        if chat_id not in self.chats:
            raise KeyError(f"Chat '{chat_id}' not found")
        return self.chats[chat_id]

    def run_conversation(
        self,
        chat_id: str,
        initial_prompt: str,
        max_turns: int = 5,
        speaker_selection: Callable[[GroupChat, list[Message]], str] = None,
    ) -> list[Message]:
        """Run a conversation between agents in a chat.

        Args:
            chat_id: The ID of the chat.
            initial_prompt: The initial prompt to start the conversation.
            max_turns: Maximum number of conversation turns.
            speaker_selection: Function to select the next speaker.

        Returns:
            The messages generated in the conversation.

        Raises:
            KeyError: If the chat is not found.
        """
        chat = self.get_chat(chat_id)
        if speaker_selection is None:

            def round_robin(chat: GroupChat, history: list[Message]) -> str:
                if not history:
                    return chat.agents[0]
                last_speaker_idx = chat.agents.index(history[-1].agent_id)
                next_speaker_idx = (last_speaker_idx + 1) % len(chat.agents)
                return chat.agents[next_speaker_idx]

            speaker_selection = round_robin
        current_messages: list[Message] = []
        first_agent_id = speaker_selection(chat, [])
        first_message = self.send_message(chat_id, first_agent_id, initial_prompt)
        current_messages.append(first_message)
        # Assume self.agent_map is a dict mapping agent_id to Agent instance
        for _ in range(max_turns):
            history = self.get_chat_history(chat_id)
            next_agent_id = speaker_selection(chat, history)
            agent = self.agent_map[next_agent_id]  # Use a mapping for agent lookup
            formatted_history = [
                {"role": "user" if msg.agent_id != next_agent_id else "assistant", "content": msg.content}
                for msg in history
            ]
            result = agent.run(
                {
                    "input": initial_prompt,
                    "conversation_history": formatted_history,
                },
            )
            response = result.get("output", "")
            message = self.send_message(chat_id, next_agent_id, response)
            current_messages.append(message)
        return current_messages
