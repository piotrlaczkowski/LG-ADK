"""Agent Tools for LG-ADK. Provides a collection of tools for agents to use."""

import asyncio
import time
import uuid
from typing import Any

from pydantic import BaseModel

from lg_adk import Agent
from lg_adk.database import DatabaseManager
from lg_adk.memory import MemoryManager
from lg_adk.utils.logging import get_logger

logger = get_logger(__name__)


class DelegationTool:
    """Tool for delegating tasks to other agents.

    This tool allows an agent to:
    1. Delegate a task to another agent
    2. Wait for the result (sync or async)
    3. Get the result when it's ready
    """

    def __init__(self, agent_registry: dict[str, Agent] | None = None):
        """Initialize the delegation tool.

        Args:
            agent_registry: A dictionary mapping agent names to Agent instances.
        """
        self.agent_registry = agent_registry or {}
        self._task_results = {}
        self._pending_tasks = {}

    def register_agent(self, name: str, agent: Agent) -> None:
        """Register an agent for delegation.

        Args:
            name: A unique name for the agent.
            agent: The Agent instance.
        """
        self.agent_registry[name] = agent

    def delegate(
        self,
        agent_name: str,
        input_data: dict[str, Any],
        wait: bool = True,
        _timeout: float | None = None,
    ) -> dict[str, Any]:
        """Delegate a task to another agent (synchronous).

        Args:
            agent_name: The name of the agent to delegate to.
            input_data: The input data for the agent.
            wait: Whether to wait for the task to complete.
            _timeout: The maximum time to wait (in seconds). (Currently unused)

        Returns:
            The result of the task.

        Raises:
            KeyError: If the agent name is not found.
            TimeoutError: If the task times out.
        """
        if agent_name not in self.agent_registry:
            raise KeyError(f"Agent '{agent_name}' not found in registry")

        agent = self.agent_registry[agent_name]

        if wait:
            # Run the agent synchronously
            return agent.run(input_data)
        else:
            # Run the agent asynchronously
            task_id = f"{agent_name}_{len(self._pending_tasks)}"

            # Create a task for the agent
            loop = asyncio.get_event_loop()
            task = loop.create_task(agent.arun(input_data))

            # Store the task
            self._pending_tasks[task_id] = task

            # Return the task ID
            return {"task_id": task_id, "status": "pending"}

    async def adelegate(
        self,
        agent_name: str,
        input_data: dict[str, Any],
        wait: bool = True,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Delegate a task to another agent (asynchronous).

        Args:
            agent_name: The name of the agent to delegate to.
            input_data: The input data for the agent.
            wait: Whether to wait for the task to complete.
            timeout: The maximum time to wait (in seconds).

        Returns:
            The result of the task.

        Raises:
            KeyError: If the agent name is not found.
            TimeoutError: If the task times out.
        """
        if agent_name not in self.agent_registry:
            raise KeyError(f"Agent '{agent_name}' not found in registry")

        agent = self.agent_registry[agent_name]

        if wait:
            # Run the agent and wait for completion
            try:
                result = await asyncio.wait_for(agent.arun(input_data), timeout)
                return result
            except asyncio.TimeoutError:
                raise TimeoutError(f"Task for agent '{agent_name}' timed out after {timeout}s")
        else:
            # Run the agent asynchronously without waiting
            task_id = f"{agent_name}_{len(self._pending_tasks)}"

            # Create a task for the agent
            task = asyncio.create_task(agent.arun(input_data))

            # Store the task
            self._pending_tasks[task_id] = task

            # Return the task ID
            return {"task_id": task_id, "status": "pending"}

    def get_result(self, task_id: str, wait: bool = True, timeout: float | None = None) -> dict[str, Any]:
        """Get the result of a previously delegated task.

        Args:
            task_id: The ID of the task.
            wait: Whether to wait for the task to complete.
            timeout: The maximum time to wait (in seconds).

        Returns:
            The result of the task.

        Raises:
            KeyError: If the task ID is not found.
            TimeoutError: If the task times out.
        """
        if task_id not in self._pending_tasks:
            if task_id in self._task_results:
                return self._task_results[task_id]
            raise KeyError(f"Task '{task_id}' not found")

        task = self._pending_tasks[task_id]

        if task.done():
            # Task is already complete
            result = task.result()

            # Store the result and remove the task
            self._task_results[task_id] = result
            del self._pending_tasks[task_id]

            return result

        if not wait:
            # Don't wait for the task
            return {"task_id": task_id, "status": "pending"}

        # Wait for the task
        loop = asyncio.get_event_loop()
        try:
            if timeout:
                result = loop.run_until_complete(asyncio.wait_for(task, timeout))
            else:
                result = loop.run_until_complete(task)

            # Store the result and remove the task
            self._task_results[task_id] = result
            del self._pending_tasks[task_id]

            return result
        except asyncio.TimeoutError:
            raise TimeoutError(f"Task '{task_id}' timed out after {timeout}s")


class MemoryToolConfig(BaseModel):
    """Configuration for the memory tool."""

    memory_manager: MemoryManager | None = None
    db_manager: DatabaseManager | None = None
    collection_name: str = "memories"
    default_ttl: int | None = None  # Time to live in seconds


class MemoryTool:
    """Tool for storing and retrieving memories.

    This tool allows an agent to:
    1. Store memories in a persistent database
    2. Retrieve memories based on various criteria
    3. Delete memories when they're no longer needed
    """

    def __init__(self, config: MemoryToolConfig | None = None):
        """Initialize the memory tool.

        Args:
            config: The configuration for the memory tool.
        """
        self.config = config or MemoryToolConfig()

        # Use the provided memory manager or create a new one
        self.memory_manager = self.config.memory_manager or MemoryManager()

        # Use the provided database manager or create a new one
        self.db_manager = self.config.db_manager

        if not self.db_manager:
            # Import here to avoid circular imports
            from lg_adk.database import DatabaseManager

            self.db_manager = DatabaseManager()

        # Ensure the collection exists
        self._init_collection()

    def _init_collection(self) -> None:
        """Initialize the memory collection."""
        if not self.db_manager.collection_exists(self.config.collection_name):
            self.db_manager.create_collection(
                self.config.collection_name,
                index_fields=["user_id", "session_id", "tags", "timestamp"],
                ttl_field="expiry" if self.config.default_ttl else None,
            )

    def store(
        self,
        session_id: str,
        content: Any,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> str:
        """Store a memory.

        Args:
            session_id: The session ID.
            content: The content to store.
            tags: Tags for categorizing the memory.
            metadata: Additional metadata.
            ttl: Time to live in seconds (overrides default).

        Returns:
            The ID of the stored memory.
        """
        # Prepare the memory document
        memory_id = str(uuid.uuid4())
        timestamp = int(time.time())

        document = {
            "memory_id": memory_id,
            "session_id": session_id,
            "content": content,
            "tags": tags or [],
            "metadata": metadata or {},
            "timestamp": timestamp,
        }

        # Add expiry if TTL is specified
        ttl_value = ttl or self.config.default_ttl
        if ttl_value:
            document["expiry"] = timestamp + ttl_value

        # Store the memory
        self.db_manager.insert(self.config.collection_name, document)

        return memory_id

    def retrieve(
        self,
        session_id: str,
        query: dict[str, Any] | None = None,
        limit: int = 10,
        sort_by: str = "timestamp",
        sort_order: str = "desc",
    ) -> list[dict[str, Any]]:
        """Retrieve memories.

        Args:
            session_id: The session ID.
            query: Additional query parameters.
            limit: Maximum number of memories to retrieve.
            sort_by: Field to sort by.
            sort_order: Sort order ("asc" or "desc").

        Returns:
            A list of memories.
        """
        # Build the query
        full_query = {"session_id": session_id}
        if query:
            full_query.update(query)

        # Retrieve the memories
        memories = self.db_manager.find(
            self.config.collection_name,
            full_query,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        return memories

    def retrieve_by_tags(
        self,
        session_id: str,
        tags: list[str],
        match_all: bool = False,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieve memories by tags.

        Args:
            session_id: The session ID.
            tags: Tags to match.
            match_all: Whether all tags must match.
            limit: Maximum number of memories to retrieve.

        Returns:
            A list of memories.
        """
        if not tags:
            return self.retrieve(session_id, limit=limit)

        # Build the query for tags
        tags_query = {"tags": {"$all": tags}} if match_all else {"tags": {"$in": tags}}
        return self.retrieve(session_id, query=tags_query, limit=limit)

    def delete(self, memory_id: str) -> bool:
        """Delete a memory.

        Args:
            memory_id: The ID of the memory.

        Returns:
            True if the memory was deleted, False otherwise.
        """
        result = self.db_manager.delete(
            self.config.collection_name,
            {"memory_id": memory_id},
        )

        return result.deleted_count > 0

    def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> str:
        """Add a memory to the memory store.

        Args:
            content: The content of the memory.
            metadata: Optional metadata for the memory.
            memory_id: Optional ID for the memory. If not provided, a random ID will be generated.

        Returns:
            The ID of the stored memory.
        """
        if metadata is None:
            metadata = {}

        if memory_id is None:
            memory_id = str(uuid.uuid4())

        memory = {
            "id": memory_id,
            "content": content,
            "metadata": metadata,
            "created_at": int(time.time()),
        }

        self.memory_manager.add_memory(memory)
        return memory_id


class UserInfoTool:
    """Tool for managing user information.

    This tool allows an agent to:
    1. Store and retrieve user preferences
    2. Access user metadata
    3. Update user settings
    """

    def __init__(self, db_manager: DatabaseManager | None = None):
        """Initialize the user info tool.

        Args:
            db_manager: The database manager to use.
        """
        # Use the provided database manager or create a new one
        self.db_manager = db_manager

        if not self.db_manager:
            # Import here to avoid circular imports
            from lg_adk.database import DatabaseManager

            self.db_manager = DatabaseManager()

        # Ensure the collections exist
        self._init_collections()

    def _init_collections(self) -> None:
        """Initialize the user collections."""
        # User profiles collection
        if not self.db_manager.collection_exists("user_profiles"):
            self.db_manager.create_collection(
                "user_profiles",
                index_fields=["user_id", "email"],
            )

        # User preferences collection
        if not self.db_manager.collection_exists("user_preferences"):
            self.db_manager.create_collection(
                "user_preferences",
                index_fields=["user_id", "category"],
            )

    def get_user_profile(self, user_id: str) -> dict[str, Any]:
        """Get a user's profile.

        Args:
            user_id: The ID of the user.

        Returns:
            The user profile or an empty dict if not found.
        """
        profile = self.db_manager.find_one("user_profiles", {"user_id": user_id})
        return profile or {}

    def update_user_profile(self, user_id: str, profile_data: dict[str, Any]) -> bool:
        """Update a user's profile.

        Args:
            user_id: The ID of the user.
            profile_data: The profile data to update.

        Returns:
            True if the profile was updated, False otherwise.
        """
        # Make sure we don't overwrite the user_id
        profile_data["user_id"] = user_id

        result = self.db_manager.update(
            "user_profiles",
            {"user_id": user_id},
            {"$set": profile_data},
            upsert=True,
        )

        return result.modified_count > 0 or result.upserted_id is not None

    def get_preference(
        self,
        user_id: str,
        key: str,
        category: str = "general",
        default: Any = None,
    ) -> Any:
        """Get a user preference.

        Args:
            user_id: The ID of the user.
            key: The preference key.
            category: The preference category.
            default: The default value if not found.

        Returns:
            The preference value or the default.
        """
        pref = self.db_manager.find_one(
            "user_preferences",
            {"user_id": user_id, "category": category, "key": key},
        )

        return pref.get("value", default) if pref else default

    def set_preference(
        self,
        user_id: str,
        key: str,
        value: Any,
        category: str = "general",
    ) -> bool:
        """Set a user preference.

        Args:
            user_id: The ID of the user.
            key: The preference key.
            value: The preference value.
            category: The preference category.

        Returns:
            True if the preference was set, False otherwise.
        """
        pref_data = {
            "user_id": user_id,
            "category": category,
            "key": key,
            "value": value,
            "updated_at": int(time.time()),
        }

        result = self.db_manager.update(
            "user_preferences",
            {"user_id": user_id, "category": category, "key": key},
            {"$set": pref_data},
            upsert=True,
        )

        return result.modified_count > 0 or result.upserted_id is not None

    def get_preferences(self, user_id: str, category: str | None = None) -> dict[str, Any]:
        """Get all preferences for a user.

        Args:
            user_id: The ID of the user.
            category: Optional category filter.

        Returns:
            A dictionary of preference keys and values.
        """
        query = {"user_id": user_id}
        if category:
            query["category"] = category

        prefs = self.db_manager.find("user_preferences", query)

        # Convert to a dictionary of key-value pairs
        return {pref["key"]: pref["value"] for pref in prefs}

    def delete_preference(
        self,
        user_id: str,
        key: str,
        category: str = "general",
    ) -> bool:
        """Delete a user preference.

        Args:
            user_id: The ID of the user.
            key: The preference key.
            category: The preference category.

        Returns:
            True if the preference was deleted, False otherwise.
        """
        result = self.db_manager.delete(
            "user_preferences",
            {"user_id": user_id, "category": category, "key": key},
        )

        return result.deleted_count > 0


# Tool integration with Agent class
def register_agent_tools(agent: Agent, tools: dict[str, DelegationTool | MemoryTool | UserInfoTool]) -> None:
    """Register tools with an agent.

    Args:
        agent: The agent to register tools with.
        tools: A dictionary mapping tool names to tool instances.
    """
    for tool_name, tool_instance in tools.items():
        agent.register_tool(tool_name, tool_instance)
