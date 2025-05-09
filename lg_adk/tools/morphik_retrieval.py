"""Morphik Retrieval tools for LG-ADK."""

import json
import logging
from typing import Any

from langchain.tools import Tool

from lg_adk.database.morphik_db import GraphEntity, GraphRelationship, MorphikDatabaseManager
from lg_adk.settings import Settings

logger = logging.getLogger(__name__)


class MorphikRetrievalTool(Tool):
    """Tool for retrieving documents from Morphik.

    This tool provides semantic search capabilities through Morphik's
    retrieval system.
    """

    name = "morphik_retrieval"
    description = """
    Use this tool to search and retrieve documents from the Morphik database.
    Input should be a search query in natural language.
    """

    def __init__(
        self,
        morphik_db=None,
    ) -> None:
        """Initialize MorphikRetrievalTool.

        Args:
            morphik_db: Optional MorphikDatabaseManager instance.
                        If not provided, will be initialized from Settings.
        """
        super().__init__(
            name=self.name,
            description=self.description,
            func=self._run,
            coroutine=self._arun,
        )
        self.morphik_db = morphik_db
        self._settings = Settings()

    def _get_morphik_db(self) -> MorphikDatabaseManager | None:
        """Get Morphik database manager, initializing from settings if needed.

        Returns:
            MorphikDatabaseManager instance or None if not available
        """
        if self.morphik_db:
            return self.morphik_db

        # Check if Morphik is set as default database
        if self._settings.USE_MORPHIK_AS_DEFAULT:
            from lg_adk.database.managers import get_database_manager

            self.morphik_db = get_database_manager()
            return self.morphik_db

        # Otherwise, create a new instance
        try:
            from lg_adk.database.morphik_db import MorphikDatabaseManager

            self.morphik_db = MorphikDatabaseManager(
                host=self._settings.MORPHIK_HOST,
                port=self._settings.MORPHIK_PORT,
                api_key=self._settings.MORPHIK_API_KEY,
                default_user=self._settings.MORPHIK_DEFAULT_USER,
                default_folder=self._settings.MORPHIK_DEFAULT_FOLDER,
            )
            return self.morphik_db
        except ImportError:
            logger.error("Morphik not available. Install with 'pip install morphik'")
            return None

    def _run(
        self,
        query: str,
        filter_metadata: dict[str, Any] | None = None,
        k: int = 5,
    ) -> str:
        """Run the tool to retrieve documents from Morphik.

        Args:
            query: The query to search with
            filter_metadata: Optional metadata filters
            k: Number of results to return

        Returns:
            String representation of the retrieved documents
        """
        morphik_db = self._get_morphik_db()
        if not morphik_db or not morphik_db.is_available():
            return "Morphik is not available. Please check your connection settings."

        # Query Morphik
        documents = morphik_db.query(query, k=k, filter_metadata=filter_metadata)

        # Format results
        result = []
        result.append(f"Retrieved {len(documents)} documents from Morphik:")
        result.append("")

        # Add documents to result
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            score = doc.get("score")
            score_str = f" (score: {score:.4f})" if score is not None else ""

            # Add document header with score if available
            result.append(f"Document {i}{score_str}:")

            # Add content (truncate if too long)
            if len(content) > 500:
                result.append(f"{content[:500]}...")
            else:
                result.append(content)

            result.append("")

        return "\n".join(result)

    async def _arun(
        self,
        query: str,
        filter_metadata: dict[str, Any] | None = None,
        k: int = 5,
    ) -> str:
        """Async version of _run.

        Args:
            query: The query to search with
            filter_metadata: Optional metadata filters
            k: Number of results to return

        Returns:
            String representation of the retrieved documents
        """
        # Currently just calls the sync version
        return self._run(query, filter_metadata, k)


class MorphikMCPTool(Tool):
    """Tool for retrieving structured context using Model Context Protocol from Morphik.

    This tool retrieves documents from Morphik and formats them according to the
    Model Context Protocol (MCP) specification for structured context.
    """

    name = "morphik_mcp_retrieval"
    description = """
    Use this tool to get structured context from Morphik using MCP format.
    Input should be a search query in natural language.
    """

    def __init__(
        self,
        morphik_db=None,
        model_provider="openai",
    ) -> None:
        """Initialize MorphikMCPTool.

        Args:
            morphik_db: Optional MorphikDatabaseManager instance.
                        If not provided, will be initialized from Settings.
            model_provider: The model provider that will use the MCP context.
                            (default: "openai")
        """
        super().__init__(
            name=self.name,
            description=self.description,
            func=self._run,
            coroutine=self._arun,
        )
        self.morphik_db = morphik_db
        self.model_provider = model_provider
        self._settings = Settings()

    def _get_morphik_db(self) -> MorphikDatabaseManager | None:
        """Get Morphik database manager, initializing from settings if needed.

        Returns:
            MorphikDatabaseManager instance or None if not available
        """
        if self.morphik_db:
            return self.morphik_db

        # Check if Morphik is set as default database
        if self._settings.USE_MORPHIK_AS_DEFAULT:
            from lg_adk.database.managers import get_database_manager

            self.morphik_db = get_database_manager()
            return self.morphik_db

        # Otherwise, create a new instance
        try:
            from lg_adk.database.morphik_db import MorphikDatabaseManager

            self.morphik_db = MorphikDatabaseManager(
                host=self._settings.MORPHIK_HOST,
                port=self._settings.MORPHIK_PORT,
                api_key=self._settings.MORPHIK_API_KEY,
                default_user=self._settings.MORPHIK_DEFAULT_USER,
                default_folder=self._settings.MORPHIK_DEFAULT_FOLDER,
            )
            return self.morphik_db
        except ImportError:
            logger.error("Morphik not available. Install with 'pip install morphik'")
            return None

    def _run(
        self,
        query: str,
        filter_metadata: dict[str, Any] | None = None,
        k: int = 5,
    ) -> str:
        """Run the tool to retrieve MCP context from Morphik.

        Args:
            query: The query to search with
            filter_metadata: Optional metadata filters
            k: Number of results to return

        Returns:
            MCP context as JSON string
        """
        morphik_db = self._get_morphik_db()
        if not morphik_db or not morphik_db.is_available():
            return "Morphik is not available. Please check your connection settings."

        # Get MCP context
        mcp_context = morphik_db.get_mcp_context(query, k=k, filter_metadata=filter_metadata)

        return mcp_context

    async def _arun(
        self,
        query: str,
        filter_metadata: dict[str, Any] | None = None,
        k: int = 5,
    ) -> str:
        """Async version of _run.

        Args:
            query: The query to search with
            filter_metadata: Optional metadata filters
            k: Number of results to return

        Returns:
            MCP context as JSON string
        """
        # Currently just calls the sync version
        return self._run(query, filter_metadata, k)


class MorphikGraphTool(Tool):
    """Tool for retrieving knowledge graph information from Morphik.

    This tool enables querying Morphik's knowledge graph capabilities to retrieve
    entity relationships and traverse knowledge graphs.
    """

    name = "morphik_graph"
    description = """
    Use this tool to query knowledge graphs from the Morphik database.
    Input should be a query about entities, relationships, or concepts.
    """

    def __init__(
        self,
        morphik_db=None,
        graph_name: str | None = None,
        hop_depth: int = 1,
        include_paths: bool = False,
    ) -> None:
        """Initialize MorphikGraphTool.

        Args:
            morphik_db: Optional MorphikDatabaseManager instance.
                      If not provided, will be initialized from Settings.
            graph_name: Optional name of the knowledge graph to query.
            hop_depth: Number of hops to use when traversing the graph.
            include_paths: Whether to include paths between entities in results.
        """
        super().__init__(
            name=self.name,
            description=self.description,
            func=self._run,
            coroutine=self._arun,
        )
        self.morphik_db = morphik_db
        self.graph_name = graph_name
        self.hop_depth = hop_depth
        self.include_paths = include_paths
        self._settings = Settings()

    def _get_morphik_db(self) -> MorphikDatabaseManager | None:
        """Get Morphik database manager, initializing from settings if needed.

        Returns:
            MorphikDatabaseManager instance or None if not available
        """
        if self.morphik_db:
            return self.morphik_db

        # Check if Morphik is set as default database
        if self._settings.USE_MORPHIK_AS_DEFAULT:
            from lg_adk.database.managers import get_database_manager

            self.morphik_db = get_database_manager()
            return self.morphik_db

        # Otherwise, create a new instance
        try:
            from lg_adk.database.morphik_db import MorphikDatabaseManager

            self.morphik_db = MorphikDatabaseManager(
                host=self._settings.MORPHIK_HOST,
                port=self._settings.MORPHIK_PORT,
                api_key=self._settings.MORPHIK_API_KEY,
                default_user=self._settings.MORPHIK_DEFAULT_USER,
                default_folder=self._settings.MORPHIK_DEFAULT_FOLDER,
            )
            return self.morphik_db
        except ImportError:
            logger.error("Morphik not available. Install with 'pip install morphik'")
            return None

    def _run(
        self,
        query: str,
        filter_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Run the tool to query the knowledge graph.

        Args:
            query: The query to search with
            filter_metadata: Optional metadata filters

        Returns:
            String representation of the graph query results
        """
        morphik_db = self._get_morphik_db()
        if not morphik_db or not morphik_db.is_available():
            return "Morphik is not available. Please check your connection settings."

        # If no graph name, get available graphs
        if not self.graph_name:
            graphs = morphik_db.get_knowledge_graphs()
            if not graphs:
                return "No knowledge graphs available in Morphik."

            result = ["Available knowledge graphs:"]
            for graph in graphs:
                result.append(f"- {graph}")
            return "\n".join(result)

        # Query Morphik with graph parameters
        results = morphik_db.query(
            query,
            filter_metadata=filter_metadata,
            graph_name=self.graph_name,
            hop_depth=self.hop_depth,
            include_paths=self.include_paths,
        )

        if not results:
            return f"No results found for query '{query}' in graph '{self.graph_name}'."

        # Categorize results
        entities = []
        relationships = []
        paths = []
        documents = []

        for result in results:
            if isinstance(result, GraphEntity):
                entities.append(result)
            elif isinstance(result, GraphRelationship):
                relationships.append(result)
            elif isinstance(result, dict) and "path" in result:
                paths.append(result["path"])
            else:
                documents.append(result)

        # Format output
        output = []

        # Add query info
        output.append(f"Knowledge Graph: {self.graph_name}")
        output.append(f"Query: {query}")
        output.append(f"Results: {len(results)} items")
        output.append("")

        # Add entities section
        if entities:
            output.append("## ENTITIES")
            for entity in entities:
                output.append(f"Entity: {entity.label} (Type: {entity.entity_type}, ID: {entity.entity_id})")
                if entity.properties:
                    output.append("  Properties:")
                    for key, value in entity.properties.items():
                        output.append(f"    - {key}: {value}")
                output.append("")

        # Add relationships section
        if relationships:
            output.append("## RELATIONSHIPS")
            for rel in relationships:
                output.append(
                    f"Relationship: {rel.source} -> {rel.relationship_type} -> {rel.target}",
                )
                output.append(f"  ID: {rel.relationship_id}")
                output.append("")

        # Add paths section
        if paths:
            output.append("## PATHS")
            for i, path in enumerate(paths, 1):
                output.append(f"Path {i}:")
                path_str = " -> ".join([node.get("label", node.get("id", "Unknown")) for node in path])
                output.append(f"  {path_str}")
                output.append("")

        # Add documents section
        if documents:
            output.append("## DOCUMENTS")
            for i, doc in enumerate(documents, 1):
                content = doc.get("content", "")
                score = doc.get("score")
                score_str = f" (score: {score:.4f})" if score is not None else ""

                output.append(f"Document {i}{score_str}:")
                if len(content) > 300:
                    output.append(f"{content[:300]}...")
                else:
                    output.append(content)
                output.append("")

        return "\n".join(output)

    async def _arun(
        self,
        query: str,
        filter_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Async version of _run.

        Args:
            query: The query to search with
            filter_metadata: Optional metadata filters

        Returns:
            String representation of the graph query results
        """
        # Currently just calls the sync version
        return self._run(query, filter_metadata)

    def retrieve(
        self,
        query: str,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve graph data or documents matching the query.

        Args:
            query: The query to search with
            filter_metadata: Optional metadata filters

        Returns:
            Processed results in a standard format
        """
        morphik_db = self._get_morphik_db()
        if not morphik_db or not morphik_db.is_available():
            return []

        results = morphik_db.query(
            query,
            filter_metadata=filter_metadata,
            graph_name=self.graph_name,
            hop_depth=self.hop_depth,
            include_paths=self.include_paths,
        )

        # Process results into a standard format
        processed_results = []
        for result in results:
            if isinstance(result, GraphEntity):
                processed_results.append(
                    {
                        "type": "entity",
                        "id": result.entity_id,
                        "label": result.label,
                        "entity_type": result.entity_type,
                        "properties": result.properties,
                        "document_ids": result.document_ids,
                    },
                )
            elif isinstance(result, GraphRelationship):
                processed_results.append(
                    {
                        "type": "relationship",
                        "id": result.relationship_id,
                        "source": result.source,
                        "target": result.target,
                        "relationship_type": result.relationship_type,
                        "document_ids": result.document_ids,
                    },
                )
            elif isinstance(result, dict) and "path" in result:
                processed_results.append(
                    {
                        "type": "path",
                        "path": result["path"],
                    },
                )
            elif isinstance(result, dict) and "content" in result:
                processed_results.append(
                    {
                        "type": "document",
                        "content": result["content"],
                        "metadata": result.get("metadata", {}),
                        "document_id": result.get("document_id"),
                        "score": result.get("score"),
                    },
                )

        return processed_results


class MorphikGraphCreationTool(Tool):
    """Tool for creating and managing knowledge graphs in Morphik.

    This tool enables creating and updating knowledge graphs from documents,
    with support for custom entity extraction and resolution.
    """

    name = "morphik_graph_creation"
    description = """
    Use this tool to create or update knowledge graphs in Morphik.
    Input format should be a JSON object with these fields:
    - action: "create" or "update"
    - graph_name: name of the graph to create or update
    - document_ids: optional list of document IDs to include
    - filters: optional metadata filters for documents to include
    """

    def __init__(
        self,
        morphik_db=None,
    ) -> None:
        """Initialize MorphikGraphCreationTool.

        Args:
            morphik_db: Optional MorphikDatabaseManager instance.
                      If not provided, will be initialized from Settings.
        """
        super().__init__(
            name=self.name,
            description=self.description,
            func=self._run,
            coroutine=self._arun,
        )
        self.morphik_db = morphik_db
        self._settings = Settings()

    def _get_morphik_db(self) -> MorphikDatabaseManager | None:
        """Get Morphik database manager, initializing from settings if needed.

        Returns:
            MorphikDatabaseManager instance or None if not available
        """
        if self.morphik_db:
            return self.morphik_db

        # Check if Morphik is set as default database
        if self._settings.USE_MORPHIK_AS_DEFAULT:
            from lg_adk.database.managers import get_database_manager

            self.morphik_db = get_database_manager()
            return self.morphik_db

        # Otherwise, create a new instance
        try:
            from lg_adk.database.morphik_db import MorphikDatabaseManager

            self.morphik_db = MorphikDatabaseManager(
                host=self._settings.MORPHIK_HOST,
                port=self._settings.MORPHIK_PORT,
                api_key=self._settings.MORPHIK_API_KEY,
                default_user=self._settings.MORPHIK_DEFAULT_USER,
                default_folder=self._settings.MORPHIK_DEFAULT_FOLDER,
            )
            return self.morphik_db
        except ImportError:
            logger.error("Morphik not available. Install with 'pip install morphik'")
            return None

    def _run(
        self,
        request: str,
    ) -> str:
        """Run the tool to create or update knowledge graphs.

        Args:
            request: JSON string with knowledge graph creation/update parameters

        Returns:
            Result message
        """
        morphik_db = self._get_morphik_db()
        if not morphik_db or not morphik_db.is_available():
            return "Morphik is not available. Please check your connection settings."

        try:
            # Parse request
            params = json.loads(request)

            action = params.get("action", "").lower()
            graph_name = params.get("graph_name")
            document_ids = params.get("document_ids")
            filters = params.get("filters")
            prompt_overrides = params.get("prompt_overrides")

            if not graph_name:
                return "Error: graph_name is required"

            # Process action
            if action == "create":
                result = morphik_db.create_knowledge_graph(
                    graph_name=graph_name,
                    document_ids=document_ids,
                    filters=filters,
                    prompt_overrides=prompt_overrides,
                )

                if result:
                    return f"Successfully created knowledge graph '{graph_name}'"
                else:
                    return f"Failed to create knowledge graph '{graph_name}'"

            elif action == "update":
                result = morphik_db.update_knowledge_graph(
                    graph_name=graph_name,
                    document_ids=document_ids,
                    filters=filters,
                    prompt_overrides=prompt_overrides,
                )

                if result:
                    return f"Successfully updated knowledge graph '{graph_name}'"
                else:
                    return f"Failed to update knowledge graph '{graph_name}'"

            elif action == "delete":
                result = morphik_db.delete_knowledge_graph(graph_name)
                if result:
                    return f"Successfully deleted knowledge graph '{graph_name}'"
                else:
                    return f"Failed to delete knowledge graph '{graph_name}'"

            elif action == "list":
                graphs = morphik_db.get_knowledge_graphs()
                if graphs:
                    result = ["Available knowledge graphs:"]
                    for graph in graphs:
                        result.append(f"- {graph}")
                    return "\n".join(result)
                else:
                    return "No knowledge graphs available"

            else:
                return f"Unknown action: {action}. Use 'create', 'update', 'delete', or 'list'."

        except json.JSONDecodeError:
            return "Error: Invalid JSON input. Please provide a valid JSON object."
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(
        self,
        request: str,
    ) -> str:
        """Async version of _run.

        Args:
            request: JSON string with knowledge graph creation/update parameters

        Returns:
            Result message
        """
        # Currently just calls the sync version
        return self._run(request)
