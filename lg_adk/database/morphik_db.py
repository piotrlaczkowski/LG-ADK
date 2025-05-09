"""Morphik database integration for LG-ADK.

This module provides integration with Morphik, an unstructured data database
designed for AI applications.
"""

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from lg_adk.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


# Define fallback classes for when Morphik is not available
class _EntityExtractionExample(BaseModel):
    """Fallback class for Morphik's EntityExtractionExample when not available."""

    label: str = Field(..., description="Entity label")
    entity_type: str = Field(..., description="Entity type")
    properties: dict[str, Any] | None = Field(None, description="Optional entity properties")


class _EntityResolutionExample(BaseModel):
    """Fallback class for Morphik's EntityResolutionExample when not available."""

    canonical: str = Field(..., description="Canonical entity name")
    variants: list[str] = Field(..., description="Entity name variants to resolve to canonical form")


class _EntityExtractionPromptOverride(BaseModel):
    """Fallback class for Morphik's EntityExtractionPromptOverride when not available."""

    prompt: str | None = Field(None, description="Custom prompt template")
    examples: list[_EntityExtractionExample] | None = Field(None, description="Example entities")


class _EntityResolutionPromptOverride(BaseModel):
    """Fallback class for Morphik's EntityResolutionPromptOverride when not available."""

    prompt: str | None = Field(None, description="Custom prompt template")
    examples: list[_EntityResolutionExample] | None = Field(None, description="Example entity resolutions")


class _GraphPromptOverrides(BaseModel):
    """Fallback class for Morphik's GraphPromptOverrides when not available."""

    entity_extraction: _EntityExtractionPromptOverride | None = Field(
        None,
        description="Entity extraction overrides",
    )
    entity_resolution: _EntityResolutionPromptOverride | None = Field(
        None,
        description="Entity resolution overrides",
    )


# Try to import Morphik, otherwise use fallback classes
try:
    import morphik
    from morphik import Morphik
    from morphik.models import (
        EntityExtractionExample,
        EntityExtractionPromptOverride,
        EntityResolutionExample,
        EntityResolutionPromptOverride,
        GraphPromptOverrides,
    )

    MORPHIK_AVAILABLE = True
except ImportError:
    logger.warning("Morphik package not available. Install with 'pip install morphik'.")
    MORPHIK_AVAILABLE = False
    # Use fallback classes if Morphik is not available
    EntityExtractionExample = _EntityExtractionExample
    EntityResolutionExample = _EntityResolutionExample
    EntityExtractionPromptOverride = _EntityExtractionPromptOverride
    EntityResolutionPromptOverride = _EntityResolutionPromptOverride
    GraphPromptOverrides = _GraphPromptOverrides


class MorphikDocument(BaseModel):
    """A document retrieved from Morphik.

    Attributes:
        content: The text content of the document.
        metadata: Metadata about the document.
        score: Relevance score for the document (if available).
        source: Source of the document.
    """

    content: str = Field(..., description="Document content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    score: float | None = Field(None, description="Relevance score")
    source: str | None = Field(None, description="Document source")


class GraphEntity(BaseModel):
    """Entity in a knowledge graph."""

    entity_id: str = Field(..., description="Unique identifier for the entity")
    label: str = Field(..., description="Display name of the entity")
    entity_type: str = Field(..., description="Type of entity (PERSON, ORGANIZATION, etc.)")
    properties: dict[str, Any] = Field(default_factory=dict, description="Additional attributes of the entity")
    document_ids: list[str] = Field(default_factory=list, description="Source document IDs containing this entity")


class GraphRelationship(BaseModel):
    """Relationship between entities in a knowledge graph."""

    relationship_id: str = Field(..., description="Unique identifier for the relationship")
    source: str = Field(..., description="Source entity ID or label")
    target: str = Field(..., description="Target entity ID or label")
    relationship_type: str = Field(..., description="Type of relationship")
    document_ids: list[str] = Field(
        default_factory=list,
        description="Source document IDs containing this relationship",
    )


class MorphikDatabaseManager(DatabaseManager):
    """Morphik database manager for LG-ADK.

    This class provides an interface to the Morphik database for LG-ADK applications.
    It handles connections to Morphik, document management, and knowledge graph operations.

    Attributes:
        host: Morphik server host
        port: Morphik server port
        api_key: API key for authentication
        default_user: Default user ID for documents
        default_folder: Default folder ID for documents
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        api_key: str | None = None,
        default_user: str = "default",
        default_folder: str = "lg-adk",
    ):
        """Initialize the Morphik database manager.

        Args:
            host: Morphik server host
            port: Morphik server port
            api_key: API key for authentication
            default_user: Default user ID for documents
            default_folder: Default folder ID for documents
        """
        super().__init__(**locals())

        if not MORPHIK_AVAILABLE:
            logger.warning("Morphik package not installed. Some features will be unavailable.")
            self._client = None
            return

        # Initialize Morphik client
        try:
            # Handle different possible initialization methods based on Morphik version
            if hasattr(morphik, "MorphikClient"):
                self._client = Morphik(
                    base_url=f"http://{host}:{port}",
                    api_key=api_key,
                )
            else:
                # Fallback to standard morphik.init
                morphik.init(
                    host=host,
                    port=port,
                    api_key=api_key,
                )
                self._client = morphik

            logger.info(f"Connected to Morphik at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Morphik: {str(e)}")
            self._client = None

    def is_available(self) -> bool:
        """Check if Morphik is available.

        Returns:
            True if Morphik is available, False otherwise
        """
        return MORPHIK_AVAILABLE and self._client is not None

    def add_document(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        folder_id: str | None = None,
    ) -> str:
        """Add a document to Morphik.

        Args:
            content: Document content
            metadata: Optional metadata
            folder_id: Optional folder ID

        Returns:
            Document ID
        """
        if not self.is_available():
            logger.error("Morphik is not available")
            return ""

        try:
            # Use default folder if not specified
            target_folder = folder_id or self.default_folder

            # Add document
            doc = self._client.ingest_text(
                text=content,
                metadata=metadata or {},
                folder=target_folder,
            )
            logger.info(f"Added document with ID: {doc.id}")
            return doc.id
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return ""

    def query(
        self,
        query_text: str,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        graph_name: str | None = None,
        hop_depth: int = 1,
        include_paths: bool = False,
    ) -> list[Any]:
        """Query Morphik for documents.

        Args:
            query_text: Query text
            k: Number of results to return
            filter_metadata: Optional metadata filter
            graph_name: Optional knowledge graph name
            hop_depth: Number of hops for graph traversal
            include_paths: Whether to include graph paths

        Returns:
            List of query results (documents or graph items)
        """
        if not self.is_available():
            logger.error("Morphik is not available")
            return []

        try:
            # If graph name is provided, query with graph
            if graph_name:
                return self._query_with_graph(
                    query_text,
                    graph_name,
                    k,
                    filter_metadata,
                    hop_depth,
                    include_paths,
                )

            # Standard query
            results = self._client.query(
                query_text,
                k=k,
                filters=filter_metadata,
            )

            # Convert to simplified document format
            documents = []
            if hasattr(results, "chunks") and results.chunks:
                for chunk in results.chunks:
                    documents.append(
                        {
                            "content": chunk.content,
                            "metadata": chunk.metadata,
                            "document_id": chunk.document_id,
                            "score": getattr(chunk, "score", None),
                        },
                    )
            return documents
        except Exception as e:
            logger.error(f"Failed to query Morphik: {e}")
            return []

    def _query_with_graph(
        self,
        query_text: str,
        graph_name: str,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        hop_depth: int = 1,
        include_paths: bool = False,
    ) -> list[GraphEntity | GraphRelationship | dict[str, Any]]:
        """Query using knowledge graph.

        Args:
            query_text: Query text
            graph_name: Knowledge graph name
            k: Number of results
            filter_metadata: Metadata filter
            hop_depth: Number of hops for traversal
            include_paths: Whether to include entity paths

        Returns:
            List of graph entities, relationships, or paths
        """
        try:
            # Query with graph parameters
            results = self._client.query(
                query_text,
                graph_name=graph_name,
                k=k,
                filters=filter_metadata,
                hop_depth=hop_depth,
                include_paths=include_paths,
            )

            # Process and convert results to our format
            processed_results = []

            # Extract graph information if available
            if hasattr(results, "metadata") and results.metadata and "graph" in results.metadata:
                graph_data = results.metadata["graph"]

                # Process entities
                if "entities" in graph_data:
                    for entity in graph_data["entities"]:
                        processed_results.append(
                            GraphEntity(
                                entity_id=entity.get("id", ""),
                                label=entity.get("label", ""),
                                entity_type=entity.get("type", ""),
                                properties=entity.get("properties", {}),
                                document_ids=entity.get("document_ids", []),
                            ),
                        )

                # Process relationships
                if "relationships" in graph_data:
                    for rel in graph_data["relationships"]:
                        processed_results.append(
                            GraphRelationship(
                                relationship_id=rel.get("id", ""),
                                source=rel.get("source", ""),
                                target=rel.get("target", ""),
                                relationship_type=rel.get("type", ""),
                                document_ids=rel.get("document_ids", []),
                            ),
                        )

                # Process paths if requested
                if include_paths and "paths" in graph_data:
                    for path in graph_data["paths"]:
                        processed_results.append({"path": path})

            # If no specific graph data was found, include document chunks
            if not processed_results and hasattr(results, "chunks"):
                for chunk in results.chunks:
                    processed_results.append(
                        {
                            "content": chunk.content,
                            "metadata": chunk.metadata,
                            "document_id": chunk.document_id,
                            "score": getattr(chunk, "score", None),
                        },
                    )

            return processed_results
        except Exception as e:
            logger.error(f"Failed to query with graph: {e}")
            return []

    def get_mcp_context(
        self,
        query_text: str,
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Get MCP context for a query.

        Args:
            query_text: Query text
            k: Number of results
            filter_metadata: Metadata filter

        Returns:
            MCP context as JSON string
        """
        if not self.is_available():
            logger.error("Morphik is not available")
            return json.dumps({"error": "Morphik is not available"})

        try:
            # Query with MCP format
            response = self._client.get_mcp_context(
                query_text,
                k=k,
                filters=filter_metadata,
            )

            # Return MCP context as JSON string
            if isinstance(response, dict):
                return json.dumps(response)
            else:
                return str(response)
        except Exception as e:
            logger.error(f"Failed to get MCP context: {e}")
            return json.dumps({"error": f"Failed to get MCP context: {str(e)}"})

    def create_knowledge_graph(
        self,
        graph_name: str,
        document_ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        prompt_overrides: dict[str, Any] | None = None,
    ) -> bool:
        """Create a knowledge graph from documents.

        Args:
            graph_name: Name of the graph
            document_ids: Optional list of document IDs
            filters: Optional metadata filters
            prompt_overrides: Optional prompt customization

        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.error("Morphik is not available")
            return False

        try:
            # Process prompt overrides if provided
            graph_prompt_overrides = None
            if prompt_overrides:
                # Handle entity extraction examples
                entity_extraction = None
                if "entity_extraction" in prompt_overrides:
                    ext_data = prompt_overrides["entity_extraction"]
                    examples = []

                    if "examples" in ext_data:
                        for example in ext_data["examples"]:
                            examples.append(
                                EntityExtractionExample(
                                    label=example["label"],
                                    entity_type=example["type"],
                                    properties=example.get("properties"),
                                ),
                            )

                    entity_extraction = EntityExtractionPromptOverride(
                        prompt=ext_data.get("prompt"),
                        examples=examples if examples else None,
                    )

                # Handle entity resolution examples
                entity_resolution = None
                if "entity_resolution" in prompt_overrides:
                    res_data = prompt_overrides["entity_resolution"]
                    examples = []

                    if "examples" in res_data:
                        for example in res_data["examples"]:
                            examples.append(
                                EntityResolutionExample(
                                    canonical=example["canonical"],
                                    variants=example["variants"],
                                ),
                            )

                    entity_resolution = EntityResolutionPromptOverride(
                        prompt=res_data.get("prompt"),
                        examples=examples if examples else None,
                    )

                # Create GraphPromptOverrides if either extraction or resolution is defined
                if entity_extraction or entity_resolution:
                    graph_prompt_overrides = GraphPromptOverrides(
                        entity_extraction=entity_extraction,
                        entity_resolution=entity_resolution,
                    )

            # Create the knowledge graph
            graph = self._client.create_graph(
                name=graph_name,
                documents=document_ids,
                filters=filters,
                prompt_overrides=graph_prompt_overrides,
            )

            logger.info(f"Created knowledge graph '{graph_name}' with {len(getattr(graph, 'entities', []))} entities")
            return True
        except Exception as e:
            logger.error(f"Failed to create knowledge graph: {e}")
            return False

    def update_knowledge_graph(
        self,
        graph_name: str,
        document_ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        prompt_overrides: dict[str, Any] | None = None,
    ) -> bool:
        """Update an existing knowledge graph.

        Args:
            graph_name: Name of the graph
            document_ids: Optional additional document IDs
            filters: Optional additional metadata filters
            prompt_overrides: Optional prompt customization

        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.error("Morphik is not available")
            return False

        try:
            # Process prompt overrides similar to create method
            graph_prompt_overrides = None
            if prompt_overrides:
                # Processing similar to create_knowledge_graph
                # This code is duplicated to maintain clarity
                entity_extraction = None
                if "entity_extraction" in prompt_overrides:
                    ext_data = prompt_overrides["entity_extraction"]
                    examples = []

                    if "examples" in ext_data:
                        for example in ext_data["examples"]:
                            examples.append(
                                EntityExtractionExample(
                                    label=example["label"],
                                    entity_type=example["type"],
                                    properties=example.get("properties"),
                                ),
                            )

                    entity_extraction = EntityExtractionPromptOverride(
                        prompt=ext_data.get("prompt"),
                        examples=examples if examples else None,
                    )

                entity_resolution = None
                if "entity_resolution" in prompt_overrides:
                    res_data = prompt_overrides["entity_resolution"]
                    examples = []

                    if "examples" in res_data:
                        for example in res_data["examples"]:
                            examples.append(
                                EntityResolutionExample(
                                    canonical=example["canonical"],
                                    variants=example["variants"],
                                ),
                            )

                    entity_resolution = EntityResolutionPromptOverride(
                        prompt=res_data.get("prompt"),
                        examples=examples if examples else None,
                    )

                if entity_extraction or entity_resolution:
                    graph_prompt_overrides = GraphPromptOverrides(
                        entity_extraction=entity_extraction,
                        entity_resolution=entity_resolution,
                    )

            # Update the knowledge graph
            graph = self._client.update_graph(
                name=graph_name,
                additional_documents=document_ids,
                additional_filters=filters,
                prompt_overrides=graph_prompt_overrides,
            )

            logger.info(f"Updated knowledge graph '{graph_name}' with {len(getattr(graph, 'entities', []))} entities")
            return True
        except Exception as e:
            logger.error(f"Failed to update knowledge graph: {e}")
            return False

    def get_knowledge_graphs(self) -> list[str]:
        """Get all available knowledge graphs.

        Returns:
            List of knowledge graph names
        """
        if not self.is_available():
            logger.error("Morphik is not available")
            return []

        try:
            # List all graphs
            graphs = self._client.list_graphs()

            # Extract names
            return [graph.name for graph in graphs]
        except Exception as e:
            logger.error(f"Failed to get knowledge graphs: {e}")
            return []

    def delete_knowledge_graph(self, graph_name: str) -> bool:
        """Delete a knowledge graph.

        Args:
            graph_name: Name of the graph

        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.error("Morphik is not available")
            return False

        try:
            # Delete the graph
            self._client.delete_graph(graph_name)
            logger.info(f"Deleted knowledge graph '{graph_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete knowledge graph: {e}")
            return False

    def add_file(
        self,
        file_path: str,
        metadata: dict[str, Any] | None = None,
        folder_id: str | None = None,
    ) -> str:
        """Add a file to Morphik.

        Args:
            file_path: Path to the file
            metadata: Optional metadata
            folder_id: Optional folder ID

        Returns:
            Document ID
        """
        if not self.is_available():
            logger.error("Morphik is not available")
            return ""

        try:
            # Use default folder if not specified
            target_folder = folder_id or self.default_folder

            # Add file
            doc = self._client.ingest_file(
                file_path=file_path,
                metadata=metadata or {},
                folder=target_folder,
            )
            logger.info(f"Added file with ID: {doc.id}")
            return doc.id
        except Exception as e:
            logger.error(f"Failed to add file: {e}")
            return ""

    def add_knowledge_graph(
        self,
        graph_document: dict[str, Any],
        user_id: str | None = None,
        folder: str | None = None,
    ) -> str | None:
        """Add a knowledge graph to Morphik.

        This is a lower-level method for adding a pre-structured knowledge graph.
        For most use cases, prefer using create_knowledge_graph.

        Args:
            graph_document: Knowledge graph document.
            user_id: Optional user ID. Uses default if not provided.
            folder: Optional folder. Uses default if not provided.

        Returns:
            Graph ID if successful, None otherwise.
        """
        if not self.is_available() or not hasattr(self._client, "add_knowledge_graph"):
            logger.error("Morphik knowledge graph functions not available")
            return None

        user = user_id or self.default_user
        folder_name = folder or self.default_folder

        try:
            # Add knowledge graph to Morphik
            result = self._client.add_knowledge_graph(
                graph=graph_document,
                user=user,
                folder=folder_name,
            )

            return result.get("id") if isinstance(result, dict) else result
        except Exception as e:
            logger.error(f"Failed to add knowledge graph to Morphik: {str(e)}")
            return None

    def get_folders(self, user_id: str | None = None) -> list[str]:
        """Get available folders for a user.

        Args:
            user_id: Optional user ID. Uses default if not provided.

        Returns:
            List of folder names.
        """
        if not self.is_available():
            logger.error("Morphik not available")
            return []

        user = user_id or self.default_user

        try:
            # Get folders from Morphik
            if hasattr(self._client, "get_folders"):
                result = self._client.get_folders(user=user)
            else:
                # Fallback to standard morphik API
                result = self._client.get_folders(user=user)

            return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"Failed to get folders from Morphik: {str(e)}")
            return []

    def create_folder(self, folder_name: str) -> str:
        """Create a folder in Morphik.

        Args:
            folder_name: Name of the folder

        Returns:
            Folder ID
        """
        if not self.is_available():
            logger.error("Morphik is not available")
            return ""

        try:
            # Check if folder exists
            folders = self._client.list_folders()
            for folder in folders:
                if folder.name == folder_name:
                    logger.info(f"Folder {folder_name} already exists with ID: {folder.id}")
                    return folder.id

            # Create folder
            folder = self._client.create_folder(folder_name)
            logger.info(f"Created folder {folder_name} with ID: {folder.id}")
            return folder.id
        except Exception as e:
            logger.error(f"Failed to create folder: {e}")
            return ""

    def delete_folder(
        self,
        folder_name: str,
        user_id: str | None = None,
    ) -> bool:
        """Delete a folder from Morphik.

        Args:
            folder_name: Name of the folder to delete.
            user_id: Optional user ID. Uses default if not provided.

        Returns:
            True if successful, False otherwise.
        """
        if not self.is_available():
            logger.error("Morphik not available")
            return False

        user = user_id or self.default_user

        try:
            # Delete folder from Morphik
            if hasattr(self._client, "delete_folder"):
                self._client.delete_folder(
                    folder=folder_name,
                    user=user,
                )
            else:
                # Fallback to standard morphik API
                self._client.delete_folder(
                    folder=folder_name,
                    user=user,
                )

            return True
        except Exception as e:
            logger.error(f"Failed to delete folder from Morphik: {str(e)}")
            return False

    def add_rule(
        self,
        rule_name: str,
        rule_content: str,
        user_id: str | None = None,
    ) -> bool:
        """Add a rule to Morphik.

        Args:
            rule_name: Name of the rule.
            rule_content: Content of the rule.
            user_id: Optional user ID. Uses default if not provided.

        Returns:
            True if successful, False otherwise.
        """
        if not self.is_available() or not hasattr(self._client, "add_rule"):
            logger.error("Morphik rule functions not available")
            return False

        user = user_id or self.default_user

        try:
            # Add rule to Morphik
            self._client.add_rule(
                name=rule_name,
                rule=rule_content,
                user=user,
            )

            return True
        except Exception as e:
            logger.error(f"Failed to add rule to Morphik: {str(e)}")
            return False

    def query_with_mcp(
        self,
        query_text: str,
        user_id: str | None = None,
        folder: str | None = None,
        rule: str | None = None,
        graph_name: str | None = None,
        hop_depth: int | None = None,
        include_paths: bool = False,
    ) -> dict[str, Any]:
        """Query Morphik using the Model Context Protocol (MCP).

        Args:
            query_text: Query text.
            user_id: Optional user ID. Uses default if not provided.
            folder: Optional folder. Uses default if not provided.
            rule: Optional rule to apply to the query.
            graph_name: Optional knowledge graph name to use for the query.
            hop_depth: Optional number of hops for graph traversal.
            include_paths: Whether to include relationship paths in results.

        Returns:
            MCP-formatted query response.
        """
        if not self.is_available() or not hasattr(self._client, "query_with_mcp"):
            logger.error("Morphik MCP functions not available")
            return {"type": "error", "message": "Morphik MCP functions not available"}

        user = user_id or self.default_user
        folder_name = folder or self.default_folder

        try:
            # Prepare kwargs for query_with_mcp call
            kwargs = {
                "query": query_text,
                "user": user,
                "folder": folder_name,
            }

            if rule:
                kwargs["rule"] = rule

            if graph_name:
                kwargs["graph_name"] = graph_name

            if hop_depth:
                kwargs["hop_depth"] = hop_depth

            if include_paths:
                kwargs["include_paths"] = include_paths

            # Query Morphik using MCP
            result = self._client.query_with_mcp(**kwargs)

            return result
        except Exception as e:
            logger.error(f"Failed to query Morphik with MCP: {str(e)}")
            return {"type": "error", "message": f"Failed to query Morphik with MCP: {str(e)}"}
