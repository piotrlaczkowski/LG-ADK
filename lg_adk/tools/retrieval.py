"""Retrieval Tools for LG-ADK. Tools for retrieving information from vector stores and knowledge bases."""

import time
from abc import abstractmethod
from typing import Any

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from lg_adk.utils.logging import get_logger

logger = get_logger(__name__)


class Document(BaseModel):
    """A document retrieved from a knowledge base."""

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None


class BaseRetrievalTool(BaseTool):
    """Base class for retrieval tools.

    A retrieval tool is used to search and retrieve documents from a knowledge base.
    """

    name: str = "base_retrieval_tool"
    description: str = Field("Base retrieval tool for knowledge base search.")
    top_k: int = 5
    score_threshold: float | None = None

    def __init__(
        self,
        name: str = None,
        description: str = None,
        top_k: int = 5,
        score_threshold: float | None = None,
        **kwargs,
    ):
        """Initialize the base retrieval tool.

        Args:
            name: Name of the tool.
            description: Description of the tool.
            top_k: Maximum number of documents to retrieve.
            score_threshold: Minimum similarity score for documents.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        init_kwargs = {}
        if name is not None:
            init_kwargs["name"] = name
        if description is not None:
            init_kwargs["description"] = description
        init_kwargs["top_k"] = top_k
        init_kwargs["score_threshold"] = score_threshold
        init_kwargs.update(kwargs)
        super().__init__(**init_kwargs)

    @abstractmethod
    def retrieve(self, query: str) -> list[Document]:
        """Retrieve documents relevant to the query.

        Args:
            query: The query to search for.

        Returns:
            List of retrieved documents.
        """
        pass

    def run(self, query: str) -> dict[str, Any]:
        """Run the retrieval tool on a query.

        Args:
            query: The query to search for.

        Returns:
            Dictionary containing the retrieved documents and metadata.
        """
        try:
            start_time = time.time()
            documents = self.retrieve(query)
            elapsed_time = time.time() - start_time

            # Filter by score threshold if provided
            if self.score_threshold is not None:
                documents = [doc for doc in documents if doc.score is None or doc.score >= self.score_threshold]

            # Limit to top_k
            documents = documents[: self.top_k]

            # Log retrieval
            logger.info(f"Retrieved {len(documents)} documents for query: {query}")

            # Format documents as strings
            formatted_docs = []
            for i, doc in enumerate(documents):
                formatted_doc = f"Document {i+1}"
                if doc.score is not None:
                    formatted_doc += f" (Score: {doc.score:.4f})"
                formatted_doc += f":\n{doc.content}\n"

                if doc.metadata:
                    formatted_doc += f"Metadata: {doc.metadata}\n"

                formatted_docs.append(formatted_doc)

            return {
                "documents": documents,
                "formatted_documents": formatted_docs,
                "num_documents": len(documents),
                "query": query,
                "elapsed_time": elapsed_time,
                "output": "\n\n".join(formatted_docs),
            }
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return {
                "error": str(e),
                "query": query,
                "output": f"Error retrieving documents: {str(e)}",
            }

    def get(self, key: str, default=None) -> Any | None:
        """Dictionary-like access to attributes for compatibility."""
        return getattr(self, key, default)

    def _run(self, query: str) -> dict[str, Any]:
        """Required by BaseTool: delegates to run."""
        return self.run(query)

    async def _arun(self, query: str) -> dict[str, Any]:
        """Required by BaseTool: delegates to retrieve asynchronously if implemented."""
        # If you have an async retrieve, use it; otherwise, fallback to sync
        if hasattr(self, "aretrieve") and callable(self.aretrieve):
            docs = await self.aretrieve(query)
        else:
            docs = self.retrieve(query)
        return {"documents": docs}


class SimpleVectorRetrievalTool(BaseRetrievalTool):
    """A simple vector retrieval tool using an in-memory vector store."""

    name: str = "simple_vector_retrieval"
    description: str = Field("A simple vector retrieval tool using an in-memory vector store.")
    vector_store: Any = None

    def __init__(
        self,
        name: str = None,
        description: str = None,
        vector_store: Any = None,
        top_k: int = 5,
        score_threshold: float | None = None,
        **kwargs,
    ):
        """Initialize the vector retrieval tool.

        Args:
            name: Name of the tool.
            description: Description of the tool.
            vector_store: The vector store to retrieve documents from.
            top_k: Maximum number of documents to retrieve.
            score_threshold: Minimum similarity score for documents.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        init_kwargs = {}
        if name is not None:
            init_kwargs["name"] = name
        if description is not None:
            init_kwargs["description"] = description
        if vector_store is not None:
            init_kwargs["vector_store"] = vector_store
        init_kwargs["top_k"] = top_k
        init_kwargs["score_threshold"] = score_threshold
        init_kwargs.update(kwargs)
        super().__init__(**init_kwargs)

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve documents from the vector store.

        Args:
            query: The query to search for.

        Returns:
            List of retrieved documents.
        """
        try:
            # Retrieve documents from the vector store
            if hasattr(self.vector_store, "similarity_search_with_score"):
                results = self.vector_store.similarity_search_with_score(query, k=self.top_k)
                return [
                    Document(
                        content=doc.page_content if hasattr(doc, "page_content") else str(doc),
                        metadata=doc.metadata if hasattr(doc, "metadata") else {},
                        score=score,
                    )
                    for doc, score in results
                ]
            elif hasattr(self.vector_store, "similarity_search"):
                results = self.vector_store.similarity_search(query, k=self.top_k)
                return [
                    Document(
                        content=doc.page_content if hasattr(doc, "page_content") else str(doc),
                        metadata=doc.metadata if hasattr(doc, "metadata") else {},
                    )
                    for doc in results
                ]
            else:
                raise ValueError("Vector store doesn't support similarity_search or similarity_search_with_score")
        except Exception as e:
            logger.error(f"Error retrieving from vector store: {str(e)}")
            return []


class ChromaDBRetrievalTool(BaseRetrievalTool):
    """Retrieval tool for ChromaDB."""

    def __init__(
        self,
        name: str,
        description: str,
        collection_name: str,
        chroma_client: Any,
        embedding_function: Any = None,
        top_k: int = 5,
        score_threshold: float | None = None,
    ):
        """Initialize the ChromaDB retrieval tool.

        Args:
            name: Name of the tool.
            description: Description of the tool.
            collection_name: Name of the ChromaDB collection.
            chroma_client: ChromaDB client.
            embedding_function: Function to convert text to embeddings.
            top_k: Number of documents to retrieve.
            score_threshold: Minimum similarity score for retrieved documents.
        """
        super().__init__(name, description, top_k, score_threshold)
        self.collection_name = collection_name
        self.chroma_client = chroma_client
        self.embedding_function = embedding_function

        # Get or create the collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve documents from ChromaDB.

        Args:
            query: The query to search for.

        Returns:
            List of retrieved documents.
        """
        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=self.top_k,
                include=["documents", "metadatas", "distances"],
            )

            # Extract and format results
            documents = []
            for i in range(len(results.get("documents", [[]])[0])):
                # Convert distance to similarity score (1 - distance)
                distance = results["distances"][0][i] if "distances" in results else None
                score = 1 - distance if distance is not None else None

                doc = Document(
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if "metadatas" in results else {},
                    score=score,
                )
                documents.append(doc)

            return documents
        except Exception as e:
            logger.error(f"Error retrieving from ChromaDB: {str(e)}")
            return []
