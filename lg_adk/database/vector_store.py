import faiss
import numpy as np


class VectorStore:
    """A simple vector store using FAISS for similarity search.

    Attributes:
        index: FAISS index for vector storage and search.
        vectors: List of stored vectors.
    """

    def __init__(self, dimension: int) -> None:
        """Initialize the vector store with a given dimension.

        Args:
            dimension: The dimensionality of the vectors.
        """
        self.index = faiss.IndexFlatL2(dimension)
        self.vectors: list[np.ndarray] = []

    def add(self, vector: np.ndarray) -> None:
        """Add a vector to the store.

        Args:
            vector: The vector to add.
        """
        self.index.add(np.array([vector]))
        self.vectors.append(vector)

    def search(self, query_vector: np.ndarray, k: int = 5) -> np.ndarray:
        """Search for the k most similar vectors to the query vector.

        Args:
            query_vector: The query vector.
            k: Number of nearest neighbors to return.

        Returns:
            Indices of the top-k most similar vectors.
        """
        distances, indices = self.index.search(np.array([query_vector]), k)
        return indices[0]
