import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.vectors = []

    def add(self, vector: np.ndarray):
        self.index.add(np.array([vector]))
        self.vectors.append(vector)

    def search(self, query_vector: np.ndarray, k: int = 5):
        distances, indices = self.index.search(np.array([query_vector]), k)
        return indices[0]
