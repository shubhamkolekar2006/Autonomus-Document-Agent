import os
from typing import List, Optional, Union
import numpy as np


class EmbeddingModel:
    """
    Embedding component using local SentenceTransformers ('all-MiniLM-L6-v2').

    --- Conceptual Explanation of Embeddings ---
    1. Document Embedding:
       Transforms a raw text chunk into a 384-dimensional dense numerical vector:
       chunk text -> vector in R^384

    2. Query Embedding:
       Transforms a user query into a vector in the EXACT same 384-dimensional vector space:
       user query -> vector in R^384

    Because both document chunks and queries are projected into the same semantic vector space,
    their semantic similarity can be computed via geometric vector operations (dot product on normalized vectors).
    """

    _instance: Optional["EmbeddingModel"] = None

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes and loads the SentenceTransformer model.
        Loads the model once in memory to avoid repeated loading overhead.
        """
        self.model_name = model_name
        self._model = None
        self._dimension = 384

    @property
    def model(self):
        """Lazy loader for the SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading local embedding model: '{self.model_name}'...")
            self._model = SentenceTransformer(self.model_name)
            # Verify embedding dimension
            test_emb = self._model.encode("test", show_progress_bar=False)
            self._dimension = len(test_emb)
        return self._model

    @property
    def dimension(self) -> int:
        """Returns the vector dimensionality of this embedding model (384 for all-MiniLM-L6-v2)."""
        return self._dimension

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Generates dense vector embeddings for a list of document chunk texts.

        Parameters:
        - texts: List of chunk text strings.

        Returns:
        - numpy.ndarray of shape (len(texts), dimension) with dtype float32.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False  # Normalization will be explicitly handled by FAISS L2 normalization
        )
        return np.ascontiguousarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generates a dense vector embedding for a single user query string.

        Parameters:
        - query: User question or search query string.

        Returns:
        - numpy.ndarray of shape (1, dimension) with dtype float32.
        """
        if not query or not query.strip():
            # Return zero vector if empty query
            return np.zeros((1, self.dimension), dtype=np.float32)

        embedding = self.model.encode(
            [query.strip()],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False
        )
        return np.ascontiguousarray(embedding, dtype=np.float32)


# Global helper to reuse the singleton embedding model across requests
_global_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingModel:
    """
    Returns a shared singleton instance of the EmbeddingModel to prevent reloading model weights.
    """
    global _global_embedding_model
    if _global_embedding_model is None or _global_embedding_model.model_name != model_name:
        _global_embedding_model = EmbeddingModel(model_name=model_name)
    return _global_embedding_model
