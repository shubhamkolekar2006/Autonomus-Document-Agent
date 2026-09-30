import faiss
import numpy as np
from typing import Dict, List, Optional


class FAISSVectorStore:
    """
    FAISS-based vector index for semantic similarity search.

    --- Mathematical & Educational Note on Cosine Similarity via Inner Product ---
    Cosine similarity between vector A and vector B is defined as:
        cosine_similarity(A, B) = (A . B) / (||A||_2 * ||B||_2)

    If vectors A and B are L2-normalized beforehand:
        ||A||_2 = 1 and ||B||_2 = 1

    Then the formula simplifies directly to the inner product (dot product):
        cosine_similarity(A, B) = A_norm . B_norm

    Therefore:
        faiss.normalize_L2(vectors) + faiss.IndexFlatIP(dimension)
    computes EXACT cosine similarity with maximum numerical efficiency and score interpretability
    where higher score (closer to 1.0) indicates higher semantic similarity.
    """

    def __init__(self, dimension: int = 384):
        """
        Initializes an empty FAISS IndexFlatIP (Inner Product) for normalized vectors.

        Parameters:
        - dimension: Vector dimensionality (384 for all-MiniLM-L6-v2).
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        # Separate list maintaining 1-to-1 position correspondence with FAISS index entries:
        # index position 0 -> metadata[0]
        # index position 1 -> metadata[1]
        self.metadata: List[Dict] = []

    def size(self) -> int:
        """Returns the total number of vectors currently stored in the index."""
        return self.index.ntotal

    def clear(self):
        """Resets the FAISS index and clears all chunk metadata."""
        self.index.reset()
        self.metadata.clear()

    def add_chunks(self, chunks: List[Dict], embeddings: np.ndarray):
        """
        L2-normalizes document embeddings, adds them to the FAISS IndexFlatIP,
        and stores the corresponding chunk metadata.

        Parameters:
        - chunks: List of chunk metadata dictionaries.
        - embeddings: 2D numpy array of shape (len(chunks), dimension) with dtype float32.
        """
        if not chunks:
            return

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Mismatch between number of chunks ({len(chunks)}) and embeddings rows ({embeddings.shape[0]})"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}"
            )

        # Ensure contiguous float32 array
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Apply L2 normalization in-place so inner product calculates cosine similarity
        faiss.normalize_L2(vectors)

        # Add vectors to FAISS index
        self.index.add(vectors)

        # Maintain exact 1-to-1 position mapping in metadata
        for chunk in chunks:
            self.metadata.append(dict(chunk))

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict]:
        """
        Searches the FAISS index for the top-k most similar chunks to the query vector.

        Parameters:
        - query_embedding: 2D numpy array of shape (1, dimension).
        - top_k: Maximum number of closest chunks to retrieve (default: 3).

        Returns:
        List of ranked result dictionaries with cosine similarity scores:
        [
            {
                "chunk_id": int,
                "source_name": str,
                "source_type": str,
                "score": float,  # Cosine similarity (higher is more similar)
                "text": str,
                "start_word": int,
                "end_word": int
            },
            ...
        ]
        """
        if self.size() == 0 or top_k <= 0:
            return []

        # Prepare query vector: ensure float32, copy, and normalize
        q_vec = np.ascontiguousarray(query_embedding, dtype=np.float32)
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)

        faiss.normalize_L2(q_vec)

        # Bound k by total stored items
        k = min(top_k, self.size())

        # Perform inner-product similarity search
        distances, indices = self.index.search(q_vec, k)

        results = []
        for i in range(k):
            idx = int(indices[0][i])
            if idx == -1:
                continue

            score = float(distances[0][i])
            meta = self.metadata[idx]

            result_item = {
                "chunk_id": meta.get("chunk_id", idx),
                "source_name": meta.get("source_name", "unknown"),
                "source_type": meta.get("source_type", "text"),
                "score": round(score, 4),
                "text": meta.get("text", ""),
                "start_word": meta.get("start_word", 0),
                "end_word": meta.get("end_word", 0)
            }
            results.append(result_item)

        return results
