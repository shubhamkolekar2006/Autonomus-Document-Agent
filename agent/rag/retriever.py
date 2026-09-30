from typing import Dict, List, Optional
from agent.rag.chunker import chunk_sources
from agent.rag.embedder import EmbeddingModel, get_embedding_model
from agent.rag.vector_store import FAISSVectorStore


class RAGRetriever:
    """
    RAG Retriever coordinating the full retrieval pipeline:
    Source Documents -> Chunks -> Embeddings -> FAISS Index -> Query Similarity Search -> Top-K Chunks

    --- Key Architectural Principle ---
    FAISS similarity search computes mathematical proximity in vector space. It always returns the closest
    vectors in the index, even if the user query is about something not present in the document.
    We do NOT apply arbitrary score cutoffs (e.g. `score < 0.5`) at this stage, so that retrieval scores
    and behaviors can be inspected and evaluated transparently.
    """

    def __init__(
        self,
        sources: Optional[List[Dict]] = None,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        embedder: Optional[EmbeddingModel] = None
    ):
        """
        Initializes the retriever with optional source documents.

        Parameters:
        - sources: List of structured source dictionaries from document ingestion.
        - chunk_size: Number of words per chunk.
        - chunk_overlap: Number of overlapping words.
        - embedder: Optional EmbeddingModel instance (defaults to shared singleton).
        """
        self.sources = sources or []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedder = embedder or get_embedding_model()
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)
        self.chunks: List[Dict] = []

        # If sources are provided on initialization, build the index automatically
        if self.sources:
            self.build_index()

    def build_index(self):
        """
        Chunks all provided sources, generates dense embeddings, and indexes them in FAISS.
        """
        self.vector_store.clear()
        if not self.sources:
            self.chunks = []
            return

        # 1. Chunk all source documents into structured items
        self.chunks = chunk_sources(
            sources=self.sources,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap
        )

        if not self.chunks:
            return

        # 2. Extract texts and generate document embeddings
        texts = [chunk["text"] for chunk in self.chunks]
        embeddings = self.embedder.embed_documents(texts)

        # 3. Add chunks and embeddings to FAISS index with L2 normalization
        self.vector_store.add_chunks(self.chunks, embeddings)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieves the top-k most relevant chunks for a user query.

        Parameters:
        - query: The search query / question string.
        - top_k: Number of closest chunks to retrieve (default: 3).

        Returns:
        List of ranked chunk dictionaries with cosine similarity scores.
        """
        if not query or not query.strip() or self.vector_store.size() == 0:
            return []

        # 1. Embed query into the same vector space
        query_vec = self.embedder.embed_query(query)

        # 2. Search FAISS IndexFlatIP (cosine similarity)
        results = self.vector_store.search(query_vec, top_k=top_k)
        return results

    def is_indexed(self) -> bool:
        """Returns True if the retriever has indexed source chunks."""
        return self.vector_store.size() > 0


def format_retrieval_debug(query: str, results: List[Dict]) -> str:
    """
    Formats retrieval results into a clean, human-readable string for debugging and inspection:

    QUERY:
    ...

    RESULT 1
    Source: ...
    Chunk ID: ...
    Similarity score: ...
    Text: ...
    """
    lines = [f"QUERY:\n{query}\n"]

    if not results:
        lines.append("No results retrieved (index empty or top_k = 0).\n")
        return "\n".join(lines)

    for i, res in enumerate(results, 1):
        lines.append(f"RESULT {i}")
        lines.append(f"Source: {res.get('source_name', 'unknown')} ({res.get('source_type', 'unknown')})")
        lines.append(f"Chunk ID: {res.get('chunk_id', '-')}")
        lines.append(f"Similarity score: {res.get('score', 0.0):.4f}")
        lines.append(f"Text:\n{res.get('text', '')}\n")

    return "\n".join(lines)


def build_retrieved_context(retrieved_chunks: List[Dict]) -> str:
    """
    Converts retrieval results into structured, LLM-readable source context with clear metadata:

    RETRIEVED SOURCE CONTEXT

    [Source: company_info.txt | Chunk: 0 | Similarity: 0.7311]
    CareFlow uses FastAPI for backend development and PostgreSQL for database management.

    [Source: roadmap.pdf | Chunk: 4 | Similarity: 0.6842]
    The planned development timeline is six months.
    """
    if not retrieved_chunks:
        return ""

    blocks = ["RETRIEVED SOURCE CONTEXT\n"]
    for chunk in retrieved_chunks:
        src = chunk.get("source_name", "unknown")
        cid = chunk.get("chunk_id", "-")
        score = chunk.get("score", 0.0)
        text = chunk.get("text", "").strip()
        blocks.append(f"[Source: {src} | Chunk: {cid} | Similarity: {score:.4f}]\n{text}\n")

    return "\n".join(blocks).strip()

