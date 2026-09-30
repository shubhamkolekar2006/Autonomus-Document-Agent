"""
RAG (Retrieval-Augmented Generation) package:
Contains standalone chunking, embedding, vector store indexing (FAISS), and semantic retrieval components.
"""

from agent.rag.chunker import create_chunks, chunk_sources
from agent.rag.embedder import EmbeddingModel, get_embedding_model
from agent.rag.vector_store import FAISSVectorStore
from agent.rag.retriever import RAGRetriever, format_retrieval_debug, build_retrieved_context

__all__ = [
    "create_chunks",
    "chunk_sources",
    "EmbeddingModel",
    "get_embedding_model",
    "FAISSVectorStore",
    "RAGRetriever",
    "format_retrieval_debug",
    "build_retrieved_context"
]
