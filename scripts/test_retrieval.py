"""
Development and manual verification script for RAG retrieval pipeline:
Demonstrates:
1. Ingesting sample knowledge sources
2. Word-based chunking with overlap
3. Dense embedding generation (all-MiniLM-L6-v2)
4. FAISS IndexFlatIP indexing with L2 normalization (cosine similarity)
5. Query similarity search and formatted debug inspection
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.rag.retriever import RAGRetriever, format_retrieval_debug


def main():
    print("=" * 60)
    print("  RAG RETRIEVAL PIPELINE - MANUAL TEST & INSPECTION")
    print("=" * 60)

    # 1. Define sample knowledge source
    sample_knowledge = (
        "NovaCare Technologies develops CareFlow, an AI healthcare platform. "
        "CareFlow uses FastAPI for backend development and high-throughput asynchronous services. "
        "The primary database is PostgreSQL for relational data and patient records. "
        "The frontend is built using React with TypeScript and Tailwind CSS. "
        "The application is deployed on AWS using ECS Fargate and CloudFront. "
        "The planned development timeline is six months, targeting Q4 2026 for beta launch."
    )

    sources = [
        {
            "source_name": "company_overview.txt",
            "source_type": "txt",
            "content": sample_knowledge
        }
    ]

    print("\n[Step 1] Initializing RAG Retriever & Building FAISS Vector Index...")
    retriever = RAGRetriever(sources=sources, chunk_size=30, chunk_overlap=10)
    print(f"Index built successfully! Total chunks indexed: {retriever.vector_store.size()}")

    # 2. Test Queries to evaluate retrieval accuracy
    test_queries = [
        "What backend technology does CareFlow use?",
        "What database does CareFlow use?",
        "What frontend technology does CareFlow use?",
        "How long is the development timeline?",
        "What is the CEO's name?"  # Test for information absent from knowledge base
    ]

    print("\n[Step 2] Executing Similarity Search Queries on FAISS Index...\n")

    for idx, query in enumerate(test_queries, 1):
        print(f"--- Query {idx} ---")
        results = retriever.retrieve(query=query, top_k=2)
        debug_output = format_retrieval_debug(query=query, results=results)
        print(debug_output)
        print("-" * 50)

    print("\n[Educational Note on Query 5 ('CEO's name')]:")
    print(
        "Notice that FAISS still returns the closest mathematical vectors even though the CEO's name "
        "does not exist in the source document. FAISS similarity search measures geometric closeness, "
        "not absence detection. In future stages, we will evaluate relevance thresholds and LLM context checks."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
