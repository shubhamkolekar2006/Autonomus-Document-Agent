import unittest
import numpy as np

from agent.rag.chunker import create_chunks, chunk_sources
from agent.rag.embedder import EmbeddingModel, get_embedding_model
from agent.rag.vector_store import FAISSVectorStore
from agent.rag.retriever import RAGRetriever, format_retrieval_debug


class TestRAGRetrievalPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Deterministic test dataset specified in Prompt 3
        cls.sample_text = (
            "NovaCare Technologies develops CareFlow, an AI healthcare platform. "
            "CareFlow uses FastAPI for backend development and high-throughput services. "
            "The primary database is PostgreSQL for relational patient records. "
            "CareFlow frontend is built using React with TypeScript. "
            "The application is deployed on AWS cloud infrastructure. "
            "The planned development timeline is six months, targeting Q4 2026."
        )

        cls.sources = [
            {
                "source_name": "careflow_spec.txt",
                "source_type": "txt",
                "content": cls.sample_text
            }
        ]

        cls.embedder = get_embedding_model()
        cls.retriever = RAGRetriever(
            sources=cls.sources,
            chunk_size=15,
            chunk_overlap=3,
            embedder=cls.embedder
        )

    # 1. Chunker Unit Tests
    def test_chunker_basic_and_overlap(self):
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        chunks = create_chunks(text, chunk_size=4, overlap=2)
        # Step = 4 - 2 = 2. Slices: [0:4], [2:6], [4:8], [6:10] -> 4 chunks
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0]["text"], "word1 word2 word3 word4")
        self.assertEqual(chunks[1]["text"], "word3 word4 word5 word6")
        self.assertEqual(chunks[0]["start_word"], 0)
        self.assertEqual(chunks[0]["end_word"], 4)

    def test_chunker_validations(self):
        with self.assertRaises(ValueError):
            create_chunks("test text", chunk_size=0)
        with self.assertRaises(ValueError):
            create_chunks("test text", chunk_size=10, overlap=-1)
        with self.assertRaises(ValueError):
            create_chunks("test text", chunk_size=10, overlap=10)

    def test_chunker_empty_input(self):
        self.assertEqual(create_chunks(""), [])
        self.assertEqual(create_chunks("   \n\t  "), [])

    def test_chunk_sources_multi_source_ids(self):
        sources = [
            {"source_name": "doc1.txt", "source_type": "txt", "content": "alpha beta gamma delta epsilon"},
            {"source_name": "doc2.md", "source_type": "md", "content": "one two three four five six seven"}
        ]
        all_chunks = chunk_sources(sources, chunk_size=3, overlap=1)
        self.assertTrue(len(all_chunks) > 0)
        # Verify sequential global chunk_ids
        for idx, chunk in enumerate(all_chunks):
            self.assertEqual(chunk["chunk_id"], idx)
        self.assertEqual(all_chunks[0]["source_name"], "doc1.txt")
        self.assertEqual(all_chunks[-1]["source_name"], "doc2.md")

    # 2. Embedder Unit Tests
    def test_embedder_dimension_and_type(self):
        emb = self.embedder.embed_query("FastAPI backend")
        self.assertEqual(emb.shape, (1, 384))
        self.assertEqual(emb.dtype, np.float32)

        docs_emb = self.embedder.embed_documents(["First document", "Second document"])
        self.assertEqual(docs_emb.shape, (2, 384))
        self.assertEqual(docs_emb.dtype, np.float32)

    # 3. Vector Store Unit Tests
    def test_vector_store_normalization_and_inner_product(self):
        store = FAISSVectorStore(dimension=384)
        chunks = [{"chunk_id": 0, "source_name": "test", "text": "Sample text"}]
        # Create un-normalized embedding
        raw_emb = np.random.randn(1, 384).astype(np.float32) * 5.0
        store.add_chunks(chunks, raw_emb)
        self.assertEqual(store.size(), 1)

        # Search with same vector
        results = store.search(raw_emb, top_k=1)
        self.assertEqual(len(results), 1)
        # Cosine similarity with self is 1.0 (within float precision)
        self.assertAlmostEqual(results[0]["score"], 1.0, places=3)

    # 4. Independent RAG Retrieval Tests on CareFlow Dataset
    def test_rag_query_1_backend_tech(self):
        """Query 1: What backend technology does CareFlow use? -> Must retrieve FastAPI."""
        query = "What backend technology does CareFlow use?"
        results = self.retriever.retrieve(query=query, top_k=1)
        self.assertTrue(len(results) > 0)
        top_chunk = results[0]
        self.assertIn("FastAPI", top_chunk["text"])
        self.assertTrue(top_chunk["score"] > 0.4)

    def test_rag_query_2_database(self):
        """Query 2: What database does CareFlow use? -> Must retrieve PostgreSQL."""
        query = "What database does CareFlow use?"
        results = self.retriever.retrieve(query=query, top_k=1)
        self.assertTrue(len(results) > 0)
        top_chunk = results[0]
        self.assertIn("PostgreSQL", top_chunk["text"])
        self.assertTrue(top_chunk["score"] > 0.4)

    def test_rag_query_3_frontend_tech(self):
        """Query 3: What frontend technology does CareFlow use? -> Must retrieve React in top chunks."""
        query = "What frontend technology does CareFlow use?"
        results = self.retriever.retrieve(query=query, top_k=2)
        self.assertTrue(len(results) > 0)
        self.assertTrue(any("React" in chunk["text"] for chunk in results))
        self.assertTrue(all(chunk["score"] > 0.4 for chunk in results))

    def test_rag_query_4_timeline(self):
        """Query 4: How long is the development timeline? -> Must retrieve six months."""
        query = "How long is the development timeline?"
        results = self.retriever.retrieve(query=query, top_k=1)
        self.assertTrue(len(results) > 0)
        top_chunk = results[0]
        self.assertIn("six months", top_chunk["text"])
        self.assertTrue(top_chunk["score"] > 0.4)

    def test_rag_query_5_ceo_absence_demonstration(self):
        """
        Query 5: What is the CEO's name?
        Demonstrates that FAISS returns the closest geometric vector without proving information exists.
        """
        query = "What is the CEO's name?"
        results = self.retriever.retrieve(query=query, top_k=1)
        self.assertTrue(len(results) > 0)
        top_chunk = results[0]
        # Text does not contain any CEO name
        self.assertNotIn("CEO", top_chunk["text"])
        # Score exists and is returned transparently
        self.assertIsInstance(top_chunk["score"], float)

    def test_empty_sources_retrieval(self):
        empty_retriever = RAGRetriever(sources=[])
        self.assertEqual(empty_retriever.retrieve("Any query"), [])

    def test_format_retrieval_debug(self):
        query = "Test query"
        mock_results = [
            {
                "chunk_id": 0,
                "source_name": "doc.txt",
                "source_type": "txt",
                "score": 0.8523,
                "text": "This is test chunk content."
            }
        ]
        debug_str = format_retrieval_debug(query, mock_results)
        self.assertIn("QUERY:\nTest query", debug_str)
        self.assertIn("RESULT 1", debug_str)
        self.assertIn("0.8523", debug_str)
        self.assertIn("doc.txt", debug_str)


if __name__ == "__main__":
    unittest.main()
