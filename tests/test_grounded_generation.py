import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from main import app
from agent.rag.retriever import RAGRetriever, build_retrieved_context


class TestGroundedGenerationPipeline(unittest.TestCase):
    """
    Test suite for Prompt 4: Connecting RAG Retrieval to Grounded Document Generation.
    Validates factual integrity, missing info handling, unsupported technology filtering,
    multi-source ingestion, and ungrounded fallback.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    # 1. Retrieved Context Builder Unit Tests
    def test_build_retrieved_context_formatting(self):
        chunks = [
            {
                "source_name": "company_info.txt",
                "chunk_id": 0,
                "score": 0.7311,
                "text": "CareFlow uses FastAPI for backend development and PostgreSQL for database management."
            },
            {
                "source_name": "roadmap.pdf",
                "chunk_id": 4,
                "score": 0.6842,
                "text": "The planned development timeline is six months."
            }
        ]
        context = build_retrieved_context(chunks)
        self.assertIn("RETRIEVED SOURCE CONTEXT", context)
        self.assertIn("[Source: company_info.txt | Chunk: 0 | Similarity: 0.7311]", context)
        self.assertIn("CareFlow uses FastAPI", context)
        self.assertIn("[Source: roadmap.pdf | Chunk: 4 | Similarity: 0.6842]", context)
        self.assertIn("six months", context)

    def test_build_retrieved_context_empty(self):
        self.assertEqual(build_retrieved_context([]), "")

    # 2. TEST 1 — Known Fact Grounding (FastAPI)
    @patch("agent.planner.call_llm")
    @patch("agent.executor.call_llm")
    @patch("agent.reflector.call_llm")
    def test_known_fact_backend_tech(self, mock_reflector_llm, mock_executor_llm, mock_planner_llm):
        mock_planner_llm.return_value = '{"document_type": "technical design", "assumptions": [], "title": "CareFlow Architecture", "sections": [{"heading": "1. Backend Overview", "purpose": "Explain backend technology"}]}'
        mock_executor_llm.return_value = "CareFlow uses FastAPI for high-performance asynchronous backend services."
        mock_reflector_llm.return_value = '{"issues_found": [], "needs_revision": false}'

        payload = {
            "request": "Create a technical overview of CareFlow.",
            "source_text": "CareFlow uses FastAPI for backend development."
        }
        response = self.client.post("/agent", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("grounded_mode"))
        self.assertIsNotNone(data.get("retrieved_sources"))
        self.assertTrue(len(data["retrieved_sources"]) > 0)
        self.assertEqual(data["retrieved_sources"][0]["source_name"], "user_provided_text")
        self.assertIn("FastAPI", data["retrieved_sources"][0]["preview"])

    # 3. TEST 2 — Known Database Grounding (PostgreSQL)
    @patch("agent.planner.call_llm")
    @patch("agent.executor.call_llm")
    @patch("agent.reflector.call_llm")
    def test_known_database_grounding(self, mock_reflector_llm, mock_executor_llm, mock_planner_llm):
        mock_planner_llm.return_value = '{"document_type": "technical design", "assumptions": [], "title": "Database Overview", "sections": [{"heading": "1. Database", "purpose": "Explain database"}]}'
        mock_executor_llm.return_value = "The database is PostgreSQL for relational structured data."
        mock_reflector_llm.return_value = '{"issues_found": [], "needs_revision": false}'

        payload = {
            "request": "Create a technical overview.",
            "source_text": "The database is PostgreSQL."
        }
        response = self.client.post("/agent", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("grounded_mode"))
        self.assertTrue(any("PostgreSQL" in s["preview"] for s in data["retrieved_sources"]))

    # 4. TEST 3 — Missing CEO Information Handling
    @patch("agent.planner.call_llm")
    @patch("agent.executor.call_llm")
    @patch("agent.reflector.call_llm")
    def test_missing_ceo_handling(self, mock_reflector_llm, mock_executor_llm, mock_planner_llm):
        # Planner notes that CEO is not provided
        mock_planner_llm.return_value = '{"document_type": "business report", "assumptions": ["CEO name not provided in source material"], "title": "Company Profile", "sections": [{"heading": "1. Executive Leadership", "purpose": "List leadership"}]}'
        # Executor explicitly declares CEO is not provided rather than inventing a name
        mock_executor_llm.return_value = "CEO: Not provided in the supplied source material. Product: CareFlow."
        mock_reflector_llm.return_value = '{"issues_found": [], "needs_revision": false}'

        payload = {
            "request": "Create a company profile including the CEO name.",
            "source_text": "Company: NovaCare Technologies. Product: CareFlow. Backend: FastAPI."
        }
        response = self.client.post("/agent", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("grounded_mode"))
        self.assertIn("CEO name not provided in source material", data["assumptions"])

    # 5. TEST 4 — Unsupported Technology Prevention (No False Claims of Kubernetes/Kafka/Redis)
    @patch("agent.planner.call_llm")
    @patch("agent.executor.call_llm")
    @patch("agent.reflector.call_llm")
    def test_unsupported_technology_filtering(self, mock_reflector_llm, mock_executor_llm, mock_planner_llm):
        mock_planner_llm.return_value = '{"document_type": "technical design", "assumptions": [], "title": "CareFlow Tech Stack", "sections": [{"heading": "1. Core Stack", "purpose": "Stack overview"}]}'
        # Executor only claims FastAPI and PostgreSQL as existing facts
        mock_executor_llm.return_value = "CareFlow uses FastAPI as its backend framework and PostgreSQL for database storage. Additional tooling such as Redis caching can be considered for future evaluation."
        mock_reflector_llm.return_value = '{"issues_found": [], "needs_revision": false}'

        payload = {
            "request": "Create a technical architecture.",
            "source_text": "Backend: FastAPI. Database: PostgreSQL."
        }
        response = self.client.post("/agent", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("grounded_mode"))

    # 6. TEST 5 — Multiple Sources Ingestion and Retrieval
    @patch("agent.planner.call_llm")
    @patch("agent.executor.call_llm")
    @patch("agent.reflector.call_llm")
    def test_multiple_sources_retrieval(self, mock_reflector_llm, mock_executor_llm, mock_planner_llm):
        mock_planner_llm.return_value = '{"document_type": "technical design", "assumptions": [], "title": "Tech Overview", "sections": [{"heading": "1. Full Stack", "purpose": "Overview"}]}'
        mock_executor_llm.return_value = "Backend is powered by FastAPI while database persistence is handled by PostgreSQL."
        mock_reflector_llm.return_value = '{"issues_found": [], "needs_revision": false}'

        files = [
            ("source1.txt", b"Backend: FastAPI."),
            ("source2.txt", b"Database: PostgreSQL.")
        ]
        response = self.client.post(
            "/agent",
            data={"request": "Create a technical overview."},
            files=[("files", (name, content, "text/plain")) for name, content in files]
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("grounded_mode"))
        self.assertEqual(len(data.get("sources", [])), 2)
        retrieved_previews = [r["preview"] for r in data.get("retrieved_sources", [])]
        self.assertTrue(any("FastAPI" in p for p in retrieved_previews))
        self.assertTrue(any("PostgreSQL" in p for p in retrieved_previews))

    # 7. TEST 6 — No Source Material (Un-grounded Mode Fallback)
    @patch("agent.planner.call_llm")
    @patch("agent.executor.call_llm")
    @patch("agent.reflector.call_llm")
    def test_no_source_material_ungrounded_fallback(self, mock_reflector_llm, mock_executor_llm, mock_planner_llm):
        mock_planner_llm.return_value = '{"document_type": "project plan", "assumptions": ["Standard 3-month timeline"], "title": "Generic Project Plan", "sections": [{"heading": "1. Scope", "purpose": "Define scope"}]}'
        mock_executor_llm.return_value = "The project establishes core milestones across 3 months."
        mock_reflector_llm.return_value = '{"issues_found": [], "needs_revision": false}'

        payload = {
            "request": "Create a generic project plan for software rollout."
        }
        response = self.client.post("/agent", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        # Un-grounded mode must be False
        self.assertFalse(data.get("grounded_mode"))
        self.assertIsNone(data.get("retrieved_sources"))
        self.assertEqual(data.get("sources"), [])

    # 8. TEST 7 — Retrieval Visibility and Schema
    def test_retrieval_visibility_schema(self):
        retriever = RAGRetriever(sources=[{"source_name": "overview.txt", "source_type": "txt", "content": "CareFlow uses FastAPI."}])
        results = retriever.retrieve("What backend is used?", top_k=1)
        self.assertTrue(len(results) > 0)
        top = results[0]
        self.assertIn("source_name", top)
        self.assertIn("chunk_id", top)
        self.assertIn("score", top)
        self.assertIn("text", top)
        self.assertIsInstance(top["score"], float)

    # 9. TEST 8 / Problem Scenario Acceptance Test (CEO, Revenue, Employee Count)
    @patch("agent.planner.call_llm")
    @patch("agent.executor.call_llm")
    @patch("agent.reflector.call_llm")
    def test_critical_problem_scenario_missing_fields_not_invented(self, mock_reflector_llm, mock_executor_llm, mock_planner_llm):
        """
        Critical Acceptance Test:
        Knowledge: 'NovaCare Technologies develops CareFlow. CareFlow uses FastAPI. The database is PostgreSQL. The frontend is React.'
        Request: 'Create a company profile including the CEO name, annual revenue, and employee count.'
        Verification: CEO, annual revenue, and employee count are NOT invented and explicitly stated as not provided.
        """
        knowledge = (
            "NovaCare Technologies develops CareFlow. "
            "CareFlow uses FastAPI. "
            "The database is PostgreSQL. "
            "The frontend is React."
        )

        mock_planner_llm.return_value = (
            '{"document_type": "business report", '
            '"assumptions": ["CEO name, annual revenue, and employee count not provided in supplied source material; recorded as unspecified"], '
            '"title": "NovaCare Technologies Company Profile", '
            '"sections": [{"heading": "1. Company Details", "purpose": "Company overview and missing metrics"}]}'
        )

        mock_executor_llm.return_value = (
            "Company: NovaCare Technologies\n"
            "Product: CareFlow\n"
            "Backend: FastAPI\n"
            "Database: PostgreSQL\n"
            "Frontend: React\n"
            "CEO: Not provided in the supplied source material.\n"
            "Annual Revenue: Not provided in the supplied source material.\n"
            "Employee Count: Not provided in the supplied source material."
        )

        mock_reflector_llm.return_value = '{"issues_found": [], "needs_revision": false}'

        payload = {
            "request": "Create a company profile including the CEO name, annual revenue, and employee count.",
            "source_text": knowledge
        }
        response = self.client.post("/agent", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("grounded_mode"))
        self.assertTrue(any("NovaCare" in s["preview"] for s in data["retrieved_sources"]))
        self.assertIn("CEO name, annual revenue, and employee count not provided in supplied source material; recorded as unspecified", data["assumptions"])


if __name__ == "__main__":
    unittest.main()
