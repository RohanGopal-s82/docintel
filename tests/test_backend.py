"""
End-to-end integration and unit test suite for AI Document Intelligence.
Tests configuration, chunking, citation grounding, PDF extraction, and FastAPI endpoints.
"""

import sys
from pathlib import Path
import unittest

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from backend.config import OPENAI_API_KEY, CHAT_MODEL, EMBEDDING_MODEL
from backend.parser import extract_pdf_pages, get_pdf_page_count
from backend.rag import create_chunks, create_index, search_document
from backend.ai import (
    validate_and_sanitize_citations,
    DocumentAnalysisSchema,
    QuestionAnswerSchema,
    _call_model,
)
from fastapi.testclient import TestClient
from backend.main import app, documents

try:
    import pymupdf as fitz
except ImportError:
    import fitz


class TestDocIntelBackend(unittest.TestCase):

    def test_01_config(self):
        self.assertIsNotNone(OPENAI_API_KEY)
        self.assertEqual(CHAT_MODEL, "gpt-4o-mini")
        self.assertEqual(EMBEDDING_MODEL, "text-embedding-3-small")

    def test_02_chunking(self):
        sample_pages = [
            {"page": 1, "text": "word " * 1500},
            {"page": 2, "text": "second page text with different content " * 100},
        ]
        chunks = create_chunks(sample_pages, chunk_size=1000, overlap=150)
        self.assertGreater(len(chunks), 2)
        self.assertEqual(chunks[0]["page"], 1)
        pages_found = {c["page"] for c in chunks}
        self.assertIn(1, pages_found)
        self.assertIn(2, pages_found)

    def test_03_citation_sanitization_out_of_bounds(self):
        # 2-page document
        pages = [
            {"page": 1, "text": "Annual fee is $500 payable on registration."},
            {"page": 2, "text": "Applicants must be at least 18 years old."},
        ]
        # AI hallucinated page 3 for the fee, and page 1 for eligibility (which is on page 2)
        hallucinated_analysis = {
            "document_type": "Application Notice",
            "summary": "Summary text",
            "fees": [{"amount": "$500", "description": "Annual fee", "page": 3}],
            "eligibility": [{"requirement": "must be at least 18 years old", "page": 1}],
            "deadlines": [],
            "required_documents": [],
            "important_clauses": [],
            "recommended_actions": [],
            "potential_risks": [],
        }

        sanitized = validate_and_sanitize_citations(hallucinated_analysis, pages)

        # The fee has $500 on Page 1 -> should be corrected from 3 to 1
        self.assertEqual(sanitized["fees"][0]["page"], 1)

        # Eligibility has "18 years old" on Page 2 -> should be moved from 1 to 2
        self.assertEqual(sanitized["eligibility"][0]["page"], 2)

    def test_04_pdf_generation_and_parsing(self):
        pdf_path = root_dir / "documents" / "test_sample.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        doc = fitz.open()
        p1 = doc.new_page()
        p1.insert_text((50, 72), "DocIntel System Specification Document. Page 1 contains details on system design.")
        p2 = doc.new_page()
        p2.insert_text((50, 72), "Eligibility requirements: Candidates must submit their application before October 31.")
        doc.save(str(pdf_path))
        doc.close()

        count = get_pdf_page_count(pdf_path)
        self.assertEqual(count, 2)

        extracted = extract_pdf_pages(pdf_path)
        self.assertEqual(len(extracted), 2)
        self.assertEqual(extracted[0]["page"], 1)
        self.assertIn("DocIntel", extracted[0]["text"])
        self.assertEqual(extracted[1]["page"], 2)
        self.assertIn("October 31", extracted[1]["text"])

    def test_05_fastapi_endpoints(self):
        client = TestClient(app)

        # 1. Health check
        res = client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"message": "AI Document Intelligence API is running"})

        # 2. Browser request for UI
        res_ui = client.get("/", headers={"accept": "text/html"})
        self.assertEqual(res_ui.status_code, 200)
        self.assertIn("DocIntel", res_ui.text)

        # 3. Status check
        res_status = client.get("/api/status")
        self.assertEqual(res_status.status_code, 200)
        data = res_status.json()
        self.assertEqual(data["status"], "online")

        # 4. Upload PDF
        pdf_path = root_dir / "documents" / "test_sample.pdf"
        with open(pdf_path, "rb") as f:
            res_upload = client.post(
                "/upload",
                files={"file": ("test_sample.pdf", f, "application/pdf")},
            )
        self.assertEqual(res_upload.status_code, 200)
        upload_data = res_upload.json()
        self.assertIn("document_id", upload_data)
        doc_id = upload_data["document_id"]
        self.assertEqual(upload_data["pages"], 2)
        self.assertIn("analysis", upload_data)

        # 5. Fetch uploaded file
        res_file = client.get(f"/documents/{doc_id}/file")
        self.assertEqual(res_file.status_code, 200)
        self.assertEqual(res_file.headers["content-type"], "application/pdf")

        # 6. Ask Question
        res_ask = client.post(
            "/ask",
            json={"document_id": doc_id, "question": "What is the deadline for application?"},
        )
        self.assertEqual(res_ask.status_code, 200)
        ask_data = res_ask.json()
        self.assertIn("answer", ask_data)
        self.assertIn("sources", ask_data)


if __name__ == "__main__":
    unittest.main()
