# AI Document Intelligence

An enterprise-grade document intelligence platform powered by **FastAPI**, **PyMuPDF**, **Tesseract OCR**, **FAISS**, and **OpenAI**. Automatically ingest dense documents (government forms, insurance policies, loan agreements, legal contracts), extract key fields with strict JSON schemas, and perform grounded, citation-backed question answering via RAG.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Frontend (index.html)     |
                      |  (Newsreader / Paper Theme) |
                      +--------------+--------------+
                                     |
                          HTTP / Multipart Form
                                     v
                      +-----------------------------+
                      |       FastAPI Backend       |
                      |          (main.py)          |
                      +-------+--------------+------+
                              |              |
              PDF Text / OCR  |              | Semantic Search
                              v              v
               +------------------+    +------------------+
               |  parser.py /     |    |     rag.py       |
               |  ocr.py          |    |   FAISS Index    |
               | (PyMuPDF / OCR)  |    | (OpenAI Embed.)  |
               +--------+---------+    +--------+---------+
                        |                       |
                        +-----------+-----------+
                                    |
                                    v
                         +--------------------+
                         |       ai.py        |
                         | OpenAI Extraction  |
                         | & Grounded QA      |
                         +--------------------+
```

---

## Directory Structure

```
docintel/
├── backend/
│   ├── __init__.py
│   ├── config.py           # Environment config & OpenAI validation
│   ├── parser.py           # PyMuPDF extraction with OCR fallback
│   ├── ocr.py              # pdf2image + pytesseract OCR pipeline
│   ├── rag.py              # FAISS indexing & semantic chunk search
│   ├── ai.py               # AI JSON extraction & grounded Q&A
│   ├── main.py             # FastAPI REST endpoints & CORS
│   ├── requirements.txt    # Python dependencies
│   └── .env.example        # Environment variable template
├── documents/              # Storage directory for uploaded documents
├── frontend/
│   └── index.html          # DocIntel web application UI
├── index.html              # Standalone web app entry point
└── README.md               # Documentation & setup guide
```

---

## Getting Started

### 1. Prerequisites

- Python 3.10+ installed
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (optional, for scanned PDF OCR fallback)
- [Poppler](https://poppler.freedesktop.org/) (optional, required if using `pdf2image` for OCR)

### 2. Environment Setup

Create and activate a virtual environment:

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

Install backend dependencies:

```bash
pip install -r backend/requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env` in the backend (or root) directory:

```bash
# Windows (PowerShell)
Copy-Item backend\.env.example .env

# Linux / macOS
cp backend/.env.example .env
```

Edit `.env` with your OpenAI API key:

```env
OPENAI_API_KEY=sk-proj-your_actual_openai_api_key_here
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### 4. Run the Backend Server

Start the FastAPI application with Uvicorn:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API Docs (Swagger UI)**: `http://localhost:8000/docs`
- **DocIntel Web Application**: `http://localhost:8000/app`
- **Health Check**: `http://localhost:8000/`

### 5. Run the Automated Test Suite

Run the full end-to-end integration and unit tests:

```bash
# Using venv python
.\.venv\Scripts\python.exe tests\test_backend.py

# Or with pytest
pytest tests/test_backend.py
```

---

## Frontend Integration

The provided frontend UI (`index.html`) is integrated directly with the backend:
1. Open `http://localhost:8000/app` in any browser, or simply double-click `index.html`.
2. Drag and drop any PDF into the upload dropzone.
3. The file is uploaded to `POST /upload`, processed with PyMuPDF, indexed in FAISS, and analyzed by OpenAI.
4. The UI displays the extracted **Document Type**, **Plain-Language Summary**, **Deadlines**, **Fees**, **Eligibility Criteria**, **Clauses**, **Actions**, and **Risks** with page-level badges.
5. Use the interactive Q&A search box to query the document — every answer cites specific source pages.

---

## API Endpoints & cURL Testing

### 1. Health Check (`GET /`)

```bash
curl -X GET http://localhost:8000/
```

**Response:**
```json
{
  "message": "AI Document Intelligence API is running"
}
```

---

### 2. Upload Document (`POST /upload`)

Upload a PDF file (max 20MB):

```bash
curl -X POST http://localhost:8000/upload \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_policy.pdf"
```

**Response:**
```json
{
  "document_id": "4a71b48b-3e5e-4c59-b1d5-21d1b9187391",
  "filename": "sample_policy.pdf",
  "pages": 4,
  "analysis": {
    "document_type": "Scholarship Program Notice",
    "summary": "Official notice outlining applicant eligibility, deadlines, and verification fees.",
    "eligibility": [
      {
        "requirement": "Family income below ₹2,50,000 per annum",
        "page": 2
      },
      {
        "requirement": "Currently enrolled in a recognized undergraduate program",
        "page": 2
      }
    ],
    "deadlines": [
      {
        "date": "15 March 2026",
        "description": "Final submission of completed application and certificates",
        "page": 2
      }
    ],
    "fees": [
      {
        "amount": "₹500",
        "description": "Non-refundable processing fee for submissions received after initial verification round",
        "page": 3
      }
    ],
    "required_documents": [
      {
        "document": "Income certificate",
        "page": 2
      },
      {
        "document": "Enrollment certificate",
        "page": 2
      }
    ],
    "important_clauses": [
      {
        "clause": "Late fee clause",
        "explanation": "Submissions after initial round incur late charge",
        "page": 3
      }
    ],
    "recommended_actions": [
      {
        "action": "Obtain certificates and submit prior to initial deadline",
        "reason": "Avoid non-refundable processing fee",
        "page": 2
      }
    ],
    "potential_risks": [
      {
        "risk": "Application rejection",
        "explanation": "Submissions after 15 March 2026 will not be accepted under any circumstances",
        "page": 2
      }
    ]
  }
}
```

---

### 3. Ask a Grounded Question (`POST /ask`)

Ask a question grounded strictly in the uploaded document's indexed text:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "4a71b48b-3e5e-4c59-b1d5-21d1b9187391",
    "question": "What is the deadline and is there any fee?"
  }'
```

**Response:**
```json
{
  "answer": "The application must be submitted no later than 15 March 2026. A non-refundable processing fee of ₹500 applies to all submissions received after the initial verification round.",
  "sources": [
    {
      "page": 2,
      "reason": "Specifies submission deadline of 15 March 2026"
    },
    {
      "page": 3,
      "reason": "Outlines ₹500 late processing fee schedule"
    }
  ]
}
```

---

### 4. Download / View Original Document (`GET /documents/{document_id}/file`)

```bash
curl -X GET http://localhost:8000/documents/4a71b48b-3e5e-4c59-b1d5-21d1b9187391/file -o downloaded_doc.pdf
```
