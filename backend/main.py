"""
FastAPI Application Entry Point for AI Document Intelligence API.
Handles file upload, PDF page extraction, vector indexing, AI analysis, and RAG QA.
"""

import os
import uuid
import logging
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.config import OPENAI_API_KEY
from backend.parser import extract_pdf_pages
from backend.rag import create_index, search_document
from backend.ai import analyze_document, answer_question

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("docintel")

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_DIR = BASE_DIR / "frontend"

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB

# Initialize FastAPI App
app = FastAPI(
    title="AI Document Intelligence API",
    description="Backend API for automated document ingestion, analysis, and grounded question answering.",
    version="1.0.0",
)

# CORS configuration: Vite React frontend (http://localhost:5173) + localhost / file origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory document storage for MVP
# Schema: {document_id: {"filename": str, "filepath": Path, "pages": list, "analysis": dict, "rag": dict}}
documents: Dict[str, Dict[str, Any]] = {}


# Pydantic Request Models
class Question(BaseModel):
    document_id: str = Field(..., description="Unique ID of the uploaded document")
    question: str = Field(..., min_length=1, description="Question to answer using document context")


@app.get("/")
def health_check(request: Request):
    """
    Health check endpoint.
    Returns JSON for API clients, and renders the DocIntel UI if requested in browser.
    """
    accept_header = request.headers.get("accept", "")
    if "text/html" in accept_header:
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
    return {"message": "AI Document Intelligence API is running"}


@app.get("/api/status")
def api_status():
    """
    Returns backend runtime status and OpenAI key configuration state.
    """
    is_key_configured = bool(
        OPENAI_API_KEY
        and OPENAI_API_KEY != "your_api_key_here"
        and not OPENAI_API_KEY.startswith("dummy")
    )
    return {
        "status": "online",
        "message": "AI Document Intelligence API is running",
        "openai_configured": is_key_configured,
        "indexed_documents": len(documents),
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document (max 20MB), extract text/pages, perform AI extraction,
    and index document content into FAISS for RAG retrieval.
    """
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files (.pdf) are accepted.",
        )

    document_id = str(uuid.uuid4())
    safe_filename = f"{document_id}_{Path(filename).name}"
    file_path = DOCUMENTS_DIR / safe_filename

    try:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB.",
            )

        with open(file_path, "wb") as f:
            f.write(content)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and store uploaded file: {e}",
        )

    # Extract pages with PyMuPDF
    try:
        pages = extract_pdf_pages(file_path)
    except Exception as e:
        logger.error(f"Error parsing PDF '{filename}': {e}")
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to extract text from PDF: {str(e)}",
        )

    if not pages:
        logger.warning(f"No text extracted from '{filename}'.")

    # Run AI Analysis
    try:
        analysis = analyze_document(pages)
    except Exception as e:
        logger.error(f"Error during AI document analysis: {e}")
        analysis = {
            "document_type": "PDF Document",
            "summary": f"Document processed ({len(pages)} pages). AI extraction encountered: {e}",
            "eligibility": [],
            "deadlines": [],
            "fees": [],
            "required_documents": [],
            "important_clauses": [],
            "recommended_actions": [],
            "potential_risks": [],
        }

    # Build FAISS vector index
    try:
        rag_store = create_index(pages)
    except Exception as e:
        logger.error(f"Error building FAISS index: {e}")
        rag_store = {"index": None, "chunks": []}

    # Store state in memory
    documents[document_id] = {
        "document_id": document_id,
        "filename": filename,
        "file_path": file_path,
        "pages": pages,
        "analysis": analysis,
        "rag": rag_store,
    }

    logger.info(f"Document uploaded: id={document_id}, pages={len(pages)}")

    return {
        "document_id": document_id,
        "filename": filename,
        "pages": len(pages),
        "analysis": analysis,
    }


@app.get("/documents/{document_id}/file")
def get_document_file(document_id: str):
    """
    Streams the raw PDF file for an uploaded document.
    """
    doc = documents.get(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    file_path = doc.get("file_path")
    if not file_path or not Path(file_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Underlying document file is missing from storage.",
        )

    return FileResponse(
        path=str(file_path),
        media_type="application/pdf",
        filename=doc["filename"],
    )


@app.post("/ask")
def ask_question(request: Question):
    """
    Answers a question about an uploaded document using semantic search (FAISS)
    and grounded AI inference with source citations.
    """
    doc = documents.get(request.document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{request.document_id}' was not found. Please upload it first.",
        )

    rag_store = doc.get("rag", {})
    if not rag_store or not rag_store.get("chunks"):
        return {
            "answer": "No readable text content is indexed for this document to answer queries.",
            "sources": [],
        }

    try:
        retrieved_chunks = search_document(rag_store, request.question, top_k=5)
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector retrieval failed: {e}",
        )

    try:
        result = answer_question(request.question, retrieved_chunks)
        return result
    except Exception as e:
        logger.error(f"Error generating AI answer: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI answering engine encountered an error: {e}",
        )


@app.get("/app", response_class=FileResponse)
def serve_ui():
    """Serves the DocIntel interactive frontend web application."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Frontend index.html not found.")


@app.get("/robots.txt", response_class=FileResponse)
def serve_robots():
    """Serves robots.txt for search engine crawlers."""
    robots_file = BASE_DIR / "robots.txt"
    if robots_file.exists():
        return FileResponse(robots_file, media_type="text/plain")
    return FileResponse(FRONTEND_DIR / "robots.txt", media_type="text/plain")


@app.get("/sitemap.xml", response_class=FileResponse)
def serve_sitemap():
    """Serves sitemap.xml for search engine indexing."""
    sitemap_file = BASE_DIR / "sitemap.xml"
    if sitemap_file.exists():
        return FileResponse(sitemap_file, media_type="application/xml")
    return FileResponse(FRONTEND_DIR / "sitemap.xml", media_type="application/xml")
