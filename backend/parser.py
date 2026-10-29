"""
Document ingestion and text extraction pipeline using PyMuPDF and OCR fallback.
Preserves page-level metadata across extracted pages.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Union
try:
    import pymupdf as fitz
except ImportError:
    import fitz  # PyMuPDF
from backend.ocr import ocr_pdf

logger = logging.getLogger(__name__)


def get_pdf_page_count(file_path: Union[str, Path]) -> int:
    """
    Returns the total number of physical pages in a PDF document.
    """
    path_obj = Path(file_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"PDF file does not exist: {path_obj}")

    doc = fitz.open(str(path_obj))
    try:
        return len(doc)
    finally:
        doc.close()


def extract_pdf_pages(
    file_path: Union[str, Path], include_empty: bool = False
) -> List[Dict[str, Any]]:
    """
    Extracts text from a PDF file preserving 1-indexed page metadata.
    Falls back to OCR if the extracted character count across all pages is < 100.
    
    Args:
        file_path: Path to the target PDF file.
        include_empty: If True, preserves empty pages as {"page": N, "text": ""}
                       to maintain strict physical page indexing.
        
    Returns:
        List of dicts: [{"page": int, "text": str}]
    """
    path_obj = Path(file_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"PDF file does not exist: {path_obj}")

    pages: List[Dict[str, Any]] = []
    total_chars: int = 0

    try:
        doc = fitz.open(str(path_obj))
    except Exception as e:
        logger.error(f"Failed to open PDF {path_obj}: {e}")
        raise ValueError(f"Corrupt or unreadable PDF document: {e}") from e

    try:
        for page_num in range(1, len(doc) + 1):
            page = doc[page_num - 1]
            text = page.get_text("text").strip()
            total_chars += len(text)
            if text or include_empty:
                pages.append({"page": page_num, "text": text})
    finally:
        doc.close()

    # Fallback to OCR if total text across all pages is less than 100 characters
    if total_chars < 100:
        logger.warning(
            f"Low character count ({total_chars} chars) in {path_obj.name}. "
            "Scanned or image-only PDF detected; delegating to OCR fallback."
        )
        try:
            ocr_pages = ocr_pdf(path_obj)
            # Filter non-empty pages unless include_empty is True
            filtered_ocr = (
                ocr_pages if include_empty else [p for p in ocr_pages if p.get("text")]
            )
            if filtered_ocr:
                return filtered_ocr
            return ocr_pages if ocr_pages else pages
        except Exception as ocr_err:
            logger.error(f"OCR fallback failed: {ocr_err}")
            if pages:
                return pages
            raise RuntimeError(
                f"Document appears scanned (<100 chars), and OCR fallback failed: {ocr_err}"
            ) from ocr_err

    return pages
