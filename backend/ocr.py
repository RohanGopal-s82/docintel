"""
OCR Pipeline for scanned or image-based PDF documents.
Converts PDF pages to images using pdf2image and runs pytesseract for OCR.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Union
import pytesseract
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)


def ocr_pdf(pdf_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Extract text from a scanned or image-only PDF using Tesseract OCR.
    
    Args:
        pdf_path: File path to the PDF document.
        
    Returns:
        List of dictionaries containing 1-indexed page numbers and extracted text:
        [{"page": 1, "text": "..."}]
    """
    path_obj = Path(pdf_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"PDF file not found: {path_obj}")

    logger.info(f"Starting OCR extraction for: {path_obj.name}")
    extracted_pages: List[Dict[str, Any]] = []

    try:
        # Convert PDF pages to PIL Images at 200 DPI
        images = convert_from_path(str(path_obj), dpi=200)
    except Exception as e:
        logger.error(f"Failed to convert PDF pages to images: {e}")
        # Note: pdf2image requires poppler to be installed and in PATH
        raise RuntimeError(
            f"Error converting PDF pages to images. Ensure Poppler is installed: {e}"
        ) from e

    for page_idx, image in enumerate(images, start=1):
        try:
            text = pytesseract.image_to_string(image).strip()
            extracted_pages.append({"page": page_idx, "text": text})
            logger.info(f"OCR completed for page {page_idx} ({len(text)} chars)")
        except Exception as e:
            logger.warning(f"OCR failed for page {page_idx}: {e}")
            extracted_pages.append({"page": page_idx, "text": ""})

    return extracted_pages
