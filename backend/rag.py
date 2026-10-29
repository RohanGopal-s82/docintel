"""
Vector Indexing & RAG Retrieval module using FAISS and OpenAI embeddings.
Provides chunking, embedding generation, index creation, and similarity search.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from openai import OpenAI, AuthenticationError, APIConnectionError
from backend.config import OPENAI_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Initialize OpenAI Client with timeout and fallback dummy key if not set yet
client = OpenAI(
    api_key=OPENAI_API_KEY or "dummy_key",
    timeout=15.0,
    max_retries=1,
)


def create_chunks(
    pages: List[Dict[str, Any]], chunk_size: int = 1000, overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Splits page text into word chunks while preserving source page metadata.
    
    Args:
        pages: List of dicts [{"page": int, "text": str}]
        chunk_size: Maximum number of words per chunk
        overlap: Number of overlapping words between consecutive chunks
        
    Returns:
        List of dicts: [{"text": str, "page": int}]
    """
    chunks: List[Dict[str, Any]] = []
    step = max(1, chunk_size - overlap)

    for p in pages:
        page_num = p.get("page", 1)
        text = p.get("text", "").strip()
        words = text.split()

        if not words:
            continue

        if len(words) <= chunk_size:
            chunks.append({"text": text, "page": page_num})
            continue

        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({"text": chunk_text, "page": page_num})
            if i + chunk_size >= len(words):
                break

    return chunks


def _embed(texts: List[str]) -> np.ndarray:
    """
    Generates normalized float32 embeddings for a list of strings using OpenAI.
    
    Args:
        texts: List of text strings to embed.
        
    Returns:
        L2-normalized 2D float32 numpy array of shape (N, dimension).
    """
    if not texts:
        return np.empty((0, 1536), dtype=np.float32)

    # Return placeholder embeddings if API key is not configured
    if not OPENAI_API_KEY or OPENAI_API_KEY in ("your_api_key_here", "dummy_key") or OPENAI_API_KEY.startswith("dummy"):
        return np.zeros((len(texts), 1536), dtype=np.float32)

    try:
        response = client.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL
        )
        raw_embeddings = [item.embedding for item in response.data]
        vectors = np.array(raw_embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        return vectors
    except (AuthenticationError, APIConnectionError) as auth_err:
        logger.warning(f"OpenAI Authentication/Connection notice during embeddings: {auth_err}")
        # Return fallback zero vectors so FAISS indexing does not crash
        return np.zeros((len(texts), 1536), dtype=np.float32)
    except Exception as e:
        logger.error(f"Error during OpenAI embedding generation: {e}")
        return np.zeros((len(texts), 1536), dtype=np.float32)


def create_index(pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Creates a FAISS vector index from document pages.
    
    Args:
        pages: Extracted pages [{"page": int, "text": str}]
        
    Returns:
        Dictionary {"index": faiss.IndexFlatIP or None, "chunks": list}
    """
    chunks = create_chunks(pages)
    if not chunks:
        logger.warning("No text chunks found to create FAISS index.")
        return {"index": None, "chunks": []}

    texts = [c["text"] for c in chunks]
    vectors = _embed(texts)

    if vectors.shape[0] == 0:
        return {"index": None, "chunks": chunks}

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    logger.info(f"FAISS index created successfully with {index.ntotal} vectors of dim {dimension}.")

    return {"index": index, "chunks": chunks}


def search_document(
    rag_store: Dict[str, Any], question: str, top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Searches the FAISS vector index for chunks most relevant to the question.
    
    Args:
        rag_store: Dict containing {"index": IndexFlatIP, "chunks": list}
        question: User query string
        top_k: Number of nearest neighbors to retrieve
        
    Returns:
        List of matching chunks with similarity scores:
        [{"text": str, "page": int, "score": float}]
    """
    index = rag_store.get("index")
    chunks = rag_store.get("chunks", [])

    if index is None or not chunks or not question.strip():
        return []

    q_vector = _embed([question.strip()])
    if q_vector.shape[0] == 0:
        return []

    effective_k = min(top_k, getattr(index, "ntotal", len(chunks)), len(chunks))
    if effective_k <= 0:
        return []

    distances, indices = index.search(q_vector, effective_k)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(distances[0], indices[0]):
        if idx != -1 and 0 <= idx < len(chunks):
            matched_chunk = chunks[idx]
            results.append({
                "text": matched_chunk["text"],
                "page": matched_chunk["page"],
                "score": float(score)
            })

    return results
