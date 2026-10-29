"""
AI Extraction and Grounded Answering module with Strict Citation Grounding.
Enforces page boundary constraints, structured outputs, and deterministic citation sanitization.
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field
from openai import OpenAI, AuthenticationError, APIError, APIConnectionError
from backend.config import OPENAI_API_KEY, CHAT_MODEL

logger = logging.getLogger(__name__)

# Initialize OpenAI Client (graceful placeholder handling and timeout)
client = OpenAI(
    api_key=OPENAI_API_KEY or "dummy_key",
    timeout=15.0,
    max_retries=1,
)

# Common stopwords to exclude from token-matching heuristic
STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what", "which",
    "this", "that", "these", "those", "then", "just", "so", "than", "such",
    "both", "through", "about", "for", "is", "of", "while", "during", "to",
    "from", "in", "out", "on", "off", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "too", "very", "can", "will", "just",
    "should", "now", "are", "was", "were", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "would", "could", "might",
    "must", "shall", "per", "all", "must", "applicable", "applies"
}


# ==============================================================================
# Helper Function Required by Prompt 4
# ==============================================================================

def _call_model(prompt: str) -> str:
    """
    Calls OpenAI Chat Completions API with CHAT_MODEL and returns completion text content.
    """
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"OpenAI completion failed: {e}")
        raise RuntimeError(f"OpenAI completion failed: {e}") from e


# ==============================================================================
# Pydantic Schemas for Strict Structured Outputs
# Note: For OpenAI Structured Outputs, all fields must be required (no default_factory).
# ==============================================================================

class EligibilityItem(BaseModel):
    requirement: str = Field(description="Eligibility criteria or applicant requirement")
    page: int = Field(description="Exact 1-indexed page number where this condition appears")


class DeadlineItem(BaseModel):
    date: str = Field(description="Date, timeframe, or submission deadline")
    description: str = Field(description="Description of action or requirement due by this date")
    page: int = Field(description="Exact 1-indexed page number")


class FeeItem(BaseModel):
    amount: str = Field(description="Fee amount, charge, penalty, or percentage")
    description: str = Field(description="Purpose, trigger condition, or description of the fee")
    page: int = Field(description="Exact 1-indexed page number")


class RequiredDocumentItem(BaseModel):
    document: str = Field(description="Specific document, certificate, or credential required")
    page: int = Field(description="Exact 1-indexed page number")


class ImportantClauseItem(BaseModel):
    clause: str = Field(description="Name, title, or summary of the clause")
    explanation: str = Field(description="Plain-language explanation of implications")
    page: int = Field(description="Exact 1-indexed page number")


class RecommendedActionItem(BaseModel):
    action: str = Field(description="Actionable step the user should take")
    reason: str = Field(description="Reason or justification for the recommended action")
    page: int = Field(description="Exact 1-indexed page number")


class PotentialRiskItem(BaseModel):
    risk: str = Field(description="Identified risk, condition for rejection, or penalty")
    explanation: str = Field(description="Explanation of triggers and consequences")
    page: int = Field(description="Exact 1-indexed page number")


class DocumentAnalysisSchema(BaseModel):
    document_type: str = Field(description="Specific document type or classification")
    summary: str = Field(description="Comprehensive plain-language summary of the entire document")
    eligibility: List[EligibilityItem]
    deadlines: List[DeadlineItem]
    fees: List[FeeItem]
    required_documents: List[RequiredDocumentItem]
    important_clauses: List[ImportantClauseItem]
    recommended_actions: List[RecommendedActionItem]
    potential_risks: List[PotentialRiskItem]


class CitationSource(BaseModel):
    page: int = Field(description="1-indexed page number where supporting evidence appears")
    reason: str = Field(description="Brief explanation of how this page grounds the answer")


class QuestionAnswerSchema(BaseModel):
    answer: str = Field(description="Direct answer strictly grounded in provided evidence")
    sources: List[CitationSource]


# ==============================================================================
# Deterministic Citation Grounding & Token Matching Engine
# ==============================================================================

def _extract_search_tokens(item: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Extracts distinct content words and high-priority numeric/currency tokens from an item.
    """
    combined_text = " ".join(str(val) for key, val in item.items() if key != "page")
    raw_tokens = re.findall(r"[^\s,;!?()\"']+", combined_text.lower())

    content_tokens: List[str] = []
    priority_tokens: List[str] = []

    for t in raw_tokens:
        clean = t.strip(".,;:\"'()[]{}")
        if not clean:
            continue
        if re.search(r"\d", clean) or any(c in clean for c in "₹$€£%"):
            priority_tokens.append(clean.replace(",", ""))
            continue
        if len(clean) >= 3 and clean not in STOP_WORDS:
            content_tokens.append(clean)

    return content_tokens, priority_tokens


def _find_best_grounded_page(
    item: Dict[str, Any],
    pages: List[Dict[str, Any]],
    claimed_page: int,
    valid_page_set: Set[int],
    min_page: int,
    max_page: int,
) -> int:
    """
    Determines the most accurate, grounded page number for an extracted item.
    """
    if not pages:
        return 1

    content_tokens, priority_tokens = _extract_search_tokens(item)

    def compute_page_score(page_text: str) -> float:
        text_lower = page_text.lower()
        score = 0.0

        for pt in priority_tokens:
            if pt in text_lower or pt.replace(".", "") in text_lower:
                score += 3.5

        for ct in content_tokens:
            if re.search(rf"\b{re.escape(ct)}\b", text_lower):
                score += 1.0

        return score

    page_scores: Dict[int, float] = {}
    for p in pages:
        p_num = p.get("page", 1)
        p_text = p.get("text", "")
        page_scores[p_num] = compute_page_score(p_text)

    claimed_score = page_scores.get(claimed_page, 0.0)
    best_candidate_page, best_score = max(page_scores.items(), key=lambda x: x[1])

    # Case A: Claimed page is within bounds and has strong grounding
    if claimed_page in valid_page_set:
        if claimed_score > 0 and claimed_score >= 0.6 * best_score:
            return claimed_page

        if claimed_score == 0.0 and best_score >= 2.0:
            logger.info(
                f"[Citation Grounding] Moving item from Page {claimed_page} "
                f"to Page {best_candidate_page} (score: {best_score:.1f} vs {claimed_score:.1f})"
            )
            return best_candidate_page

        return claimed_page

    # Case B: Claimed page is out of bounds (e.g. Page 3 on 2-page document)
    logger.warning(
        f"[Citation Grounding] Out-of-bounds Page {claimed_page} detected. Valid range: {min_page}..{max_page}"
    )
    if best_score > 0.0:
        logger.info(f"-> Overwriting with grounded Page {best_candidate_page} (score: {best_score:.1f})")
        return best_candidate_page

    return min_page  # Fallback to 1 or earliest valid reference


def validate_and_sanitize_citations(
    analysis: Dict[str, Any], pages: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Post-processes document extraction output to guarantee strict citation grounding.
    """
    if not pages:
        return analysis

    valid_page_set: Set[int] = {p.get("page", idx) for idx, p in enumerate(pages, start=1)}
    min_page: int = min(valid_page_set)
    max_page: int = max(valid_page_set)

    citation_sections = [
        "eligibility",
        "deadlines",
        "fees",
        "required_documents",
        "important_clauses",
        "recommended_actions",
        "potential_risks",
    ]

    for section in citation_sections:
        items = analysis.get(section, [])
        if not isinstance(items, list):
            analysis[section] = []
            continue

        sanitized_items = []
        for item in items:
            if not isinstance(item, dict):
                continue

            raw_page = item.get("page")
            try:
                page_int = int(raw_page)
            except (ValueError, TypeError):
                page_int = min_page

            verified_page = _find_best_grounded_page(
                item=item,
                pages=pages,
                claimed_page=page_int,
                valid_page_set=valid_page_set,
                min_page=min_page,
                max_page=max_page,
            )
            item["page"] = verified_page
            sanitized_items.append(item)

        analysis[section] = sanitized_items

    return analysis


# ==============================================================================
# Model Invocation & AI Extraction Pipeline
# ==============================================================================

def _clean_json_str(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def _call_model_structured(
    system_prompt: str, user_prompt: str, response_model: type[BaseModel]
) -> Dict[str, Any]:
    """
    Calls OpenAI Chat Completions enforcing structured outputs via Pydantic model.
    Attempts modern client.beta.chat.completions.parse, with fallback to json_object mode.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Attempt 1: Strict Structured Outputs
    try:
        completion = client.beta.chat.completions.parse(
            model=CHAT_MODEL,
            messages=messages,
            response_format=response_model,
            temperature=0.0,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is not None:
            return parsed.model_dump()
    except AuthenticationError:
        raise
    except Exception as parse_err:
        logger.warning(
            f"client.beta.chat.completions.parse notice ({parse_err}). Using json_object mode fallback."
        )

    # Attempt 2: Structured JSON Object Mode
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = _clean_json_str(response.choices[0].message.content or "{}")
        try:
            validated_obj = response_model.model_validate_json(content)
            return validated_obj.model_dump()
        except Exception:
            return json.loads(content)
    except AuthenticationError:
        raise
    except Exception as e:
        logger.error(f"OpenAI completion failed: {e}")
        raise RuntimeError(f"OpenAI completion failed: {e}") from e


def _extract_offline_analysis(pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extracts summary, dates, fees, and requirements directly from document text
    when no OpenAI API key is configured.
    """
    if not pages:
        return {
            "document_type": "PDF Document",
            "summary": "No text content found in document.",
            "eligibility": [],
            "deadlines": [],
            "fees": [],
            "required_documents": [],
            "important_clauses": [],
            "recommended_actions": [],
            "potential_risks": [],
        }

    total_pages = len(pages)
    first_page_text = pages[0].get("text", "")
    all_text_start = " ".join(p.get("text", "") for p in pages[:3])

    # Infer document type
    doc_type = "PDF Document"
    lower_start = all_text_start.lower()
    if any(k in lower_start for k in ["agreement", "contract", "memorandum", "lease", "tenant"]):
        doc_type = "Legal Agreement / Contract"
    elif any(k in lower_start for k in ["policy", "insurance", "coverage", "premium"]):
        doc_type = "Insurance / Policy Document"
    elif any(k in lower_start for k in ["scholarship", "fellowship", "grant", "fellow"]):
        doc_type = "Scholarship / Grant Document"
    elif any(k in lower_start for k in ["invoice", "receipt", "billing", "tax invoice"]):
        doc_type = "Financial Invoice / Bill"
    elif any(k in lower_start for k in ["specification", "requirements", "manual", "guide", "handbook"]):
        doc_type = "Technical Specification / Manual"
    elif any(k in lower_start for k in ["report", "analysis", "audit"]):
        doc_type = "Report / Analytical Document"

    # Extract substantive sentences for an informative summary
    cleaned_sentences = []
    for raw_s in re.split(r"(?<=[.!?])\s+", first_page_text):
        s = raw_s.strip()
        if len(s) > 25 and not s.startswith("http") and not re.match(r"^[\d\s\W]+$", s):
            cleaned_sentences.append(s)
        if len(cleaned_sentences) >= 3:
            break

    if cleaned_sentences:
        extractive_summary = " ".join(cleaned_sentences)
    else:
        extractive_summary = first_page_text[:350].strip() or f"Document containing {total_pages} page(s)."

    deadlines = []
    fees = []
    eligibility = []
    important_clauses = []

    date_pattern = re.compile(
        r"\b(?:\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE,
    )
    fee_pattern = re.compile(r"(?:₹|Rs\.?|\$|USD|EUR|€)\s?[\d,]+(?:\.\d{2})?", re.IGNORECASE)

    for p in pages:
        p_num = p.get("page", 1)
        p_text = p.get("text", "")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p_text) if len(s.strip()) > 15]

        for s in sentences:
            # Check dates / deadlines
            if len(deadlines) < 5:
                date_match = date_pattern.search(s)
                if date_match and any(w in s.lower() for w in ["deadline", "due", "submit", "submission", "before", "by", "date", "expire"]):
                    deadlines.append({
                        "date": date_match.group(0),
                        "description": s[:150],
                        "page": p_num,
                    })

            # Check fees
            if len(fees) < 5:
                fee_match = fee_pattern.search(s)
                if fee_match and any(w in s.lower() for w in ["fee", "cost", "charge", "amount", "pay", "penalty", "premium"]):
                    fees.append({
                        "amount": fee_match.group(0),
                        "description": s[:150],
                        "page": p_num,
                    })

            # Check eligibility / requirements
            if len(eligibility) < 5:
                if any(w in s.lower() for w in ["must be", "required to", "eligible", "eligibility", "qualification", "shall have"]):
                    eligibility.append({
                        "requirement": s[:160],
                        "page": p_num,
                    })

            # Check clauses
            if len(important_clauses) < 4:
                if any(w in s.lower() for w in ["clause", "liability", "termination", "indemnity", "penalty", "restriction", "jurisdiction"]):
                    important_clauses.append({
                        "clause": s[:60] + "...",
                        "explanation": s[:160],
                        "page": p_num,
                    })

    return {
        "document_type": doc_type,
        "summary": extractive_summary,
        "eligibility": eligibility,
        "deadlines": deadlines,
        "fees": fees,
        "required_documents": [],
        "important_clauses": important_clauses,
        "recommended_actions": [],
        "potential_risks": [],
    }


def analyze_document(pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extracts key fields, metadata, clauses, and actions from document pages.
    Guarantees strict citation grounding and valid page ranges.
    """
    fallback_result: Dict[str, Any] = {
        "document_type": "Unknown",
        "summary": "No readable text extracted.",
        "eligibility": [],
        "deadlines": [],
        "fees": [],
        "required_documents": [],
        "important_clauses": [],
        "recommended_actions": [],
        "potential_risks": [],
    }

    if not pages:
        return fallback_result

    # If API key is not configured, generate smart extractive summary directly from text
    if not OPENAI_API_KEY or OPENAI_API_KEY in ("your_api_key_here", "dummy_key") or OPENAI_API_KEY.startswith("dummy"):
        return _extract_offline_analysis(pages)

    total_pages = len(pages)
    min_page = min((p.get("page", idx) for idx, p in enumerate(pages, 1)), default=1)
    max_page = max((p.get("page", idx) for idx, p in enumerate(pages, 1)), default=total_pages)

    # Format document payload with distinct, unambiguous visual delimiters
    doc_sections: List[str] = []
    for p in pages:
        p_num = p.get("page", 1)
        p_text = p.get("text", "")
        doc_sections.append(
            f"--- START OF PAGE {p_num} ---\n{p_text}\n--- END OF PAGE {p_num} ---"
        )
    formatted_document_payload = "\n\n".join(doc_sections)

    system_prompt = f"""You are a Principal Document Intelligence AI specializing in precision extraction and citation grounding.

DOCUMENT METADATA & BOUNDARIES:
- The document contains exactly {total_pages} page(s) (numbered from {min_page} to {max_page}).
- Each page is bounded by "--- START OF PAGE X ---" and "--- END OF PAGE X ---".
- You must NEVER cite a page number outside the range {min_page} to {max_page}.

STRICT CITATION RULES:
1. Every cited "page" property must be an integer matching the exact page delimiter where that specific statement appears.
2. If a field combines information from multiple sections, set "page" to the earliest page where the primary clause appears.
3. Do NOT guess, estimate, or hallucinate page numbers.
4. If a condition appears on Page 1, cite 1. If on Page 2, cite 2.
5. If an item cannot be grounded in the text, omit it or return an empty array. Never invent information."""

    user_prompt = f"""Analyze the provided document pages and extract structured facts.

Document Content:
\"\"\"
{formatted_document_payload}
\"\"\"

Extract the required fields into a strict JSON object matching this schema:
{{
  "document_type": "string describing type of document",
  "summary": "plain-language summary of what the document says and means",
  "eligibility": [{{"requirement": "string", "page": int}}],
  "deadlines": [{{"date": "string", "description": "string", "page": int}}],
  "fees": [{{"amount": "string", "description": "string", "page": int}}],
  "required_documents": [{{"document": "string", "page": int}}],
  "important_clauses": [{{"clause": "string", "explanation": "string", "page": int}}],
  "recommended_actions": [{{"action": "string", "reason": "string", "page": int}}],
  "potential_risks": [{{"risk": "string", "explanation": "string", "page": int}}]
}}"""

    try:
        raw_analysis = _call_model_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=DocumentAnalysisSchema,
        )
    except AuthenticationError as auth_err:
        logger.error(f"OpenAI Authentication Error: {auth_err}")
        return {
            "document_type": "PDF Document",
            "summary": "OpenAI Authentication Failed: Please update OPENAI_API_KEY in your .env file with a valid OpenAI API key.",
            "error": "Invalid OpenAI API Key",
            "eligibility": [],
            "deadlines": [],
            "fees": [],
            "required_documents": [],
            "important_clauses": [],
            "recommended_actions": [],
            "potential_risks": [],
        }
    except Exception as e:
        logger.error(f"Error during AI document analysis: {e}")
        return fallback_result

    # Deterministic Post-Processing & Citation Validation Layer
    grounded_analysis = validate_and_sanitize_citations(raw_analysis, pages)

    return grounded_analysis


def answer_question(
    question: str, retrieved_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Answers a question strictly using the provided retrieved context chunks,
    with enforced citation grounding to retrieved pages.
    """
    fallback_response = {
        "answer": "I couldn't find this information in the uploaded document.",
        "sources": [],
    }

    if not retrieved_chunks:
        return fallback_response

    # If API key is not configured, extract the most relevant sentence from retrieved chunks
    if not OPENAI_API_KEY or OPENAI_API_KEY in ("your_api_key_here", "dummy_key") or OPENAI_API_KEY.startswith("dummy"):
        q_words = [w.lower() for w in re.findall(r"\w+", question) if len(w) > 2 and w.lower() not in STOP_WORDS]
        best_sentence = ""
        best_page = retrieved_chunks[0].get("page", 1) if retrieved_chunks else 1
        best_matches = 0

        for chunk in retrieved_chunks:
            p_num = chunk.get("page", 1)
            text = chunk.get("text", "")
            for s in re.split(r"(?<=[.!?])\s+", text):
                s_clean = s.strip()
                matches = sum(1 for w in q_words if w in s_clean.lower())
                if matches > best_matches:
                    best_matches = matches
                    best_sentence = s_clean
                    best_page = p_num

        if best_sentence:
            ans = f'Relevant excerpt from Page {best_page}: "{best_sentence}"'
        else:
            first_chunk = retrieved_chunks[0].get("text", "")[:250].strip()
            ans = f'Relevant context from Page {best_page}: "{first_chunk}..."'

        return {
            "answer": ans,
            "sources": [{"page": c.get("page", 1), "reason": "Retrieved context chunk"} for c in retrieved_chunks[:2]],
        }

    valid_evidence_pages: Set[int] = {c.get("page", 1) for c in retrieved_chunks}
    valid_pages_str = ", ".join(f"Page {p}" for p in sorted(valid_evidence_pages))

    evidence_blocks: List[str] = []
    for chunk in retrieved_chunks:
        page_num = chunk.get("page", 1)
        text = chunk.get("text", "")
        evidence_blocks.append(
            f"--- START OF EVIDENCE [PAGE {page_num}] ---\n{text}\n--- END OF EVIDENCE [PAGE {page_num}] ---"
        )
    evidence_payload = "\n\n".join(evidence_blocks)

    system_prompt = f"""You are a precise, grounded Document Intelligence Q&A agent.
Answer the user's question based STRICTLY and ONLY on the provided evidence excerpts.

CITATION RULES:
1. Valid source pages in these excerpts are: {valid_pages_str}.
2. You must ONLY cite page numbers from this valid list ({sorted(list(valid_evidence_pages))}).
3. Never cite a page number that does not appear in the evidence delimiters.
4. If the answer is not present in the excerpts, state: "I couldn't find this information in the uploaded document." and return an empty sources list."""

    user_prompt = f"""Evidence Excerpts:
\"\"\"
{evidence_payload}
\"\"\"

Question:
{question}

Return a JSON object with:
{{
  "answer": "direct grounded answer",
  "sources": [{{"page": int, "reason": "string"}}]
}}"""

    try:
        raw_qa = _call_model_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=QuestionAnswerSchema,
        )
    except AuthenticationError:
        return {
            "answer": "OpenAI Authentication Failed: Please check your OPENAI_API_KEY in the .env file.",
            "sources": [],
        }
    except Exception as e:
        logger.error(f"Error during Q&A inference: {e}")
        return fallback_response

    sanitized_sources = []
    for s in raw_qa.get("sources", []):
        try:
            page_int = int(s.get("page", 0))
            if page_int in valid_evidence_pages:
                sanitized_sources.append({"page": page_int, "reason": s.get("reason", "")})
            elif valid_evidence_pages:
                closest_page = min(valid_evidence_pages, key=lambda p: abs(p - page_int))
                sanitized_sources.append({"page": closest_page, "reason": s.get("reason", "")})
        except Exception:
            continue

    raw_qa["sources"] = sanitized_sources
    return raw_qa
