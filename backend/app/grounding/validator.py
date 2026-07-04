import re
import logging
from typing import TypedDict, List
from uuid import UUID

logger = logging.getLogger(__name__)

class GroundingValidationError(Exception):
    """
    Raised when the LLM generates a citation that is not present
    in the retrieved chunks list (a hallucination/grounding violation).
    """
    pass

class ValidatedCitation(TypedDict):
    filename: str
    page_number: int
    chunk_id: UUID

def validate_citations(text: str, retrieved_chunks: List[dict]) -> List[ValidatedCitation]:
    """
    Extracts all inline citations from the text (format: [FILENAME, Page X])
    and validates that each cited file and page number exists in the retrieved_chunks pool.
    
    Raises GroundingValidationError if any citation is ungrounded (fail closed).
    Returns a list of validated citations with their database chunk IDs.
    """
    # Normalize Chinese full-width thick brackets to standard brackets
    normalized_text = text.replace("【", "[").replace("】", "]")
    # Normalize thin non-breaking spaces and other special space characters to normal spaces
    normalized_text = re.sub(r"[\u200b\u200f\u202f\u00a0\u2002\u2003\u2009]", " ", normalized_text)

    # Matches markdown-like inline citations like: [RELIANCE_2025.pdf, Page 59] or [TCS_2025.pdf, page 12]
    # Supports optional bold asterisks (e.g. [**TCS_2025.pdf, Page 59**] or [**TCS_2025.pdf**, Page 59]) and spaces in filenames.
    citation_pattern = r"\[\s*\**\s*([^,\]\*]+?)\s*\**\s*,\s*\**\s*[pP]age\s*\**\s*(\d+)\s*\**\s*\]"
    matches = re.findall(citation_pattern, normalized_text)
    
    if not matches:
        # No citation markup present at all — this is valid for meta-query answers
        # (e.g. "which documents do you have?") that don't cite specific pages.
        # Log at debug level rather than warning so grounding errors aren't falsely triggered.
        logger.debug("No citation markup found in response — treating as a meta-query answer.")
        return []
        
    validated_citations: List[ValidatedCitation] = []
    seen_citations = set()
    
    for filename, page_str in matches:
        clean_filename = filename.strip()
        page_num = int(page_str.strip())
        citation_key = (clean_filename.lower(), page_num)
        
        if citation_key in seen_citations:
            continue
            
        # Verify if this citation matches any retrieved chunk source document
        is_grounded = False
        matching_chunk_id = None
        
        for r in retrieved_chunks:
            c = r["chunk"]
            if c.document.filename.lower() == clean_filename.lower() and c.page_number == page_num:
                is_grounded = True
                matching_chunk_id = c.id
                break
                
        if not is_grounded:
            error_msg = (
                f"Grounding validation failed: LLM cited '{clean_filename}' (Page {page_num}), "
                f"which was not part of the retrieved context segments for this query."
            )
            logger.error(error_msg)
            raise GroundingValidationError(error_msg)
            
        seen_citations.add(citation_key)
        validated_citations.append({
            "filename": clean_filename,
            "page_number": page_num,
            "chunk_id": matching_chunk_id
        })
        
    logger.info(f"Successfully validated {len(validated_citations)} unique citations.")
    return validated_citations
