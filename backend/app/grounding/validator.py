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


def validate_citations(
    text: str, retrieved_chunks: List[dict]
) -> List[ValidatedCitation]:
    """
    Extracts all inline citations from the text (format: [FILENAME, Page X])
    and validates that each cited file and page number exists in the retrieved_chunks pool.

    Raises GroundingValidationError if any citation is ungrounded (fail closed).
    Returns a list of validated citations with their database chunk IDs.
    """
    normalized_text = text.replace("【", "[").replace("】", "]")
    normalized_text = re.sub(
        r"[\u200b\u200f\u202f\u00a0\u2002\u2003\u2009]", " ", normalized_text
    )

    citation_pattern = (
        r"\[\s*\**\s*([^,\]\*]+?)\s*\**\s*,\s*\**\s*[pP]age\s*\**\s*(\d+)\s*\**\s*\]"
    )
    matches = re.findall(citation_pattern, normalized_text)

    if not matches:
        logger.debug(
            "No citation markup found in response — treating as a meta-query answer."
        )
        return []

    validated_citations: List[ValidatedCitation] = []
    seen_citations = set()

    for filename, page_str in matches:
        clean_filename = filename.strip()
        page_num = int(page_str.strip())
        citation_key = (clean_filename.lower(), page_num)

        if citation_key in seen_citations:
            continue

        is_grounded = False
        matching_chunk_id = None

        for r in retrieved_chunks:
            c = r["chunk"]
            if (
                c.document.filename.lower() == clean_filename.lower()
                and c.page_number == page_num
            ):
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
        validated_citations.append(
            {
                "filename": clean_filename,
                "page_number": page_num,
                "chunk_id": matching_chunk_id,
            }
        )

    logger.info(f"Successfully validated {len(validated_citations)} unique citations.")
    return validated_citations
