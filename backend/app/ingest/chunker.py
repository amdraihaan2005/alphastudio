import re
from typing import TypedDict


class DocumentChunkData(TypedDict):
    chunk_index: int
    page_number: int
    section_name: str | None
    text_content: str


SECTION_KEYWORDS = [
    r"independent auditor's report",
    r"board's report",
    r"directors' report",
    r"balance sheet",
    r"statement of profit and loss",
    r"cash flow statement",
    r"notes to financial statements",
    r"management discussion and analysis",
    r"corporate governance report",
    r"financial statements",
    r"directors’ report",
]


def detect_section_header(line: str) -> str | None:
    """
    1. Detects if a line looks like a major section header.
    Returns the cleaned section name if detected, otherwise None.
    """
    cleaned = line.strip()
    if not cleaned:
        return None

    if cleaned.startswith("#"):
        header_text = cleaned.lstrip("#").strip()
        if len(header_text) < 100:
            return header_text

    if len(cleaned) < 100 and (
        cleaned.isupper()
        or any(re.search(kw, cleaned, re.IGNORECASE) for kw in SECTION_KEYWORDS)
    ):
        stripped = re.sub(r"^[\d\.\-\s]+", "", cleaned).strip()
        if len(stripped) > 3:
            return stripped

    return None


def split_text_into_overlapping_chunks(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[str]:
    """
    2. Splits text into chunks of target character size and overlap.
    Attempts to break at clean word boundaries or punctuation.
    """
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        best_boundary = end
        for i in range(end, max(start, end - 100), -1):
            if text[i] == "\n":
                best_boundary = i
                break
            elif text[i] in (".", "!", "?") and (
                i + 1 < text_length and text[i + 1].isspace()
            ):
                best_boundary = i + 1
                break
            elif text[i].isspace() and best_boundary == end:
                best_boundary = i

        chunk = text[start:best_boundary].strip()
        if chunk:
            chunks.append(chunk)

        start = best_boundary - chunk_overlap
        if start >= text_length:
            break

        if start <= chunks[-1].rfind(chunk[:10]) + start:
            start = best_boundary

    return chunks


def chunk_page(
    page_text: str,
    page_number: int,
    start_chunk_idx: int,
    current_section: str | None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> tuple[list[DocumentChunkData], str | None]:
    """
    3. Chunks a single page's text, scanning for any new section header.
    Returns the list of chunks generated, and the updated current section name.
    """
    lines = page_text.split("\n")
    active_section = current_section

    for line in lines[:8]:
        detected = detect_section_header(line)
        if detected:
            active_section = detected
            break

    text_chunks = split_text_into_overlapping_chunks(
        page_text, chunk_size, chunk_overlap
    )

    chunks: list[DocumentChunkData] = []
    for idx, content in enumerate(text_chunks):
        chunks.append(
            {
                "chunk_index": start_chunk_idx + idx,
                "page_number": page_number,
                "section_name": active_section,
                "text_content": content,
            }
        )

    return chunks, active_section
