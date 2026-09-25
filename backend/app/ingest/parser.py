"""
PDF Parser module for Alpha Studio.
Uses PyMuPDF and pymupdf4llm for high-performance layout-aware extraction,
converting PDFs into structured Markdown with preserved headers, reading flow, and tables.
"""

import logging
from typing import TypedDict

import fitz
import pymupdf4llm

logger = logging.getLogger(__name__)


class ParsedPage(TypedDict):
    page_number: int
    text: str


def _extract_markdown_pages(doc: fitz.Document) -> list[ParsedPage]:
    """
    Extracts structured Markdown page-by-page from an open PyMuPDF Document.
    Filters out empty/whitespace pages and returns a clean list of ParsedPage objects.
    """
    if doc.is_encrypted and not doc.authenticate(""):
        raise ValueError(
            "PDF document is password-protected/encrypted and cannot be parsed."
        )

    page_data = pymupdf4llm.to_markdown(doc, page_chunks=True)
    pages: list[ParsedPage] = []

    for idx, page in enumerate(page_data):
        raw_page = page.get("metadata", {}).get("page_number", idx + 1)
        try:
            page_number = int(raw_page)
        except (ValueError, TypeError):
            page_number = idx + 1

        text = page.get("text", "").strip()

        if not text:
            logger.debug(f"Skipping empty or whitespace-only page {page_number}")
            continue

        pages.append({"page_number": page_number, "text": text})

    return pages


def parse_pdf_pages(file_path: str) -> list[ParsedPage]:
    """
    Parses a PDF file from a local filesystem path.
    Returns a list of ParsedPage dicts containing the 1-indexed page_number and non-empty Markdown text.
    """
    logger.info(f"Opening PDF file for parsing: {file_path}")
    try:
        with fitz.open(file_path) as doc:
            return _extract_markdown_pages(doc)
    except fitz.EmptyFileError as e:
        logger.error(f"Empty PDF file provided at {file_path}: {e}")
        raise ValueError(f"The provided PDF file is empty (0 bytes): {file_path}") from e
    except fitz.FileDataError as e:
        logger.error(f"Corrupt or invalid PDF file at {file_path}: {e}")
        raise ValueError(
            f"The provided file is corrupted, invalid, or unreadable: {file_path}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error parsing PDF file {file_path}: {e}", exc_info=True)
        raise e


def parse_pdf_bytes(file_bytes: bytes) -> list[ParsedPage]:
    """
    Parses a PDF from raw in-memory bytes (e.g. from FastAPI file uploads).
    Returns a list of ParsedPage dicts containing the 1-indexed page_number and non-empty Markdown text.
    """
    if not file_bytes:
        raise ValueError("Cannot parse empty (0 bytes) PDF payload.")

    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return _extract_markdown_pages(doc)
    except fitz.EmptyFileError as e:
        logger.error(f"Empty PDF byte stream provided: {e}")
        raise ValueError("The provided PDF byte stream is empty (0 bytes).") from e
    except fitz.FileDataError as e:
        logger.error(f"Corrupt or invalid PDF byte stream: {e}")
        raise ValueError(
            "The uploaded file is corrupted, invalid, or not a valid PDF document."
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error parsing in-memory PDF bytes: {e}", exc_info=True)
        raise e