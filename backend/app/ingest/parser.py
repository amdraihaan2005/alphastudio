"""
PDF Parser module for Alpha Studio.

Ingestion Architecture:
- Single Upload Entrypoint: Users upload one PDF at a time via the web interface.
- Multi-Document Comparison: Users query and compare multiple documents in a single prompt.
  To support cross-document retrieval, comparison, and accurate citations down the pipeline,
  every parsed page preserves document identity, filename provenance, page numbering,
  and rich PDF metadata.
- Single-Pass Global Layout & Semantic Hierarchy: Runs pymupdf4llm in a single pass across the entire
  document via `page_chunks=True`. This prevents N+1 engine re-initialization overhead and preserves
  global font-size baselines for accurate Markdown header hierarchy (#, ##, ###).
- Direct In-Memory C-Bindings: Directly opens byte streams via PyMuPDF C-bindings with zero disk
  spooling, zero temporary files, and flat memory footprint.
- Official LangChain Integration:
  - `PyPDFToLangChainParser`: A first-class subclass of `langchain_core.document_loaders.BaseBlobParser`
    for parsing PDF Blobs into LangChain `Document` objects.
  - `PyPDFBytesLoader`: A first-class subclass of `langchain_core.document_loaders.BaseLoader`
    enabling standard lazy and eager document ingestion pipelines.
"""

from collections.abc import Iterator
import logging
from typing import Any, TypedDict

import fitz
from langchain_core.document_loaders import BaseBlobParser, BaseLoader, Blob
from langchain_core.documents import Document
import pymupdf4llm

logger = logging.getLogger(__name__)

# Disable pymupdf4llm stateful layout cache to guarantee deterministic extraction
if hasattr(pymupdf4llm, "use_layout"):
    pymupdf4llm.use_layout(False)


class ParsedPage(TypedDict):
    """
    Structured representation of a parsed PDF page for downstream pipeline stages.
    Maintains compatibility with chunker and orchestrator stages while retaining
    full document provenance and PDF metadata for multi-document comparisons.
    """

    page_number: int
    text: str
    metadata: dict[str, Any]


class PyPDFToLangChainParser(BaseBlobParser):
    """
    Official LangChain BaseBlobParser implementation powered by PyMuPDF and PyMuPDF4LLM.

    Capabilities:
    - Parses PDF byte streams and Blobs into canonical LangChain `Document` objects.
    - Evaluates global document layout and font-size distributions in a single pass (`page_chunks=True`),
      preserving Markdown header levels (#, ##, ###) and table formatting.
    - Captures comprehensive document provenance (source, file_path, 1-based page numbers, total_pages,
      author, creationdate, title) essential for grounded multi-document financial analysis.
    - Translates PyMuPDF C-binding exceptions (encrypted, corrupted, empty) into descriptive errors.
    """

    def __init__(self, password: str | None = None):
        self.password = password

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """
        Lazily parses a PDF Blob into an iterator of page-level LangChain Document objects.
        """
        file_bytes = blob.as_bytes()
        if not file_bytes:
            raise ValueError("Cannot parse empty (0 bytes) PDF payload.")

        source_name = blob.source or blob.path or "document.pdf"
        password = self.password or (
            blob.metadata.get("password") if blob.metadata else None
        )

        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except fitz.FileDataError as e:
            raise ValueError(
                f"The uploaded file '{source_name}' is corrupted, invalid, or not a valid PDF document."
            ) from e

        try:
            if doc.is_encrypted and doc.needs_pass:
                if not password:
                    raise ValueError(
                        f"The uploaded document '{source_name}' is password-protected "
                        "and cannot be parsed without a valid password."
                    )
                if doc.authenticate(password) == 0:
                    raise ValueError(
                        f"Invalid password provided for encrypted document '{source_name}'."
                    )

            doc_metadata = dict(doc.metadata) if doc.metadata else {}
            raw_chunks = pymupdf4llm.to_markdown(doc, page_chunks=True)
        finally:
            doc.close()

        total_pages = len(raw_chunks)
        for idx, chunk in enumerate(raw_chunks):
            text = chunk.get("text", "").strip()
            if not text:
                continue

            meta = dict(doc_metadata)
            meta.update(chunk.get("metadata", {}))

            page_num = meta.get("page_number", idx + 1)
            meta["source"] = source_name
            meta["file_path"] = source_name
            meta["page"] = page_num
            meta["page_number"] = page_num
            meta["total_pages"] = meta.get("page_count", total_pages)

            yield Document(page_content=text, metadata=meta)

    def parse(self, blob: Blob) -> list[Document]:
        """Eagerly parses a PDF Blob into a list of LangChain Document objects."""
        return list(self.lazy_parse(blob))


class PyPDFBytesLoader(BaseLoader):
    """
    Official LangChain BaseLoader implementation for in-memory PDF byte streams.
    Enables streaming Document ingestion with zero temporary files and zero disk I/O.
    """

    def __init__(
        self,
        file_bytes: bytes,
        filename: str | None = None,
        password: str | None = None,
        parser: BaseBlobParser | None = None,
    ):
        self.file_bytes = file_bytes
        self.filename = filename or "document.pdf"
        self.password = password
        self.parser = parser or PyPDFToLangChainParser(password=password)

    def lazy_load(self) -> Iterator[Document]:
        """Lazily yields page Document objects from in-memory PDF bytes."""
        blob = Blob.from_data(
            data=self.file_bytes,
            path=self.filename,
            metadata={"password": self.password} if self.password else {},
        )
        yield from self.parser.lazy_parse(blob)

    def load(self) -> list[Document]:
        """Eagerly loads all page Document objects from in-memory PDF bytes."""
        return list(self.lazy_load())


def parse_pdf_to_documents(
    file_bytes: bytes,
    filename: str | None = None,
    password: str | None = None,
) -> list[Document]:
    """
    Parses a user-uploaded PDF into canonical LangChain Document objects.
    Maintains full document provenance and global header hierarchy.
    """
    loader = PyPDFBytesLoader(
        file_bytes=file_bytes, filename=filename, password=password
    )
    return loader.load()


def lazy_parse_pdf_to_documents(
    file_bytes: bytes,
    filename: str | None = None,
    password: str | None = None,
) -> Iterator[Document]:
    """
    Streaming generator variant: lazily parses PDF bytes into LangChain Document objects.
    Prevents holding full document copies in memory before downstream chunking.
    """
    loader = PyPDFBytesLoader(
        file_bytes=file_bytes, filename=filename, password=password
    )
    yield from loader.lazy_load()


def parse_pdf_bytes(
    file_bytes: bytes,
    filename: str | None = None,
    password: str | None = None,
) -> list[ParsedPage]:
    """
    Backward-compatible entry point returning list of ParsedPage TypedDicts.
    """
    docs = parse_pdf_to_documents(
        file_bytes=file_bytes, filename=filename, password=password
    )
    return [
        {
            "page_number": int(d.metadata.get("page", 1)),
            "text": d.page_content,
            "metadata": d.metadata,
        }
        for d in docs
    ]