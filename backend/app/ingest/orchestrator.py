import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from app.config import settings
from app.ingest.chunker import DocumentChunkData, chunk_page
from app.ingest.embedder import get_embeddings
from app.ingest.parser import parse_pdf_bytes

logger = logging.getLogger(__name__)

# Dedicated thread pool executor for CPU-intensive PDF parsing & regex chunking
MAX_CPU_WORKERS = getattr(settings, "MAX_PDF_PARSING_THREADS", 4)
_cpu_thread_pool = ThreadPoolExecutor(max_workers=MAX_CPU_WORKERS)


def _sync_parse_and_chunk(file_bytes: bytes) -> list[DocumentChunkData]:
    """
    Synchronous CPU-bound worker:
    1. Parses PDF bytes into structured Markdown pages using PyMuPDF / pymupdf4llm.
    2. Sequentially chunks each page while preserving global state:
       - Increments start_chunk_idx across page boundaries.
       - Passes and updates header_stack (breadcrumbs) across page boundaries.
    """
    pages = parse_pdf_bytes(file_bytes)
    if not pages:
        logger.warning("No extractable pages found in PDF bytes.")
        return []

    all_chunks: list[DocumentChunkData] = []
    header_stack: list[tuple[int, str]] = []
    chunk_idx: int = 0

    for page in pages:
        page_chunks, header_stack = chunk_page(
            page_text=page["text"],
            page_number=page["page_number"],
            start_chunk_idx=chunk_idx,
            header_stack=header_stack,
        )
        all_chunks.extend(page_chunks)
        chunk_idx += len(page_chunks)

    logger.info(
        f"Parsed and chunked PDF into {len(all_chunks)} chunks across {len(pages)} pages."
    )
    return all_chunks


async def process_pdf_to_embeddings(
    file_bytes: bytes,
    filename: str,
) -> tuple[list[DocumentChunkData], list[list[float]]]:
    """
    ASGI-safe end-to-end ingestion pipeline:
    1. Offloads CPU-bound PDF extraction and Markdown AST chunking to a ThreadPoolExecutor
       via loop.run_in_executor(), never blocking the async event loop.
    2. Builds enriched semantic prefixes: 'Document: {filename} | Section: {section_name}'.
    3. Concurrently embeds the chunks via Cohere AsyncClientV2 with rate-limit backoff.
    4. Updates each chunk's section_name to the enriched prefix string for persistence.
    5. Returns a strictly aligned tuple of (chunks, embeddings).
    """
    if not file_bytes:
        raise ValueError("Cannot process empty (0 bytes) PDF payload.")

    loop = asyncio.get_running_loop()

    # Offload synchronous CPU-bound parsing & chunking to the thread pool
    chunks: list[DocumentChunkData] = await loop.run_in_executor(
        _cpu_thread_pool, _sync_parse_and_chunk, file_bytes
    )

    if not chunks:
        logger.warning(f"Pipeline completed with 0 chunks generated from '{filename}'.")
        return [], []

    # Map chunks to texts and dynamically enriched contextual prefixes
    texts: list[str] = [c["text_content"] for c in chunks]
    prefixes: list[str | None] = [
        f"Document: {filename} | Section: {c['section_name'] or 'General'}"
        for c in chunks
    ]

    # Await the asynchronous Cohere embedder with true bounded concurrency
    embeddings: list[list[float]] = await get_embeddings(texts, prefixes=prefixes)

    if len(chunks) != len(embeddings):
        raise RuntimeError(
            f"Embedding count mismatch: generated {len(chunks)} chunks "
            f"but received {len(embeddings)} embeddings."
        )

    # Persist the exact enriched semantic breadcrumb in the chunk metadata
    for chunk, prefix in zip(chunks, prefixes):
        chunk["section_name"] = prefix

    return chunks, embeddings
