"""
Ingest Orchestrator: Canonical LangChain sync-to-async ingestion pipeline engine.

Architecture (Pipelined Producer-Consumer Stream):
1. The Producer (Worker Thread via ThreadPoolExecutor):
   - Streams PDF page Documents via PyPDFToLangChainParser (inherits BaseBlobParser).
   - Streams structure-aware chunk Documents via RecursiveLangChainChunker (inherits BaseDocumentTransformer).
   - Bundles chunks into batches of up to COHERE_MAX_BATCH_SIZE (96) and pushes them thread-safely
     to an asyncio.Queue.
   - Monitors a threading.Event for cancellation or consumer errors, exiting immediately without resource leakage.
2. The Consumer (Async Loop):
   - Dequeues chunk batches as soon as they become available from the producer.
   - Enriches contextual section breadcrumbs ("Document: {filename} | Section: {section_name}").
   - Bounds concurrent outbound HTTP embedding requests using asyncio.Semaphore to prevent rate limits.
   - Dispatches embedding tasks immediately upon batch availability without waiting for the entire document to finish chunking.
   - Preserves canonical LangChain Document objects throughout the entire pipeline.
   - Provides LangChain Ingest Orchestrator runnable interface for LCEL composition.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import threading
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableConfig

from app.config import settings
from app.ingest.chunker import RecursiveLangChainChunker
from app.ingest.embedder import get_embeddings
from app.ingest.parser import parse_pdf_to_documents

logger = logging.getLogger(__name__)

# Dedicated thread pool executor for CPU-intensive PDF parsing & regex chunking
MAX_CPU_WORKERS = getattr(settings, "MAX_PDF_PARSING_THREADS", 4)
_cpu_thread_pool = ThreadPoolExecutor(max_workers=MAX_CPU_WORKERS)
_document_chunker = RecursiveLangChainChunker()


def _sync_produce_chunks(
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue,
    stop_event: threading.Event,
    file_bytes: bytes,
    filename: str,
    batch_size: int = 96,
    chunker: RecursiveLangChainChunker | None = None,
) -> None:
    """
    Synchronous CPU worker running in a background thread:
    1. Parses PDF bytes into LangChain page Documents in a single global layout pass.
    2. Lazily chunks pages with table integrity and section breadcrumb preservation.
    3. Bundles chunks into batches of size `batch_size` and pushes them thread-safely
       to the async queue so the consumer can dispatch embeddings concurrently.
    4. Respects `stop_event` to abort immediately if the consumer cancels or fails.
    """
    try:
        page_docs = parse_pdf_to_documents(file_bytes, filename=filename)
        active_chunker = chunker or _document_chunker
        chunk_stream = active_chunker.lazy_transform_documents(page_docs)

        batch: list[Document] = []
        for chunk_doc in chunk_stream:
            if stop_event.is_set():
                logger.debug(
                    f"Stop event detected for '{filename}', halting producer thread."
                )
                return

            batch.append(chunk_doc)
            if len(batch) >= batch_size:
                loop.call_soon_threadsafe(queue.put_nowait, (batch, None))
                batch = []

        if batch and not stop_event.is_set():
            loop.call_soon_threadsafe(queue.put_nowait, (batch, None))

        if not stop_event.is_set():
            # Push EOF sentinel: (None, None)
            loop.call_soon_threadsafe(queue.put_nowait, (None, None))

    except Exception as exc:
        if not stop_event.is_set():
            logger.warning(
                f"Error in sync chunk producer for '{filename}': {exc}"
            )
            loop.call_soon_threadsafe(queue.put_nowait, (None, exc))


class LangChainIngestOrchestrator(
    Runnable[tuple[bytes, str], tuple[list[Document], list[list[float]]]]
):
    """
    Canonical LangChain Ingestion Pipeline Orchestrator & LCEL Runnable.
    Coordinates asynchronous sync-to-async pipelined document ingestion:
    PDF Bytes -> LangChain Document Pages -> Chunk Documents -> Vector Embeddings.
    """

    def __init__(
        self,
        chunker: RecursiveLangChainChunker | None = None,
        batch_size: int | None = None,
        max_concurrency: int | None = None,
        max_cpu_workers: int | None = None,
    ) -> None:
        self.chunker = chunker or _document_chunker
        self.batch_size = (
            batch_size
            if batch_size is not None
            else getattr(settings, "COHERE_MAX_BATCH_SIZE", 96)
        )
        self.max_concurrency = (
            max_concurrency
            if max_concurrency is not None
            else getattr(settings, "MAX_CONCURRENT_EMBED_REQUESTS", 5)
        )
        self.max_cpu_workers = (
            max_cpu_workers
            if max_cpu_workers is not None
            else getattr(settings, "MAX_PDF_PARSING_THREADS", 4)
        )

    async def aprocess(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> tuple[list[Document], list[list[float]]]:
        """
        Pipelined ASGI-safe ingestion engine implementing Producer-Consumer overlap:
        1. Spawns sync producer in ThreadPoolExecutor to parse and chunk pages.
        2. Concurrently consumes chunk batches from asyncio.Queue, throttling requests
           via asyncio.Semaphore to protect against rate limits.
        3. Gathers all batch embeddings concurrently via asyncio.gather.
        4. Returns canonical LangChain Document chunks aligned 1:1 with dense vector embeddings.
        """
        if not file_bytes:
            raise ValueError("Cannot process empty (0 bytes) PDF payload.")

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[list[Document] | None, Exception | None]] = (
            asyncio.Queue()
        )
        stop_event = threading.Event()
        batch_size = self.batch_size
        semaphore = asyncio.Semaphore(self.max_concurrency)

        # Launch CPU producer in the thread pool
        producer_future = loop.run_in_executor(
            _cpu_thread_pool,
            _sync_produce_chunks,
            loop,
            queue,
            stop_event,
            file_bytes,
            filename,
            batch_size,
            self.chunker,
        )

        async def _embed_batch_throttled(
            batch_texts: list[str],
            batch_prefixes: list[str],
        ) -> list[list[float]]:
            async with semaphore:
                return await get_embeddings(
                    batch_texts, prefixes=batch_prefixes
                )

        pending_batches: list[list[Document]] = []
        pending_tasks: list[asyncio.Task[list[list[float]]]] = []

        try:
            while True:
                batch, err = await queue.get()
                if err is not None:
                    raise err
                if batch is None:
                    break

                # Prepare texts and semantic prefixes for this batch
                texts = [doc.page_content for doc in batch]
                prefixes = []
                for doc in batch:
                    section = doc.metadata.get("section_name") or "General"
                    prefix = f"Document: {filename} | Section: {section}"
                    doc.metadata["section_name"] = prefix
                    prefixes.append(prefix)

                # Fire off rate-throttled async Cohere embed request immediately
                embed_task = asyncio.create_task(
                    _embed_batch_throttled(texts, prefixes)
                )
                pending_batches.append(batch)
                pending_tasks.append(embed_task)

            # Await producer thread to guarantee clean completion
            await producer_future

            # Concurrently await all in-flight embedding tasks via asyncio.gather
            batch_results: list[list[list[float]]] = (
                await asyncio.gather(*pending_tasks) if pending_tasks else []
            )

            # Flatten in strictly preserved sequential batch order
            all_chunk_docs: list[Document] = [
                doc for batch in pending_batches for doc in batch
            ]
            all_embeddings: list[list[float]] = [
                emb for b_embs in batch_results for emb in b_embs
            ]

            if not all_chunk_docs:
                logger.warning(
                    f"Pipeline completed with 0 chunks generated from '{filename}'."
                )
                return [], []

            if len(all_chunk_docs) != len(all_embeddings):
                raise RuntimeError(
                    f"Embedding count mismatch: generated {len(all_chunk_docs)} chunks "
                    f"but received {len(all_embeddings)} embeddings."
                )

            logger.info(
                f"Successfully processed and embedded '{filename}': {len(all_chunk_docs)} chunks "
                f"across {len(pending_tasks)} throttled batches."
            )
            return all_chunk_docs, all_embeddings

        except Exception:
            stop_event.set()
            for task in pending_tasks:
                if not task.done():
                    task.cancel()
            raise
        finally:
            stop_event.set()

    def invoke(
        self,
        input: tuple[bytes, str],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> tuple[list[Document], list[list[float]]]:
        """LCEL synchronous invocation entrypoint."""
        import asyncio

        return asyncio.run(self.aprocess(input[0], input[1]))

    async def ainvoke(
        self,
        input: tuple[bytes, str],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> tuple[list[Document], list[list[float]]]:
        """LCEL asynchronous invocation entrypoint."""
        return await self.aprocess(input[0], input[1])


default_orchestrator = LangChainIngestOrchestrator()


async def process_pdf_to_embeddings(
    file_bytes: bytes,
    filename: str,
) -> tuple[list[Document], list[list[float]]]:
    """
    Public entrypoint for processing PDF bytes into LangChain Documents and vector embeddings.
    Delegates directly to default_orchestrator.aprocess.
    """
    return await default_orchestrator.aprocess(
        file_bytes=file_bytes, filename=filename
    )
