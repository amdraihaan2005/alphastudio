"""
Retriever Module: Multi-Query Hybrid RAG Engine for Alpha Studio.

Architecture:
1. Multi-Query Decomposition: Employs GroqCloud ChatGroq (or injectable LangChain LLM)
   to decompose complex, comparative, or multi-metric prompts into 2-4 focused atomic sub-queries.
2. Parallel Hybrid Retrieval: Concurrently runs pgvector cosine similarity search and
   PostgreSQL websearch full-text search across all sub-queries using AlphaCohereEmbeddings.
3. Multi-Query RRF Fusion: Fuses and deduplicates candidate rankings across all sub-queries
   using Reciprocal Rank Fusion (multi_reciprocal_rank_fusion).
4. Batched Neighbor Context Expansion: Loads adjacent chunks (chunk_index ± 1) with optimized
   SQL batching and joinedload to preserve cross-chunk table/footnote continuity without N+1 queries.
5. LangChain Integration: Exposes `MultiQueryHybridRetriever(BaseRetriever)` yielding canonical
   LangChain `Document` objects with full provenance metadata for multi-file comparisons.
6. Non-blocking Concurrency: Synchronous database queries are safely executed in worker threads
   via `asyncio.to_thread` during async retrieval to prevent event-loop starvation.
7. Backward Compatibility: Preserves `retrieve_hybrid(...)` returning model dictionaries.
"""

import asyncio
from collections import defaultdict
import concurrent.futures
import logging
from typing import Any, Sequence
from uuid import UUID

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session, joinedload

from app.database.models.document_chunk import DocumentChunk
from app.ingest.embedder import get_embeddings_model
from app.retrieval.decomposer import decompose_query, get_groq_llm
from app.retrieval.fusion import multi_reciprocal_rank_fusion
from app.retrieval.queries import full_text_search, semantic_search

logger = logging.getLogger(__name__)

# Persistent thread pool executor for bridging async calls into synchronous contexts
# without allocating and tearing down OS threads on every invocation.
_SYNC_BRIDGE_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None


def _get_sync_bridge_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Lazily initializes and returns the shared persistent bridge executor."""
    global _SYNC_BRIDGE_EXECUTOR
    if _SYNC_BRIDGE_EXECUTOR is None or _SYNC_BRIDGE_EXECUTOR._shutdown:
        _SYNC_BRIDGE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="alpha-retriever-bridge",
        )
    return _SYNC_BRIDGE_EXECUTOR


def _run_async(coro: Any) -> Any:
    """
    Safely bridges async coroutine execution into a synchronous context.
    Reuses a persistent thread pool when called from an active event loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        future = _get_sync_bridge_executor().submit(asyncio.run, coro)
        return future.result()
    else:
        return asyncio.run(coro)


def _fetch_candidates(
    db: Session,
    sub_queries: list[str],
    query_embeddings: list[list[float]],
    candidate_limit: int,
    user_id: UUID | None,
    document_ids: Sequence[UUID] | None,
    filenames: Sequence[str] | None,
) -> list[list[DocumentChunk]]:
    """
    Executes synchronous database candidate searches (semantic + full-text)
    across all decomposed sub-queries in a worker thread.
    """
    streams: list[list[DocumentChunk]] = []
    for sub_query, query_emb in zip(sub_queries, query_embeddings):
        sem_res = semantic_search(
            db=db,
            query_embedding=query_emb,
            user_id=user_id,
            document_ids=document_ids,
            filenames=filenames,
            limit=candidate_limit,
        )
        fts_res = full_text_search(
            db=db,
            query_text=sub_query,
            user_id=user_id,
            document_ids=document_ids,
            filenames=filenames,
            limit=candidate_limit,
        )
        streams.append(sem_res)
        streams.append(fts_res)
    return streams


def _fetch_neighbors(
    db: Session,
    fused_results: list[tuple[DocumentChunk, float]],
) -> dict[tuple[UUID, int], DocumentChunk]:
    """
    Batches adjacent chunk queries (chunk_index ± 1) grouped by source document
    to minimize SQL filter expressions and load models with joinedload.
    """
    neighbor_map: dict[tuple[UUID, int], DocumentChunk] = {}
    if not fused_results:
        return neighbor_map

    doc_to_target_indices: dict[UUID, set[int]] = defaultdict(set)
    for chunk, _ in fused_results:
        if chunk.chunk_index > 0:
            doc_to_target_indices[chunk.source_document_id].add(chunk.chunk_index - 1)
        doc_to_target_indices[chunk.source_document_id].add(chunk.chunk_index + 1)

    neighbor_conditions = [
        and_(
            DocumentChunk.source_document_id == doc_id,
            DocumentChunk.chunk_index.in_(list(indices)),
        )
        for doc_id, indices in doc_to_target_indices.items()
        if indices
    ]

    if neighbor_conditions:
        neighbor_chunks = (
            db.query(DocumentChunk)
            .options(joinedload(DocumentChunk.document))
            .filter(or_(*neighbor_conditions))
            .all()
        )
        for nc in neighbor_chunks:
            neighbor_map[(nc.source_document_id, nc.chunk_index)] = nc

    return neighbor_map


def _build_result_dicts(
    fused_results: list[tuple[DocumentChunk, float]],
    neighbor_map: dict[tuple[UUID, int], DocumentChunk],
) -> list[dict[str, Any]]:
    """Formats fused chunks and their adjacent neighbors into result dictionaries."""
    results: list[dict[str, Any]] = []
    for chunk, score in fused_results:
        preceding_chunk = neighbor_map.get(
            (chunk.source_document_id, chunk.chunk_index - 1)
        )
        succeeding_chunk = neighbor_map.get(
            (chunk.source_document_id, chunk.chunk_index + 1)
        )
        results.append(
            {
                "chunk": chunk,
                "score": score,
                "preceding_chunk": preceding_chunk,
                "succeeding_chunk": succeeding_chunk,
            }
        )
    return results


async def aretrieve_hybrid_raw(
    db: Session,
    query_text: str,
    user_id: UUID | None = None,
    document_ids: Sequence[UUID] | None = None,
    filenames: Sequence[str] | None = None,
    limit: int = 5,
    fetch_neighbors: bool = True,
    use_multi_query: bool = True,
    embeddings: Embeddings | None = None,
    llm: BaseLanguageModel | None = None,
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Asynchronous core engine for Multi-Query Hybrid Retrieval.
    Offloads synchronous database I/O to worker threads to prevent event loop blocking.
    Returns list of dicts with keys: 'chunk', 'score', 'preceding_chunk', 'succeeding_chunk'.
    """
    cleaned_query = query_text.strip()
    if not cleaned_query:
        return []

    # 1. Multi-query decomposition via GroqCloud / injectable LLM
    if use_multi_query:
        active_llm = llm if llm is not None else get_groq_llm()
        sub_queries = await decompose_query(cleaned_query, llm=active_llm)
    else:
        sub_queries = [cleaned_query]

    # 2. Concurrently generate query embeddings for all sub-queries
    embed_model = embeddings if embeddings is not None else get_embeddings_model()
    if hasattr(embed_model, "aembed_query"):
        embeddings_tasks = [embed_model.aembed_query(sq) for sq in sub_queries]
        query_embeddings = await asyncio.gather(*embeddings_tasks)
    else:
        query_embeddings = [embed_model.embed_query(sq) for sq in sub_queries]

    # 3. Gather candidate streams across all sub-queries (non-blocking in worker thread)
    candidate_limit = limit * 4
    all_ranking_streams = await asyncio.to_thread(
        _fetch_candidates,
        db=db,
        sub_queries=sub_queries,
        query_embeddings=query_embeddings,
        candidate_limit=candidate_limit,
        user_id=user_id,
        document_ids=document_ids,
        filenames=filenames,
    )

    # 4. Multi-stream Reciprocal Rank Fusion
    fused_results = multi_reciprocal_rank_fusion(
        all_ranking_streams, k=k, limit=limit
    )

    # 5. Batched neighbor context loading (chunk_index ± 1) in worker thread
    neighbor_map: dict[tuple[UUID, int], DocumentChunk] = {}
    if fetch_neighbors and fused_results:
        neighbor_map = await asyncio.to_thread(
            _fetch_neighbors, db=db, fused_results=fused_results
        )

    return _build_result_dicts(fused_results, neighbor_map)


def chunk_to_langchain_document(result_dict: dict[str, Any] | Document) -> Document:
    """
    Transforms a raw retrieval result dictionary into a canonical LangChain Document
    with strict document provenance for citations and multi-document comparison.
    Idempotent if input is already a Document.
    """
    if isinstance(result_dict, Document):
        return result_dict

    chunk: DocumentChunk = result_dict["chunk"]
    score: float = result_dict["score"]
    preceding: DocumentChunk | None = result_dict.get("preceding_chunk")
    succeeding: DocumentChunk | None = result_dict.get("succeeding_chunk")

    filename = (
        chunk.document.filename
        if getattr(chunk, "document", None) and getattr(chunk.document, "filename", None)
        else "unknown.pdf"
    )

    metadata = {
        "source": filename,
        "page": chunk.page_number,
        "chunk_index": chunk.chunk_index,
        "section_name": chunk.section_name or "General",
        "score": score,
        "chunk_id": str(chunk.id),
        "document_id": str(chunk.source_document_id),
        "preceding_text": preceding.text_content if preceding else None,
        "succeeding_text": succeeding.text_content if succeeding else None,
    }

    return Document(page_content=chunk.text_content, metadata=metadata)


async def aretrieve_hybrid_documents(
    db: Session,
    query_text: str,
    user_id: UUID | None = None,
    document_ids: Sequence[UUID] | None = None,
    filenames: Sequence[str] | None = None,
    limit: int = 5,
    fetch_neighbors: bool = True,
    use_multi_query: bool = True,
    embeddings: Embeddings | None = None,
    llm: BaseLanguageModel | None = None,
    k: int = 60,
) -> list[Document]:
    """
    Asynchronously retrieves top relevant passages as canonical LangChain Document objects.
    """
    raw_results = await aretrieve_hybrid_raw(
        db=db,
        query_text=query_text,
        user_id=user_id,
        document_ids=document_ids,
        filenames=filenames,
        limit=limit,
        fetch_neighbors=fetch_neighbors,
        use_multi_query=use_multi_query,
        embeddings=embeddings,
        llm=llm,
        k=k,
    )
    return [chunk_to_langchain_document(r) for r in raw_results]


def retrieve_hybrid_documents(
    db: Session,
    query_text: str,
    user_id: UUID | None = None,
    document_ids: Sequence[UUID] | None = None,
    filenames: Sequence[str] | None = None,
    limit: int = 5,
    fetch_neighbors: bool = True,
    use_multi_query: bool = True,
    embeddings: Embeddings | None = None,
    llm: BaseLanguageModel | None = None,
    k: int = 60,
) -> list[Document]:
    """
    Synchronous wrapper returning canonical LangChain Document objects.
    """
    return _run_async(
        aretrieve_hybrid_documents(
            db=db,
            query_text=query_text,
            user_id=user_id,
            document_ids=document_ids,
            filenames=filenames,
            limit=limit,
            fetch_neighbors=fetch_neighbors,
            use_multi_query=use_multi_query,
            embeddings=embeddings,
            llm=llm,
            k=k,
        )
    )


def retrieve_hybrid(
    db: Session,
    query_text: str,
    user_id: UUID | None = None,
    document_ids: Sequence[UUID] | None = None,
    filenames: Sequence[str] | None = None,
    limit: int = 5,
    fetch_neighbors: bool = True,
    use_multi_query: bool = True,
    embeddings: Embeddings | None = None,
    llm: BaseLanguageModel | None = None,
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Backward-compatible synchronous entry point.
    Returns list of dicts with 'chunk', 'score', 'preceding_chunk', and 'succeeding_chunk'.
    """
    return _run_async(
        aretrieve_hybrid_raw(
            db=db,
            query_text=query_text,
            user_id=user_id,
            document_ids=document_ids,
            filenames=filenames,
            limit=limit,
            fetch_neighbors=fetch_neighbors,
            use_multi_query=use_multi_query,
            embeddings=embeddings,
            llm=llm,
            k=k,
        )
    )


class MultiQueryHybridRetriever(BaseRetriever):
    """
    Production-grade LangChain BaseRetriever implementation for Alpha Studio.

    Capabilities:
    - Multi-Query decomposition using GroqCloud ChatGroq (or custom injectable LLM).
    - Concurrent dense semantic search (AlphaCohereEmbeddings + pgvector cosine similarity).
    - Sparse keyword search (PostgreSQL websearch_to_tsquery).
    - Multi-Stream Reciprocal Rank Fusion (multi_reciprocal_rank_fusion).
    - Batched neighbor chunk expansion (chunk_index ± 1) with joinedload.
    - Strict document provenance metadata for multi-file comparisons and grounding.
    - Full LCEL Runnable compliance: invoke, ainvoke, batch, abatch, as_tool.
    """

    db: Any = Field(description="SQLAlchemy Database Session")
    user_id: UUID | None = Field(default=None, description="Optional user ID filter")
    document_ids: Sequence[UUID] | None = Field(default=None, description="Target document IDs")
    filenames: Sequence[str] | None = Field(default=None, description="Target document filenames")
    limit: int = Field(default=5, description="Number of documents to return", ge=1)
    fetch_neighbors: bool = Field(default=True, description="Fetch adjacent chunks")
    use_multi_query: bool = Field(default=True, description="Decompose into multiple queries")
    embeddings: Any | None = Field(default=None, description="LangChain Embeddings instance")
    llm: Any | None = Field(default=None, description="LangChain BaseLanguageModel for query decomposition")
    k: int = Field(default=60, description="Reciprocal Rank Fusion smoothing constant", ge=1)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """
        Synchronous retrieval conforming to BaseRetriever interface.
        Handles query execution and callback notifications.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        if run_manager:
            run_manager.on_text(f"Executing MultiQueryHybridRetriever for: '{cleaned_query}'")

        return retrieve_hybrid_documents(
            db=self.db,
            query_text=cleaned_query,
            user_id=self.user_id,
            document_ids=self.document_ids,
            filenames=self.filenames,
            limit=self.limit,
            fetch_neighbors=self.fetch_neighbors,
            use_multi_query=self.use_multi_query,
            embeddings=self.embeddings,
            llm=self.llm,
            k=self.k,
        )

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """
        Asynchronous retrieval conforming to BaseRetriever interface.
        Handles non-blocking query execution and async callback notifications.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        if run_manager:
            await run_manager.on_text(f"Executing async MultiQueryHybridRetriever for: '{cleaned_query}'")

        return await aretrieve_hybrid_documents(
            db=self.db,
            query_text=cleaned_query,
            user_id=self.user_id,
            document_ids=self.document_ids,
            filenames=self.filenames,
            limit=self.limit,
            fetch_neighbors=self.fetch_neighbors,
            use_multi_query=self.use_multi_query,
            embeddings=self.embeddings,
            llm=self.llm,
            k=self.k,
        )

    def get_relevant_chunks(self, query: str) -> list[dict[str, Any]]:
        """Convenience method returning raw dictionary results with DocumentChunk references."""
        return retrieve_hybrid(
            db=self.db,
            query_text=query,
            user_id=self.user_id,
            document_ids=self.document_ids,
            filenames=self.filenames,
            limit=self.limit,
            fetch_neighbors=self.fetch_neighbors,
            use_multi_query=self.use_multi_query,
            embeddings=self.embeddings,
            llm=self.llm,
            k=self.k,
        )

    async def aget_relevant_chunks(self, query: str) -> list[dict[str, Any]]:
        """Asynchronous convenience method returning raw dictionary results with DocumentChunk references."""
        return await aretrieve_hybrid_raw(
            db=self.db,
            query_text=query,
            user_id=self.user_id,
            document_ids=self.document_ids,
            filenames=self.filenames,
            limit=self.limit,
            fetch_neighbors=self.fetch_neighbors,
            use_multi_query=self.use_multi_query,
            embeddings=self.embeddings,
            llm=self.llm,
            k=self.k,
        )
