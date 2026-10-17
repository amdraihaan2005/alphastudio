"""
Reciprocal Rank Fusion (RRF) Module for Multi-Query and Hybrid Retrieval.

Provides:
- `ReciprocalRankFusion`: First-class `RunnableSerializable` implementing the Reciprocal Rank
  Fusion algorithm with support for custom stream weights, item deduplication across LangChain
  `Document` and `DocumentChunk` types, and LCEL pipeline composition.
- `multi_reciprocal_rank_fusion`: Functional entrypoint for multi-stream candidate fusion.
- `reciprocal_rank_fusion`: Backward-compatible functional entrypoint for two-stream fusion.
"""

from typing import Any, List, Sequence, Tuple, TypeVar

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from pydantic import ConfigDict, Field

T = TypeVar("T")


def extract_item_key(item: Any) -> Any:
    """
    Deterministically extracts a unique hashable identifier for an item.
    Robustly handles LangChain Document objects (preventing collisions on doc.id = None),
    database DocumentChunk models, dictionaries, and arbitrary objects.
    """
    if isinstance(item, Document):
        if item.id is not None:
            return item.id
        chunk_id = item.metadata.get("chunk_id") or item.metadata.get("id")
        if chunk_id is not None:
            return chunk_id
        source = item.metadata.get("source")
        chunk_index = item.metadata.get("chunk_index")
        if source is not None and chunk_index is not None:
            return (source, chunk_index)
        return id(item)

    if hasattr(item, "id") and getattr(item, "id") is not None:
        return getattr(item, "id")

    if isinstance(item, dict):
        if "id" in item and item["id"] is not None:
            return item["id"]
        if "chunk" in item and hasattr(item["chunk"], "id"):
            return getattr(item["chunk"], "id")

    return id(item)


class ReciprocalRankFusion(RunnableSerializable[Sequence[Sequence[Any]], List[Tuple[Any, float]]]):
    """
    LangChain RunnableSerializable implementing the Reciprocal Rank Fusion (RRF) algorithm.

    Formula:
        RRF_Score(item) = sum(weight_i / (k + rank_i)) for each stream where item appears.

    Capabilities:
    - Fuses an arbitrary number of candidate ranking streams.
    - Supports custom stream weighting (e.g. dense semantic vs sparse keyword).
    - Prevents document key collision when fusing canonical LangChain `Document` objects.
    - LCEL composable: `streams | ReciprocalRankFusion(k=60, limit=10)`.
    """

    k: int = Field(
        default=60,
        ge=1,
        description="Smoothing constant preventing high-rank saturation (standard default: 60)",
    )
    limit: int = Field(
        default=20,
        ge=1,
        description="Maximum number of fused items to return",
    )
    weights: list[float] | None = Field(
        default=None,
        description="Optional list of weighting multipliers aligned with input ranking streams",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def fuse(
        self,
        rankings: Sequence[Sequence[T]],
        weights: Sequence[float] | None = None,
    ) -> List[Tuple[T, float]]:
        """Core fusion algorithm implementation."""
        if not rankings:
            return []

        active_weights = weights if weights is not None else self.weights
        scores: dict[Any, float] = {}
        item_map: dict[Any, T] = {}

        for stream_idx, ranked_list in enumerate(rankings):
            stream_weight = (
                active_weights[stream_idx]
                if active_weights and stream_idx < len(active_weights)
                else 1.0
            )

            for index, item in enumerate(ranked_list):
                item_key = extract_item_key(item)
                if item_key not in item_map:
                    item_map[item_key] = item

                rank = index + 1
                reciprocal_rank = stream_weight * (1.0 / (self.k + rank))
                scores[item_key] = scores.get(item_key, 0.0) + reciprocal_rank

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(item_map[key], score) for key, score in sorted_scores[: self.limit]]

    def invoke(
        self,
        input: Sequence[Sequence[T]],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> List[Tuple[T, float]]:
        """Synchronous LCEL invocation."""
        return self.fuse(input)

    async def ainvoke(
        self,
        input: Sequence[Sequence[T]],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> List[Tuple[T, float]]:
        """Asynchronous LCEL invocation."""
        return self.fuse(input)


def multi_reciprocal_rank_fusion(
    rankings: Sequence[Sequence[T]],
    k: int = 60,
    limit: int = 20,
    weights: Sequence[float] | None = None,
) -> List[Tuple[T, float]]:
    """
    Fuses multiple ranked lists using the Reciprocal Rank Fusion (RRF) algorithm.
    Essential for multi-query retrieval where multiple sub-queries each generate
    their own semantic and full-text ranking streams.

    Args:
        rankings: Sequence of ranked item lists.
        k: Smoothing constant preventing high-rank saturation (standard default: 60).
        limit: Maximum number of fused items to return.
        weights: Optional weighting factors for each ranking stream.

    Returns:
        Sorted list of (item, rrf_score) tuples in descending order of relevance.
    """
    fusion = ReciprocalRankFusion(k=k, limit=limit, weights=list(weights) if weights else None)
    return fusion.fuse(rankings)


def reciprocal_rank_fusion(
    semantic_results: List[T],
    fts_results: List[T],
    k: int = 60,
    limit: int = 20,
    weights: Sequence[float] | None = None,
) -> List[Tuple[T, float]]:
    """
    Combines results from semantic search and full-text search using
    the Reciprocal Rank Fusion (RRF) algorithm.
    Delegates directly to multi_reciprocal_rank_fusion.
    """
    return multi_reciprocal_rank_fusion(
        [semantic_results, fts_results], k=k, limit=limit, weights=weights
    )
