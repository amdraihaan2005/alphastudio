from typing import TypeVar, List, Tuple

T = TypeVar("T")


def reciprocal_rank_fusion(
    semantic_results: List[T], fts_results: List[T], k: int = 60, limit: int = 20
) -> List[Tuple[T, float]]:
    """
    Combines results from semantic search and full-text search using
    the Reciprocal Rank Fusion (RRF) algorithm.

    Formula: RRF_Score(doc) = sum(1 / (k + rank_i)) for rank_i in result_rankings.
    Returns a sorted list of Tuples (document_chunk_object, rrf_score) descending.
    """
    scores = {}
    chunk_map = {}

    for index, chunk in enumerate(semantic_results):
        chunk_id = chunk.id
        chunk_map[chunk_id] = chunk
        rank = index + 1
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (k + rank))

    for index, chunk in enumerate(fts_results):
        chunk_id = chunk.id
        chunk_map[chunk_id] = chunk
        rank = index + 1
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (k + rank))

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    fused_results = []
    for chunk_id, score in sorted_scores[:limit]:
        fused_results.append((chunk_map[chunk_id], score))

    return fused_results
