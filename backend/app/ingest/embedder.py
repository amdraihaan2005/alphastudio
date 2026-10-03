import asyncio
from functools import wraps
import logging
from typing import Any, Callable, Coroutine, TypeVar
import cohere
from cohere.core.api_error import ApiError
from app.config import settings

logger = logging.getLogger(__name__)

# Type variable preserving wrapped coroutine signature
F = TypeVar("F", bound=Callable[..., Coroutine[Any, Any, Any]])

# Lazy-loaded singleton holder
_cohere_client: cohere.AsyncClientV2 | None = None


def get_cohere_client() -> cohere.AsyncClientV2:
    """
    Lazy-loading singleton getter function.
    Instantiates the AsyncClientV2 inside the running event loop on first call,
    preventing 'RuntimeError: Event loop is closed' across ASGI worker forks.
    """
    global _cohere_client
    if _cohere_client is None:
        if not settings.COHERE_API_KEY or not settings.COHERE_API_KEY.strip():
            raise ValueError("COHERE_API_KEY is not configured in settings/environment.")
        _cohere_client = cohere.AsyncClientV2(api_key=settings.COHERE_API_KEY)
    return _cohere_client


def retry_with_backoff(
    retries_attr: str, delay_attr: str
) -> Callable[[F], F]:
    """
    Asynchronous decorator implementing exponential backoff resilience.
    Accepts setting attribute names (strings) instead of static values,
    resolving configuration dynamically via getattr(settings, attr) on each call.

    Catches ApiError strictly:
      - 429: Rate-limit hit -> triggers exponential backoff (base_delay * 2^attempt).
      - Other status codes (400, 401, 403, 404, etc.): Instantly raised without retry.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            max_retries: int = getattr(settings, retries_attr, 3)
            base_delay: float = getattr(settings, delay_attr, 2.0)

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except ApiError as e:
                    status_code = getattr(e, "status_code", None)
                    if status_code == 429 and attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        logger.warning(
                            f"Cohere rate limit (429) in {func.__name__}. "
                            f"Retrying in {delay:.1f}s (Attempt {attempt + 1}/{max_retries})..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Cohere API error ({status_code}) in {func.__name__}: {e}",
                            exc_info=True,
                        )
                        raise e
                except Exception as e:
                    logger.error(
                        f"Unexpected error in {func.__name__}: {e}", exc_info=True
                    )
                    raise e

        return wrapper  # type: ignore[return-value]

    return decorator


@retry_with_backoff(
    retries_attr="GET_EMBEDDINGS_RETRIES",
    delay_attr="GET_EMBEDDINGS_DELAY",
)
async def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Helper coroutine executing a single bounded batch embed call with dynamic retry."""
    client = get_cohere_client()
    response = await client.embed(
        texts=texts,
        model=settings.EMBEDDING_MODEL,
        input_type="search_document",
    )
    embeddings = response.embeddings.float
    if not embeddings:
        raise ValueError("Cohere API response did not contain float embeddings.")

    expected_dim = settings.EMBEDDING_DIMENSIONS
    for idx, emb in enumerate(embeddings):
        if len(emb) != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch at index {idx}. "
                f"Expected {expected_dim}, got {len(emb)}"
            )

    return embeddings


async def get_embeddings(
    texts: list[str], prefixes: list[str | None] | None = None
) -> list[list[float]]:
    """
    Generates dense vector embeddings using Cohere's AsyncClientV2.
    Applies contextual prefixing, safe payload chunking (COHERE_MAX_BATCH_SIZE),
    and concurrent batch execution bounded by an asyncio.Semaphore.
    Guarantees strict input-to-output ordering.
    """
    if not texts:
        return []

    if prefixes and len(prefixes) != len(texts):
        raise ValueError("Length of prefixes must match length of texts if provided.")

    embedded_inputs = (
        [
            f"{p.strip()}\n\n{t}" if isinstance(p, str) and p.strip() else t
            for p, t in zip(prefixes, texts)
        ]
        if prefixes
        else texts
    )

    max_batch_size: int = getattr(settings, "COHERE_MAX_BATCH_SIZE", 96)
    max_concurrency: int = getattr(settings, "MAX_CONCURRENT_EMBED_REQUESTS", 5)

    # Slice payload into bounded chunks
    batches: list[list[str]] = [
        embedded_inputs[i : i + max_batch_size]
        for i in range(0, len(embedded_inputs), max_batch_size)
    ]

    logger.info(
        f"Generating Cohere embeddings for {len(embedded_inputs)} chunks "
        f"across {len(batches)} batches (max_batch_size={max_batch_size}, "
        f"max_concurrency={max_concurrency})."
    )

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _embed_batch_bounded(batch_texts: list[str]) -> list[list[float]]:
        async with semaphore:
            return await _embed_batch(batch_texts)

    # Process all batches concurrently with bounded semaphore while preserving order
    batch_results: list[list[list[float]]] = await asyncio.gather(
        *[_embed_batch_bounded(b) for b in batches]
    )

    # Flatten results sequentially to strictly preserve original input ordering
    all_embeddings: list[list[float]] = [
        emb for single_batch in batch_results for emb in single_batch
    ]

    return all_embeddings


@retry_with_backoff(
    retries_attr="GET_QUERY_RETRIES",
    delay_attr="GET_QUERY_DELAY",
)
async def get_query_embedding(query_text: str) -> list[float]:
    """
    Generates a dense vector embedding for a search query using Cohere's AsyncClientV2.
    Uses input_type='search_query' with non-blocking rate-limit resilience.
    """
    if not query_text.strip():
        raise ValueError("Query text cannot be empty.")

    client = get_cohere_client()
    response = await client.embed(
        texts=[query_text],
        model=settings.EMBEDDING_MODEL,
        input_type="search_query",
    )
    embeddings = response.embeddings.float
    if not embeddings or not embeddings[0]:
        raise ValueError("Cohere API response did not contain query embedding.")

    expected_dim = settings.EMBEDDING_DIMENSIONS
    if len(embeddings[0]) != expected_dim:
        raise ValueError(
            f"Query embedding dimension mismatch. Expected {expected_dim}, got {len(embeddings[0])}"
        )

    return embeddings[0]
