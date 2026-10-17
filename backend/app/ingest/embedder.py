"""
Embedder Module: Canonical LangChain-compatible Cohere V2 Embedding Engine.

Provides:
- `AlphaCohereEmbeddings`: First-class subclass of `langchain_core.embeddings.Embeddings`
  and `langchain_core.runnables.RunnableSerializable` that uses Cohere's `AsyncClientV2`
  with contextual prefixing, bounded concurrency throttling, dimension integrity verification,
  and exponential backoff on HTTP 429 rate limits and 5xx transient server errors.
- `get_embeddings`: Backward-compatible coroutine for batch document embedding.
- `get_query_embedding`: Backward-compatible coroutine for search query embedding.
- `get_embeddings_model`: Factory and singleton getter for the LangChain Embeddings instance.
"""

import asyncio
from collections.abc import Coroutine, Sequence
import concurrent.futures
import logging
from typing import Any, Callable, TypeVar, Union

import cohere
from cohere.core.api_error import ApiError
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from pydantic import ConfigDict, Field, PrivateAttr, SecretStr

from app.config import settings

logger = logging.getLogger(__name__)

# Type variable preserving wrapped coroutine signature for backward-compatible decorators
F = TypeVar("F", bound=Callable[..., Coroutine[Any, Any, Any]])

# Lazy-loaded singleton holders
_cohere_client: cohere.AsyncClientV2 | None = None
_embeddings_model: "AlphaCohereEmbeddings | None" = None


def get_cohere_client() -> cohere.AsyncClientV2:
    """
    Lazy-loading singleton getter function.
    Instantiates the AsyncClientV2 inside the running event loop on first call,
    preventing 'RuntimeError: Event loop is closed' across ASGI worker forks.
    """
    global _cohere_client
    if _cohere_client is None:
        if not settings.COHERE_API_KEY or not settings.COHERE_API_KEY.strip():
            raise ValueError(
                "COHERE_API_KEY is not configured in settings/environment."
            )
        _cohere_client = cohere.AsyncClientV2(api_key=settings.COHERE_API_KEY)
    return _cohere_client


def _run_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """Safely bridges an async coroutine into a synchronous context across ASGI/threads."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


class AlphaCohereEmbeddings(
    Embeddings,
    RunnableSerializable[
        Union[str, Sequence[Union[str, Document]]],
        Union[list[float], list[list[float]]],
    ],
):
    """
    Production-grade LangChain Embeddings provider powered by Cohere V2 API.

    Conforms strictly to the `langchain_core.embeddings.Embeddings` and `RunnableSerializable` interfaces:
    - Provides native asynchronous methods: `aembed_documents` and `aembed_query`.
    - Provides synchronous wrapper methods: `embed_documents` and `embed_query`.
    - Supports full LCEL invocation via `invoke` and `ainvoke`.
    - Natively supports both raw `str` and LangChain `Document` objects with automatic
      contextual prefix extraction from `doc.metadata["section_name"]`.
    - Resilient retry architecture with exponential backoff on HTTP 429 (rate limits)
      and 5xx (transient gateway/server errors). Fails fast on 4xx authorization/client errors.
    - Slices inputs to `COHERE_MAX_BATCH_SIZE` and throttles concurrent requests with `asyncio.Semaphore`.
    - Validates returned vector dimensions against `dimensions`.
    """

    model: str = Field(
        default_factory=lambda: getattr(
            settings, "EMBEDDING_MODEL", "embed-english-v3.0"
        ),
        description="Cohere embedding model identifier",
    )
    dimensions: int = Field(
        default_factory=lambda: getattr(settings, "EMBEDDING_DIMENSIONS", 1024),
        description="Expected embedding vector dimensionality",
    )
    cohere_api_key: str | SecretStr | None = Field(
        default=None,
        description="Explicit Cohere API key override; defaults to settings.COHERE_API_KEY",
    )
    max_batch_size: int = Field(
        default_factory=lambda: getattr(settings, "COHERE_MAX_BATCH_SIZE", 96),
        description="Maximum number of texts embedded per Cohere API call",
    )
    max_concurrency: int = Field(
        default_factory=lambda: getattr(
            settings, "MAX_CONCURRENT_EMBED_REQUESTS", 5
        ),
        description="Maximum concurrent HTTP embedding requests to avoid rate limits",
    )
    max_retries: int = Field(
        default_factory=lambda: getattr(settings, "GET_EMBEDDINGS_RETRIES", 3),
        description="Maximum retry attempts on retryable rate-limit or transient server errors",
    )
    base_delay: float = Field(
        default_factory=lambda: getattr(settings, "GET_EMBEDDINGS_DELAY", 2.0),
        description="Base backoff delay in seconds for exponential backoff (delay * 2^attempt)",
    )
    user_agent: str = Field(
        default="AlphaStudio-LangChain/1.0",
        description="Client user-agent identifier sent in API requests",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True, populate_by_name=True
    )

    _client: Any = PrivateAttr(default=None)
    _async_client: Any = PrivateAttr(default=None)

    def get_async_client(self) -> cohere.AsyncClientV2:
        """Resolves the active AsyncClientV2 instance."""
        if self._async_client is not None:
            return self._async_client
        if self.cohere_api_key:
            key = (
                self.cohere_api_key.get_secret_value()
                if isinstance(self.cohere_api_key, SecretStr)
                else self.cohere_api_key
            )
            return cohere.AsyncClientV2(
                api_key=key, client_name=self.user_agent
            )
        return get_cohere_client()

    async def aembed_with_retry(
        self, texts: list[str], input_type: str = "search_document"
    ) -> list[list[float]]:
        """
        Executes a bounded batch embed call against Cohere AsyncClientV2 with exponential backoff.
        """
        client = self.get_async_client()
        max_retries = self.max_retries
        base_delay = self.base_delay

        for attempt in range(max_retries):
            try:
                response = await client.embed(
                    texts=texts,
                    model=self.model,
                    input_type=input_type,
                )
                embeddings = response.embeddings.float
                if not embeddings:
                    raise ValueError(
                        "Cohere API response did not contain float embeddings."
                    )

                expected_dim = self.dimensions
                for idx, emb in enumerate(embeddings):
                    if len(emb) != expected_dim:
                        raise ValueError(
                            f"Embedding dimension mismatch at index {idx}. "
                            f"Expected {expected_dim}, got {len(emb)}"
                        )

                return embeddings

            except ApiError as e:
                status_code = getattr(e, "status_code", None)
                # Retry on 429 (rate limits) and 5xx (transient gateway/server errors)
                is_retryable = status_code == 429 or (
                    status_code is not None and status_code >= 500
                )
                if is_retryable and attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"Cohere API retryable error ({status_code}) in aembed_with_retry. "
                        f"Retrying in {delay:.1f}s (Attempt {attempt + 1}/{max_retries})..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Cohere API error ({status_code}) in aembed_with_retry: {e}",
                        exc_info=True,
                    )
                    raise e
            except Exception as e:
                logger.error(
                    f"Unexpected error in aembed_with_retry: {e}", exc_info=True
                )
                raise e

        raise RuntimeError(
            "aembed_with_retry exceeded max retries without response."
        )

    async def aembed_documents(
        self,
        texts: Sequence[Union[str, Document]],
        prefixes: list[str | None] | None = None,
    ) -> list[list[float]]:
        """
        Asynchronously embed a sequence of documents or strings.

        Args:
            texts: Sequence of text strings or LangChain Document objects.
            prefixes: Optional list of contextual prefix strings aligned with texts.
                      If not provided and items are Documents, `section_name` metadata is used.

        Returns:
            List of float vector embeddings aligned 1:1 with input items.
        """
        if not texts:
            return []

        if prefixes and len(prefixes) != len(texts):
            raise ValueError(
                "Length of prefixes must match length of texts if provided."
            )

        embedded_inputs: list[str] = []
        for idx, item in enumerate(texts):
            if isinstance(item, Document):
                text_content = item.page_content
                prefix = (
                    prefixes[idx]
                    if prefixes
                    else item.metadata.get("section_name")
                )
            else:
                text_content = str(item)
                prefix = prefixes[idx] if prefixes else None

            if isinstance(prefix, str) and prefix.strip():
                embedded_inputs.append(f"{prefix.strip()}\n\n{text_content}")
            else:
                embedded_inputs.append(text_content)

        max_batch_size = self.max_batch_size
        max_concurrency = self.max_concurrency

        batches: list[list[str]] = [
            embedded_inputs[i : i + max_batch_size]
            for i in range(0, len(embedded_inputs), max_batch_size)
        ]

        logger.info(
            f"Generating Cohere embeddings for {len(embedded_inputs)} items "
            f"across {len(batches)} batches (model={self.model}, "
            f"max_batch_size={max_batch_size}, max_concurrency={max_concurrency})."
        )

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _embed_batch_bounded(
            batch_texts: list[str],
        ) -> list[list[float]]:
            async with semaphore:
                return await self.aembed_with_retry(
                    batch_texts, input_type="search_document"
                )

        batch_results: list[list[list[float]]] = await asyncio.gather(
            *[_embed_batch_bounded(b) for b in batches]
        )

        return [emb for single_batch in batch_results for emb in single_batch]

    async def aembed_query(self, text: str) -> list[float]:
        """
        Asynchronously embed a search query string with input_type='search_query'.
        """
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty.")
        embeddings = await self.aembed_with_retry(
            [text.strip()], input_type="search_query"
        )
        if not embeddings or not embeddings[0]:
            raise ValueError(
                "Cohere API response did not contain query embedding."
            )
        return embeddings[0]

    def embed_documents(
        self,
        texts: Sequence[Union[str, Document]],
        prefixes: list[str | None] | None = None,
    ) -> list[list[float]]:
        """
        Synchronous embedding conforming to LangChain Embeddings interface.
        Safely bridges to aembed_documents across synchronous worker threads or scripts.
        """
        return _run_sync(self.aembed_documents(texts, prefixes=prefixes))

    def embed_query(self, text: str) -> list[float]:
        """
        Synchronous query embedding conforming to LangChain Embeddings interface.
        """
        return _run_sync(self.aembed_query(text))

    def invoke(
        self,
        input: Union[str, Sequence[Union[str, Document]]],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Union[list[float], list[list[float]]]:
        """LCEL Runnable synchronous execution."""
        if isinstance(input, str):
            return self.embed_query(input)
        return self.embed_documents(input)

    async def ainvoke(
        self,
        input: Union[str, Sequence[Union[str, Document]]],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Union[list[float], list[list[float]]]:
        """LCEL Runnable asynchronous execution."""
        if isinstance(input, str):
            return await self.aembed_query(input)
        return await self.aembed_documents(input)


def get_embeddings_model() -> AlphaCohereEmbeddings:
    """Singleton getter for the LangChain-compatible AlphaCohereEmbeddings instance."""
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = AlphaCohereEmbeddings()
    return _embeddings_model


async def get_embeddings(
    texts: Sequence[Union[str, Document]],
    prefixes: list[str | None] | None = None,
) -> list[list[float]]:
    """
    Backward-compatible coroutine for generating dense vector embeddings.
    Delegates directly to AlphaCohereEmbeddings.aembed_documents.
    """
    model = get_embeddings_model()
    return await model.aembed_documents(texts, prefixes=prefixes)


async def get_query_embedding(query_text: str) -> list[float]:
    """
    Backward-compatible coroutine for generating a search query embedding.
    Delegates directly to AlphaCohereEmbeddings.aembed_query.
    """
    model = get_embeddings_model()
    return await model.aembed_query(query_text)


async def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Backward-compatible helper coroutine delegating to the singleton model."""
    model = get_embeddings_model()
    return await model.aembed_with_retry(
        texts, input_type="search_document"
    )


async def _embed_query_raw(query_text: str) -> list[float]:
    """Backward-compatible helper coroutine delegating to the singleton model."""
    model = get_embeddings_model()
    return await model.aembed_query(query_text)
