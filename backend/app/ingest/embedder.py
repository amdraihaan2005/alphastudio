import cohere
import logging
import time
from app.config import settings

logger = logging.getLogger(__name__)

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generates dense vector embeddings using Cohere's latest ClientV2.
    Includes robust rate-limit detection (429) and exponential backoff retry.
    """
    if not texts:
        return []
        
    logger.info(f"Generating Cohere embeddings for batch of {len(texts)} chunks.")
    
    # Fail fast if API key is not present
    if not settings.COHERE_API_KEY or settings.COHERE_API_KEY.strip() == "":
        raise ValueError("COHERE_API_KEY is not configured in settings/environment.")
        
    max_retries = 5
    base_delay = 5.0  # seconds
    
    for attempt in range(max_retries):
        try:
            co = cohere.ClientV2(api_key=settings.COHERE_API_KEY)
            
            response = co.embed(
                texts=texts,
                model=settings.EMBEDDING_MODEL,
                input_type="search_document"
            )
            
            embeddings = response.embeddings.float
            if not embeddings:
                raise ValueError("Cohere API response did not contain float embeddings.")
                
            # Verify shape
            expected_dim = settings.EMBEDDING_DIMENSIONS
            for idx, emb in enumerate(embeddings):
                if len(emb) != expected_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch at index {idx}. "
                        f"Expected {expected_dim}, got {len(emb)}"
                    )
                    
            return embeddings
            
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate limit" in error_str.lower() or "too many requests" in error_str.lower()
            
            if is_rate_limit and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Cohere API rate limit hit (429). "
                    f"Retrying in {delay:.1f} seconds... (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)
            else:
                logger.error(f"Failed to generate embeddings from Cohere: {e}", exc_info=True)
                raise e

def get_query_embedding(query_text: str) -> list[float]:
    """
    Generates a dense vector embedding for a search query using Cohere's ClientV2.
    Uses input_type='search_query' as recommended for queries.
    """
    if not query_text.strip():
        raise ValueError("Query text cannot be empty.")
        
    try:
        co = cohere.ClientV2(api_key=settings.COHERE_API_KEY)
        response = co.embed(
            texts=[query_text],
            model=settings.EMBEDDING_MODEL,
            input_type="search_query"
        )
        embeddings = response.embeddings.float
        if not embeddings or not embeddings[0]:
            raise ValueError("Cohere API response did not contain query embedding.")
        return embeddings[0]
    except Exception as e:
        logger.error(f"Failed to generate query embedding: {e}", exc_info=True)
        raise e
