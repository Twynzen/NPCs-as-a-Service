"""Embeddings client for generating vector representations of text.

Supports Ollama embeddings with nomic-embed-text model (768 dimensions).
Includes caching via content hash to avoid redundant embedding generation.

Key concepts:
- Embeddings are normalized vectors (L2 norm = 1) for cosine similarity
- nomic-embed-text supports 8192 token context (ideal for long conversations)
- Batch processing improves throughput for multiple texts
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class EmbeddingClient(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Vector of floats (normalized to unit length)
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of vectors (one per input text)
        """
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of embeddings."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name for tracking."""
        pass


class OllamaEmbeddings(EmbeddingClient):
    """Ollama embeddings client using nomic-embed-text.

    Uses the /api/embed endpoint (not /api/embeddings which is deprecated).
    Vectors are returned pre-normalized by Ollama.

    Configuration:
        - Model: nomic-embed-text (768 dims, 8192 context)
        - Alternative: mxbai-embed-large (1024 dims, 512 context)
        - Alternative: all-minilm (384 dims, 512 context, fastest)
    """

    # Dimension mapping for supported models
    MODEL_DIMENSIONS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
        "snowflake-arctic-embed": 1024,
    }

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        timeout: float = 30.0,
        batch_size: int = 32,
    ):
        """Initialize Ollama embeddings client.

        Args:
            base_url: Ollama API base URL
            model: Embedding model name
            timeout: Request timeout in seconds
            batch_size: Max texts per batch request
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size
        self._client: Optional[httpx.AsyncClient] = None

        # Embedding cache: hash(text) -> embedding
        self._cache: dict[str, list[float]] = {}
        self._cache_max_size = 10000

        # Determine dimensions from model
        self._dimensions = self.MODEL_DIMENSIONS.get(model.split(":")[0], 768)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text content hash."""
        return hashlib.sha256(f"{self.model}:{text}".encode()).hexdigest()

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Uses cache if available. For new texts, calls Ollama API.

        Args:
            text: Text to embed

        Returns:
            768-dimensional vector (for nomic-embed-text)
        """
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            logger.debug(f"Embedding cache hit for text: {text[:50]}...")
            return self._cache[cache_key]

        # Call Ollama API
        client = await self._get_client()

        try:
            response = await client.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model,
                    "input": text,
                },
            )
            response.raise_for_status()

            data = response.json()
            # API returns {"embeddings": [[...]]} for single input
            embedding = data["embeddings"][0]

            # Cache the result
            self._cache_embedding(cache_key, embedding)

            return embedding

        except httpx.HTTPError as e:
            logger.error(f"Ollama embedding request failed: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}") from e

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Processes in batches of batch_size. Uses cache where available.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (preserving input order)
        """
        if not texts:
            return []

        results: list[Optional[list[float]]] = [None] * len(texts)
        texts_to_embed: list[tuple[int, str]] = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                texts_to_embed.append((i, text))

        if not texts_to_embed:
            # All texts were cached
            return [r for r in results if r is not None]

        # Process uncached texts in batches
        client = await self._get_client()

        for batch_start in range(0, len(texts_to_embed), self.batch_size):
            batch = texts_to_embed[batch_start:batch_start + self.batch_size]
            batch_texts = [text for _, text in batch]

            try:
                response = await client.post(
                    f"{self.base_url}/api/embed",
                    json={
                        "model": self.model,
                        "input": batch_texts,
                    },
                )
                response.raise_for_status()

                data = response.json()
                embeddings = data["embeddings"]

                # Store results and cache
                for (original_idx, text), embedding in zip(batch, embeddings):
                    results[original_idx] = embedding
                    cache_key = self._get_cache_key(text)
                    self._cache_embedding(cache_key, embedding)

            except httpx.HTTPError as e:
                logger.error(f"Ollama batch embedding request failed: {e}")
                raise RuntimeError(f"Failed to generate batch embeddings: {e}") from e

        return [r for r in results if r is not None]

    def _cache_embedding(self, key: str, embedding: list[float]) -> None:
        """Add embedding to cache, evicting oldest if full."""
        if len(self._cache) >= self._cache_max_size:
            # Simple eviction: remove first 10%
            keys_to_remove = list(self._cache.keys())[:self._cache_max_size // 10]
            for k in keys_to_remove:
                del self._cache[k]

        self._cache[key] = embedding

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions for current model."""
        return self._dimensions

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self.model

    async def is_available(self) -> bool:
        """Check if Ollama server is available with the configured model."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                return False

            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]

            # Check if model is available
            return any(
                self.model == m or
                self.model == m.split(":")[0] or
                m.startswith(f"{self.model}:")
                for m in model_names
            )
        except httpx.RequestError:
            return False

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    def cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self._cache_max_size,
            "hit_rate": "N/A",  # Would need tracking
        }
