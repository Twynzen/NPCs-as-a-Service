"""Hybrid retriever combining vector search and BM25 for optimal recall.

Key concepts:
- Vector search: Captures semantic similarity (synonyms, paraphrasing)
- BM25: Captures exact term matching (proper nouns, IDs, specific terms)
- Hybrid: Combines both using Reciprocal Rank Fusion (RRF)

Why hybrid search is necessary:
- Vector search fails with proper nouns: "Marcus" embeds similar to other names
- BM25 fails with synonyms: "casa" won't match "hogar"
- Hybrid captures both: exact matches AND semantic relationships

RRF Formula: score(d) = sum(1/(k + rank_i(d))) where k=60 (empirical constant)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .embeddings import EmbeddingClient
from .vector_store import VectorStore, VectorPoint

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from hybrid search."""
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector_score: float = 0.0  # Cosine similarity (0-1)
    bm25_score: float = 0.0    # Normalized BM25 score (0-1)
    rrf_score: float = 0.0     # Final RRF score
    vector_rank: int = 0       # Rank in vector results
    bm25_rank: int = 0         # Rank in BM25 results


class BM25Index:
    """BM25 (Best Matching 25) keyword search index.

    BM25 scoring formula:
    score(D, Q) = sum(IDF(qi) * TF(qi,D) * (k1+1) / (TF(qi,D) + k1*(1-b+b*|D|/avgdl)))

    where:
    - TF: Term frequency (with saturation - 10th mention < 1st mention)
    - IDF: Inverse document frequency (rare terms score higher)
    - k1: Controls TF saturation (default 1.2)
    - b: Length normalization (default 0.75)

    Note: This is a simplified in-memory implementation.
    For production with large datasets, use rank-bm25 or Elasticsearch.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """Initialize BM25 index.

        Args:
            k1: TF saturation parameter (higher = more weight to repetitions)
            b: Length normalization (higher = more penalty for long docs)
        """
        self.k1 = k1
        self.b = b
        self._documents: dict[str, str] = {}  # id -> content
        self._tokenized: dict[str, list[str]] = {}  # id -> tokens
        self._doc_lengths: dict[str, int] = {}  # id -> token count
        self._avg_doc_length: float = 0.0
        self._doc_count: int = 0
        self._term_doc_counts: dict[str, int] = {}  # term -> doc count

    def add(self, doc_id: str, content: str) -> None:
        """Add document to index."""
        tokens = self._tokenize(content)
        self._documents[doc_id] = content
        self._tokenized[doc_id] = tokens
        self._doc_lengths[doc_id] = len(tokens)

        # Update term document frequencies
        unique_terms = set(tokens)
        for term in unique_terms:
            self._term_doc_counts[term] = self._term_doc_counts.get(term, 0) + 1

        # Update average doc length
        self._doc_count += 1
        self._avg_doc_length = sum(self._doc_lengths.values()) / self._doc_count

    def remove(self, doc_id: str) -> None:
        """Remove document from index."""
        if doc_id not in self._documents:
            return

        tokens = self._tokenized[doc_id]
        unique_terms = set(tokens)

        for term in unique_terms:
            self._term_doc_counts[term] = max(0, self._term_doc_counts.get(term, 1) - 1)

        del self._documents[doc_id]
        del self._tokenized[doc_id]
        del self._doc_lengths[doc_id]

        self._doc_count = max(0, self._doc_count - 1)
        if self._doc_count > 0:
            self._avg_doc_length = sum(self._doc_lengths.values()) / self._doc_count
        else:
            self._avg_doc_length = 0.0

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search index and return (doc_id, score) tuples.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) sorted by score descending
        """
        if not self._documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: dict[str, float] = {}

        for doc_id, doc_tokens in self._tokenized.items():
            score = 0.0
            doc_length = self._doc_lengths[doc_id]

            for term in query_tokens:
                # Term frequency in document
                tf = doc_tokens.count(term)
                if tf == 0:
                    continue

                # Inverse document frequency
                doc_freq = self._term_doc_counts.get(term, 0)
                if doc_freq == 0:
                    continue

                # IDF with smoothing
                import math
                idf = math.log(
                    (self._doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1
                )

                # BM25 TF component with length normalization
                tf_component = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (
                        1 - self.b + self.b * (doc_length / max(self._avg_doc_length, 1))
                    )
                )

                score += idf * tf_component

            if score > 0:
                scores[doc_id] = score

        # Sort by score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase and split on whitespace/punctuation."""
        import re
        # Split on non-alphanumeric, keep only words
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def clear(self) -> None:
        """Clear the index."""
        self._documents.clear()
        self._tokenized.clear()
        self._doc_lengths.clear()
        self._term_doc_counts.clear()
        self._doc_count = 0
        self._avg_doc_length = 0.0

    def count(self) -> int:
        """Get document count."""
        return self._doc_count


class HybridRetriever:
    """Hybrid retriever combining vector search and BM25.

    Architecture:
    1. Generate query embedding
    2. Run vector search (semantic)
    3. Run BM25 search (keyword)
    4. Combine results using RRF

    Configuration:
        vector_weight: Weight for vector search (default 0.7)
        bm25_weight: Weight for BM25 search (default 0.3)
        rrf_k: RRF constant (default 60, empirically optimal)
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
        collection_name: str,
        rrf_k: int = 60,
        over_fetch_factor: int = 3,
    ):
        """Initialize hybrid retriever.

        Args:
            embedding_client: Client for generating embeddings
            vector_store: Vector database for similarity search
            collection_name: Collection/index name
            rrf_k: RRF constant (higher = less emphasis on top ranks)
            over_fetch_factor: How many extra results to fetch for RRF
        """
        self.embeddings = embedding_client
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.rrf_k = rrf_k
        self.over_fetch_factor = over_fetch_factor

        # BM25 index (maintained separately for keyword search)
        self._bm25_index = BM25Index()

    async def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: dict[str, Any],
        embedding: Optional[list[float]] = None,
    ) -> None:
        """Add document to both vector store and BM25 index.

        Args:
            doc_id: Unique document ID
            content: Document text
            metadata: Document metadata (npc_id, player_id, importance, etc.)
            embedding: Pre-computed embedding (optional, will generate if not provided)
        """
        # Generate embedding if not provided
        if embedding is None:
            embedding = await self.embeddings.embed(content)

        # Add to vector store
        point = VectorPoint(
            id=doc_id,
            vector=embedding,
            payload={
                **metadata,
                "content": content,  # Store content in payload for retrieval
            },
        )
        await self.vector_store.add(self.collection_name, [point])

        # Add to BM25 index
        self._bm25_index.add(doc_id, content)

    async def add_documents_batch(
        self,
        documents: list[tuple[str, str, dict[str, Any]]],
    ) -> None:
        """Batch add documents.

        Args:
            documents: List of (doc_id, content, metadata) tuples
        """
        if not documents:
            return

        # Generate embeddings in batch
        contents = [content for _, content, _ in documents]
        embeddings = await self.embeddings.embed_batch(contents)

        # Add to vector store
        points = [
            VectorPoint(
                id=doc_id,
                vector=embedding,
                payload={**metadata, "content": content},
            )
            for (doc_id, content, metadata), embedding in zip(documents, embeddings)
        ]
        await self.vector_store.add(self.collection_name, points)

        # Add to BM25 index
        for doc_id, content, _ in documents:
            self._bm25_index.add(doc_id, content)

    async def remove_document(self, doc_id: str) -> None:
        """Remove document from both indexes."""
        await self.vector_store.delete(self.collection_name, [doc_id])
        self._bm25_index.remove(doc_id)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
        use_hybrid: bool = True,
    ) -> list[SearchResult]:
        """Search for relevant documents using hybrid approach.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters (e.g., npc_id, player_id)
            use_hybrid: Whether to use BM25 (False = vector only)

        Returns:
            List of SearchResult sorted by relevance (RRF score)
        """
        # Over-fetch for better RRF results
        fetch_k = top_k * self.over_fetch_factor

        # 1. Vector search
        query_embedding = await self.embeddings.embed(query)
        vector_results = await self.vector_store.search(
            collection=self.collection_name,
            query_vector=query_embedding,
            top_k=fetch_k,
            filters=filters,
        )

        # Build result map
        result_map: dict[str, SearchResult] = {}
        for rank, point in enumerate(vector_results, start=1):
            result_map[point.id] = SearchResult(
                id=point.id,
                content=point.payload.get("content", ""),
                metadata={k: v for k, v in point.payload.items() if k != "content"},
                vector_score=point.score,
                vector_rank=rank,
            )

        # 2. BM25 search (if enabled)
        if use_hybrid:
            bm25_results = self._bm25_index.search(query, top_k=fetch_k)

            # Normalize BM25 scores to 0-1
            max_bm25 = max(score for _, score in bm25_results) if bm25_results else 1.0

            for rank, (doc_id, score) in enumerate(bm25_results, start=1):
                normalized_score = score / max_bm25 if max_bm25 > 0 else 0.0

                if doc_id in result_map:
                    result_map[doc_id].bm25_score = normalized_score
                    result_map[doc_id].bm25_rank = rank
                else:
                    # Document found by BM25 but not vector search
                    # We don't have the full metadata, so skip for now
                    # In production, we'd fetch from vector store
                    pass

        # 3. Compute RRF scores
        for result in result_map.values():
            rrf_vector = 0.0
            rrf_bm25 = 0.0

            if result.vector_rank > 0:
                rrf_vector = 1.0 / (self.rrf_k + result.vector_rank)

            if result.bm25_rank > 0:
                rrf_bm25 = 1.0 / (self.rrf_k + result.bm25_rank)

            result.rrf_score = rrf_vector + rrf_bm25

        # 4. Sort by RRF score and return top-k
        sorted_results = sorted(
            result_map.values(),
            key=lambda x: x.rrf_score,
            reverse=True,
        )

        return sorted_results[:top_k]

    async def search_vector_only(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Vector-only search (no BM25).

        Use when you want pure semantic search without keyword matching.
        """
        return await self.search(query, top_k, filters, use_hybrid=False)

    async def get_all_for_context(
        self,
        npc_id: str,
        player_id: str,
        limit: int = 100,
    ) -> list[VectorPoint]:
        """Get all documents for an NPC-player pair.

        Use for building BM25 index on startup.
        """
        return await self.vector_store.scroll(
            collection=self.collection_name,
            filters={"npc_id": npc_id, "player_id": player_id},
            limit=limit,
        )

    async def rebuild_bm25_index(
        self,
        npc_id: str,
        player_id: str,
    ) -> int:
        """Rebuild BM25 index for an NPC-player pair.

        Should be called on session start to ensure BM25 index is fresh.

        Returns:
            Number of documents indexed
        """
        self._bm25_index.clear()

        points = await self.get_all_for_context(npc_id, player_id)
        for point in points:
            content = point.payload.get("content", "")
            if content:
                self._bm25_index.add(point.id, content)

        return len(points)

    def clear_bm25_index(self) -> None:
        """Clear BM25 index (call when switching NPC-player context)."""
        self._bm25_index.clear()


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Standalone RRF implementation for combining multiple rankings.

    Args:
        rankings: List of rankings, each is list of (doc_id, score)
        k: RRF constant (default 60)

    Returns:
        Fused ranking as list of (doc_id, rrf_score)

    Example:
        vector_ranking = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        bm25_ranking = [("doc3", 5.0), ("doc1", 3.0), ("doc4", 2.0)]
        fused = reciprocal_rank_fusion([vector_ranking, bm25_ranking])
        # doc1 gets: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
    """
    scores: dict[str, float] = {}

    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
