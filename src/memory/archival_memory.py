"""Archival Memory - Long-term semantic memory with RAG support.

Stores important memories with embeddings for semantic search.
Uses hybrid retrieval (vector + BM25) for optimal recall.

Key improvements over keyword-only search:
- Semantic understanding: "hogar" matches "casa"
- Contextual relevance: "familia" retrieves "hermano", "madre"
- Hybrid search: Exact matches for proper nouns + semantic for concepts
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field
import hashlib

logger = logging.getLogger(__name__)


class Memory(BaseModel):
    """A single memory entry with embedding support."""
    id: str
    npc_id: str
    player_id: str
    content: str
    importance: float = Field(default=5.0, ge=1.0, le=10.0)
    memory_type: str = "observation"  # observation, reflection, fact
    topics: list[str] = Field(default_factory=list)
    emotional_context: Optional[str] = None
    embedding: Optional[list[float]] = None
    embedding_model: Optional[str] = None  # Track which model generated embedding
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    access_count: int = 0

    @classmethod
    def create(
        cls,
        npc_id: str,
        player_id: str,
        content: str,
        importance: float = 5.0,
        memory_type: str = "observation",
        topics: Optional[list[str]] = None,
        emotional_context: Optional[str] = None,
    ) -> "Memory":
        """Create a new memory with generated ID."""
        hash_input = f"{npc_id}:{player_id}:{content}:{datetime.now().isoformat()}"
        memory_id = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        return cls(
            id=f"mem_{memory_id}",
            npc_id=npc_id,
            player_id=player_id,
            content=content,
            importance=importance,
            memory_type=memory_type,
            topics=topics or [],
            emotional_context=emotional_context,
        )

    def touch(self) -> None:
        """Record access to this memory."""
        self.last_accessed = datetime.now()
        self.access_count += 1

    def get_recency_score(self, decay_factor: float = 0.995) -> float:
        """Calculate recency score using exponential decay.

        Formula: score = decay_factor^hours_since_access

        With decay_factor=0.995:
        - 1 hour ago: 0.995
        - 24 hours ago: 0.886
        - 1 week ago: 0.700
        - 1 month ago: 0.360
        """
        hours_since = (datetime.now() - self.last_accessed).total_seconds() / 3600
        return decay_factor ** hours_since

    def to_prompt_text(self) -> str:
        """Format for inclusion in prompt."""
        return f"- [{self.memory_type}] {self.content}"

    def to_metadata(self) -> dict[str, Any]:
        """Convert to metadata dict for vector store."""
        return {
            "npc_id": self.npc_id,
            "player_id": self.player_id,
            "importance": self.importance,
            "memory_type": self.memory_type,
            "topics": self.topics,
            "emotional_context": self.emotional_context,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
        }

    @classmethod
    def from_search_result(cls, result: "SearchResult") -> "Memory":
        """Reconstruct Memory from search result."""
        from src.rag.retriever import SearchResult

        return cls(
            id=result.id,
            npc_id=result.metadata.get("npc_id", ""),
            player_id=result.metadata.get("player_id", ""),
            content=result.content,
            importance=result.metadata.get("importance", 5.0),
            memory_type=result.metadata.get("memory_type", "observation"),
            topics=result.metadata.get("topics", []),
            emotional_context=result.metadata.get("emotional_context"),
            created_at=datetime.fromisoformat(result.metadata.get("created_at", datetime.now().isoformat())),
            last_accessed=datetime.fromisoformat(result.metadata.get("last_accessed", datetime.now().isoformat())),
            access_count=result.metadata.get("access_count", 0),
        )


class ArchivalMemory:
    """
    Long-term memory store with RAG-powered semantic search.

    Features:
    - Automatic embedding generation via Ollama
    - Hybrid search (vector + BM25) for optimal retrieval
    - Recency × Importance × Relevance scoring (Stanford formula)
    - Fallback to keyword matching if RAG not available

    Architecture:
        ┌─────────────┐     ┌──────────────┐
        │   Query     │────▶│  Embedding   │
        └─────────────┘     └──────┬───────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
            ┌───────────────┐             ┌───────────────┐
            │ Vector Search │             │ BM25 Search   │
            │ (Semantic)    │             │ (Keywords)    │
            └───────┬───────┘             └───────┬───────┘
                    │                              │
                    └──────────────┬───────────────┘
                                   ▼
                           ┌───────────────┐
                           │  RRF Fusion   │
                           └───────┬───────┘
                                   ▼
                           ┌───────────────┐
                           │ Final Ranking │
                           │ (with decay)  │
                           └───────────────┘
    """

    def __init__(
        self,
        npc_id: str,
        player_id: str,
        retriever=None,  # HybridRetriever
        embedding_client=None,  # EmbeddingClient
        use_rag: bool = True,
        decay_factor: float = 0.995,
        importance_weight: float = 0.3,
        recency_weight: float = 0.3,
        relevance_weight: float = 0.4,
    ):
        """Initialize archival memory.

        Args:
            npc_id: NPC identifier
            player_id: Player identifier
            retriever: HybridRetriever for RAG search (optional)
            embedding_client: EmbeddingClient for generating embeddings (optional)
            use_rag: Whether to use RAG (False = fallback to keyword search)
            decay_factor: Recency decay per hour (0.995 = ~6 day half-life)
            importance_weight: Weight for importance in scoring
            recency_weight: Weight for recency in scoring
            relevance_weight: Weight for relevance in scoring
        """
        self.npc_id = npc_id
        self.player_id = player_id
        self.retriever = retriever
        self.embedding_client = embedding_client
        self.use_rag = use_rag and (retriever is not None or embedding_client is not None)

        # Scoring weights (should sum to 1.0)
        self.decay_factor = decay_factor
        self.importance_weight = importance_weight
        self.recency_weight = recency_weight
        self.relevance_weight = relevance_weight

        # Fallback local storage (used when RAG not available)
        self._local_memories: list[Memory] = []
        self._max_local_memories = 1000

        # Flag to track if RAG index needs rebuild
        self._rag_initialized = False

    async def initialize_rag(self) -> None:
        """Initialize RAG components (call on first use)."""
        if self._rag_initialized or not self.retriever:
            return

        try:
            # Rebuild BM25 index for this NPC-player pair
            count = await self.retriever.rebuild_bm25_index(self.npc_id, self.player_id)
            logger.info(f"RAG initialized for {self.npc_id}:{self.player_id} with {count} memories")
            self._rag_initialized = True
        except Exception as e:
            logger.warning(f"Failed to initialize RAG: {e}. Falling back to keyword search.")
            self.use_rag = False

    async def add(self, memory: Memory) -> None:
        """Add a memory to the archive with automatic embedding.

        Args:
            memory: Memory to add
        """
        # Always keep local copy for fallback
        self._local_memories.append(memory)

        # Prune if too many (keep most important)
        if len(self._local_memories) > self._max_local_memories:
            self._local_memories.sort(key=lambda m: m.importance, reverse=True)
            self._local_memories = self._local_memories[:self._max_local_memories]

        # Add to RAG if available
        if self.use_rag and self.retriever:
            try:
                # Generate embedding if not present
                if memory.embedding is None and self.embedding_client:
                    memory.embedding = await self.embedding_client.embed(memory.content)
                    memory.embedding_model = self.embedding_client.model_name

                # Add to retriever (both vector store and BM25)
                await self.retriever.add_document(
                    doc_id=memory.id,
                    content=memory.content,
                    metadata=memory.to_metadata(),
                    embedding=memory.embedding,
                )
                logger.debug(f"Added memory to RAG: {memory.id}")
            except Exception as e:
                logger.error(f"Failed to add memory to RAG: {e}")

    def add_sync(self, memory: Memory) -> None:
        """Synchronous wrapper for add (for backwards compatibility)."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, schedule for later
            asyncio.create_task(self.add(memory))
        else:
            loop.run_until_complete(self.add(memory))

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_importance: float = 0.0,
    ) -> list[Memory]:
        """
        Search for relevant memories using hybrid RAG.

        Uses vector + BM25 search with RRF fusion, then applies
        recency × importance × relevance scoring.

        Args:
            query: Search query
            top_k: Number of results to return
            min_importance: Minimum importance threshold

        Returns:
            List of relevant memories, scored and sorted
        """
        # Initialize RAG on first search
        if self.use_rag and not self._rag_initialized:
            await self.initialize_rag()

        # Use RAG if available
        if self.use_rag and self.retriever:
            try:
                results = await self.retriever.search(
                    query=query,
                    top_k=top_k * 2,  # Over-fetch for post-filtering
                    filters={
                        "npc_id": self.npc_id,
                        "player_id": self.player_id,
                    },
                )

                # Convert to Memory objects and apply scoring
                memories = []
                for result in results:
                    memory = Memory.from_search_result(result)

                    if memory.importance < min_importance:
                        continue

                    # Apply combined scoring
                    recency = memory.get_recency_score(self.decay_factor)
                    importance_norm = memory.importance / 10.0
                    relevance = result.rrf_score * 10  # Scale RRF to roughly 0-1

                    combined_score = (
                        self.recency_weight * recency +
                        self.importance_weight * importance_norm +
                        self.relevance_weight * relevance
                    )

                    memory.touch()
                    memories.append((memory, combined_score))

                # Sort by combined score
                memories.sort(key=lambda x: x[1], reverse=True)

                return [m for m, _ in memories[:top_k]]

            except Exception as e:
                logger.error(f"RAG search failed: {e}. Falling back to keyword search.")

        # Fallback: keyword matching
        return self._keyword_search(query, top_k, min_importance)

    def search_sync(
        self,
        query: str,
        top_k: int = 5,
        min_importance: float = 0.0,
    ) -> list[Memory]:
        """Synchronous wrapper for search (for backwards compatibility)."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Can't run sync in async context - use keyword fallback
            return self._keyword_search(query, top_k, min_importance)
        return loop.run_until_complete(self.search(query, top_k, min_importance))

    def _keyword_search(
        self,
        query: str,
        top_k: int,
        min_importance: float,
    ) -> list[Memory]:
        """Fallback keyword-based search with scoring.

        Uses word overlap as relevance proxy when RAG not available.
        """
        query_words = set(query.lower().split())

        scored_memories = []
        for memory in self._local_memories:
            if memory.importance < min_importance:
                continue

            # Calculate relevance (keyword overlap)
            memory_words = set(memory.content.lower().split())
            memory_words.update(t.lower() for t in memory.topics)
            overlap = len(query_words & memory_words)
            relevance = overlap / max(len(query_words), 1)

            # Calculate final score (Stanford formula)
            recency = memory.get_recency_score(self.decay_factor)
            importance_norm = memory.importance / 10.0

            score = (
                self.recency_weight * recency +
                self.importance_weight * importance_norm +
                self.relevance_weight * relevance
            )

            if score > 0.1:  # Minimum threshold
                scored_memories.append((memory, score))

        # Sort by score and return top-k
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        results = []
        for memory, _ in scored_memories[:top_k]:
            memory.touch()
            results.append(memory)

        return results

    def get_recent(self, n: int = 10) -> list[Memory]:
        """Get N most recent memories."""
        sorted_memories = sorted(
            self._local_memories,
            key=lambda m: m.created_at,
            reverse=True,
        )
        return sorted_memories[:n]

    def get_important(self, n: int = 10, min_importance: float = 7.0) -> list[Memory]:
        """Get N most important memories."""
        important = [m for m in self._local_memories if m.importance >= min_importance]
        important.sort(key=lambda m: m.importance, reverse=True)
        return important[:n]

    async def to_prompt_text(self, query: str, max_memories: int = 5) -> str:
        """Get relevant memories formatted for prompt."""
        memories = await self.search(query, top_k=max_memories)

        if not memories:
            return ""

        lines = ["## Relevant memories:"]
        for memory in memories:
            lines.append(memory.to_prompt_text())

        return "\n".join(lines)

    def to_prompt_text_sync(self, query: str, max_memories: int = 5) -> str:
        """Synchronous version of to_prompt_text."""
        memories = self.search_sync(query, top_k=max_memories)

        if not memories:
            return ""

        lines = ["## Relevant memories:"]
        for memory in memories:
            lines.append(memory.to_prompt_text())

        return "\n".join(lines)

    def count(self) -> int:
        """Get total memory count."""
        return len(self._local_memories)

    def clear(self) -> None:
        """Clear all memories (use with caution)."""
        self._local_memories = []
        if self.retriever:
            self.retriever.clear_bm25_index()

    def get_all_memories(self) -> list[Memory]:
        """Get all memories (for migration/backup)."""
        return list(self._local_memories)

    async def migrate_to_rag(self) -> int:
        """Migrate existing local memories to RAG index.

        Call this when enabling RAG for an existing NPC with memories.

        Returns:
            Number of memories migrated
        """
        if not self.retriever or not self.embedding_client:
            return 0

        count = 0
        for memory in self._local_memories:
            try:
                # Generate embedding if needed
                if memory.embedding is None:
                    memory.embedding = await self.embedding_client.embed(memory.content)
                    memory.embedding_model = self.embedding_client.model_name

                # Add to retriever
                await self.retriever.add_document(
                    doc_id=memory.id,
                    content=memory.content,
                    metadata=memory.to_metadata(),
                    embedding=memory.embedding,
                )
                count += 1
            except Exception as e:
                logger.error(f"Failed to migrate memory {memory.id}: {e}")

        self._rag_initialized = True
        return count
