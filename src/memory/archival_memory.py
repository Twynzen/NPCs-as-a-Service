"""Archival Memory - Long-term semantic memory.

Stores important memories with embeddings for semantic search.
Falls back to keyword matching if no vector DB available.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
import hashlib


class Memory(BaseModel):
    """A single memory entry."""
    id: str
    npc_id: str
    player_id: str
    content: str
    importance: float = Field(default=5.0, ge=1.0, le=10.0)
    memory_type: str = "observation"  # observation, reflection, fact
    topics: list[str] = Field(default_factory=list)
    emotional_context: Optional[str] = None
    embedding: Optional[list[float]] = None
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
        # Generate deterministic ID from content
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

    def get_recency_score(self) -> float:
        """Calculate recency score (decays over time)."""
        hours_since = (datetime.now() - self.last_accessed).total_seconds() / 3600
        # Decay factor: 0.995 per hour (half-life ~138 hours / ~6 days)
        return 0.995 ** hours_since

    def to_prompt_text(self) -> str:
        """Format for inclusion in prompt."""
        return f"- [{self.memory_type}] {self.content}"


class ArchivalMemory:
    """
    Long-term memory store with semantic search.

    Uses vector DB if available, falls back to keyword matching.
    Implements the retrieval formula: score = recency × importance × relevance
    """

    def __init__(self, npc_id: str, player_id: str, vector_store=None):
        self.npc_id = npc_id
        self.player_id = player_id
        self.vector_store = vector_store  # Optional Qdrant/other
        self._local_memories: list[Memory] = []  # Fallback storage
        self._max_local_memories = 1000

    def add(self, memory: Memory) -> None:
        """Add a memory to the archive."""
        if self.vector_store:
            # TODO: Add to vector store with embedding
            pass

        # Always keep local copy for fallback
        self._local_memories.append(memory)

        # Prune if too many (keep most important)
        if len(self._local_memories) > self._max_local_memories:
            self._local_memories.sort(key=lambda m: m.importance, reverse=True)
            self._local_memories = self._local_memories[:self._max_local_memories]

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_importance: float = 0.0,
    ) -> list[Memory]:
        """
        Search for relevant memories.

        Uses vector similarity if available, falls back to keyword matching.
        """
        if self.vector_store:
            # TODO: Vector search
            pass

        # Fallback: keyword matching with scoring
        return self._keyword_search(query, top_k, min_importance)

    def _keyword_search(
        self,
        query: str,
        top_k: int,
        min_importance: float,
    ) -> list[Memory]:
        """Simple keyword-based search with scoring."""
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

            # Calculate final score
            recency = memory.get_recency_score()
            importance_norm = memory.importance / 10.0

            # Weighted combination (Stanford formula)
            score = (recency * 0.3) + (importance_norm * 0.3) + (relevance * 0.4)

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

    def to_prompt_text(self, query: str, max_memories: int = 5) -> str:
        """Get relevant memories formatted for prompt."""
        memories = self.search(query, top_k=max_memories)

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
