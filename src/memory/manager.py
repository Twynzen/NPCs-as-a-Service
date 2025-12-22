"""Memory Manager - Orchestrates all memory systems with RAG support.

Provides a unified interface for the NPC engine to interact with
all memory layers: core, recall, and archival (with RAG).

Architecture (3-layer memory following MemGPT pattern):
┌─────────────────────────────────────────────────────────────┐
│ CAPA 1: CORE MEMORY (siempre en contexto)                   │
│ Ubicación: En el prompt del LLM                             │
│ Contenido: Personalidad NPC, relación con jugador, objetivos│
│ Actualización: Al detectar cambios significativos           │
├─────────────────────────────────────────────────────────────┤
│ CAPA 2: RECALL MEMORY (Redis en producción)                 │
│ Ubicación: Base de datos externa                            │
│ Contenido: Historial de conversación reciente               │
│ Acceso: Scroll por tiempo, últimos N mensajes               │
├─────────────────────────────────────────────────────────────┤
│ CAPA 3: ARCHIVAL MEMORY (Qdrant + RAG)                      │
│ Ubicación: Vector database                                  │
│ Contenido: Todas las memorias indexadas semánticamente      │
│ Acceso: Búsqueda híbrida (vector + BM25)                    │
└─────────────────────────────────────────────────────────────┘
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional
from pydantic import BaseModel

from .core_memory import CoreMemory
from .recall_memory import RecallMemory, ConversationTurn
from .archival_memory import ArchivalMemory, Memory

logger = logging.getLogger(__name__)


class MemoryContext(BaseModel):
    """Complete memory context for a single prompt."""
    core: str
    recall: str
    archival: str
    total_tokens_estimate: int

    def to_prompt_text(self) -> str:
        """Combine all memory into prompt text."""
        sections = []

        if self.core:
            sections.append(self.core)
        if self.archival:
            sections.append(self.archival)
        if self.recall:
            sections.append(self.recall)

        return "\n\n".join(sections)


class MemoryManager:
    """
    Unified memory management for NPCs with RAG support.

    Handles:
    - Core memory (always in context)
    - Recall memory (recent conversation)
    - Archival memory (long-term semantic with RAG)

    Provides memory editing capabilities following MemGPT pattern.
    """

    def __init__(
        self,
        vector_store=None,
        embedding_client=None,
        kv_store=None,
        use_rag: bool = True,
    ):
        """Initialize memory manager.

        Args:
            vector_store: VectorStore for archival memory (Qdrant, InMemory)
            embedding_client: EmbeddingClient for generating embeddings
            kv_store: Key-value store for core/recall (Redis, or None for in-memory)
            use_rag: Whether to use RAG for archival memory
        """
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.kv_store = kv_store
        self.use_rag = use_rag

        # In-memory fallbacks
        self._core_memories: dict[str, CoreMemory] = {}
        self._recall_memories: dict[str, RecallMemory] = {}
        self._archival_memories: dict[str, ArchivalMemory] = {}

        # Shared retriever for all archival memories (if RAG enabled)
        self._retriever = None
        self._rag_initialized = False

        # Configuration (can be overridden from Settings)
        self._decay_factor = 0.995
        self._importance_weight = 0.3
        self._recency_weight = 0.3
        self._relevance_weight = 0.4

    async def initialize_rag(self, collection_name: str = "npc_memories") -> bool:
        """Initialize RAG components.

        Call this during app startup if RAG is enabled.

        Args:
            collection_name: Qdrant collection name

        Returns:
            True if RAG initialized successfully
        """
        if self._rag_initialized:
            return True

        if not self.use_rag or not self.vector_store or not self.embedding_client:
            logger.info("RAG disabled or components not configured")
            return False

        try:
            from src.rag.retriever import HybridRetriever

            # Create collection in vector store
            await self.vector_store.create_collection(
                name=collection_name,
                dimensions=self.embedding_client.dimensions,
            )

            # Create shared retriever
            self._retriever = HybridRetriever(
                embedding_client=self.embedding_client,
                vector_store=self.vector_store,
                collection_name=collection_name,
            )

            self._rag_initialized = True
            logger.info(f"RAG initialized with collection '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            self.use_rag = False
            return False

    def configure_scoring(
        self,
        decay_factor: float = 0.995,
        importance_weight: float = 0.3,
        recency_weight: float = 0.3,
        relevance_weight: float = 0.4,
    ) -> None:
        """Configure memory scoring weights.

        Args:
            decay_factor: Recency decay per hour
            importance_weight: Weight for importance (0-1)
            recency_weight: Weight for recency (0-1)
            relevance_weight: Weight for relevance (0-1)
        """
        self._decay_factor = decay_factor
        self._importance_weight = importance_weight
        self._recency_weight = recency_weight
        self._relevance_weight = relevance_weight

    def _get_key(self, npc_id: str, player_id: str) -> str:
        """Generate storage key."""
        return f"{npc_id}:{player_id}"

    def _get_session_key(self, npc_id: str, player_id: str, session_id: str) -> str:
        """Generate session-specific key."""
        return f"{npc_id}:{player_id}:{session_id}"

    # === Core Memory ===

    def get_core(self, npc_id: str, player_id: str) -> CoreMemory:
        """Get or create core memory for this NPC-player pair."""
        key = self._get_key(npc_id, player_id)

        if key not in self._core_memories:
            self._core_memories[key] = CoreMemory(
                npc_id=npc_id,
                player_id=player_id,
            )

        return self._core_memories[key]

    def update_core(
        self,
        npc_id: str,
        player_id: str,
        updates: dict,
    ) -> CoreMemory:
        """
        Apply updates to core memory.

        Supported updates:
        - player_name: str
        - player_fact: str (adds to list)
        - relationship_delta: int
        - emotion: str
        """
        core = self.get_core(npc_id, player_id)

        if "player_name" in updates:
            core.update_player_name(updates["player_name"])

        if "player_fact" in updates:
            core.add_player_fact(updates["player_fact"])

        if "relationship_delta" in updates:
            core.update_relationship(updates["relationship_delta"])

        if "emotion" in updates:
            core.set_emotion(updates["emotion"])

        return core

    # === Recall Memory ===

    def get_recall(
        self,
        npc_id: str,
        player_id: str,
        session_id: str,
    ) -> RecallMemory:
        """Get or create recall memory for this session."""
        key = self._get_session_key(npc_id, player_id, session_id)

        if key not in self._recall_memories:
            self._recall_memories[key] = RecallMemory(
                npc_id=npc_id,
                player_id=player_id,
                session_id=session_id,
            )

        return self._recall_memories[key]

    def add_turn(
        self,
        npc_id: str,
        player_id: str,
        session_id: str,
        role: str,
        content: str,
        actions: Optional[list[dict]] = None,
    ) -> None:
        """Add a conversation turn to recall memory."""
        recall = self.get_recall(npc_id, player_id, session_id)
        recall.add_turn(role, content, actions)

        # Also update core memory interaction count
        core = self.get_core(npc_id, player_id)
        core.record_interaction()

    # === Archival Memory ===

    def get_archival(self, npc_id: str, player_id: str) -> ArchivalMemory:
        """Get or create archival memory for this NPC-player pair."""
        key = self._get_key(npc_id, player_id)

        if key not in self._archival_memories:
            self._archival_memories[key] = ArchivalMemory(
                npc_id=npc_id,
                player_id=player_id,
                retriever=self._retriever if self.use_rag else None,
                embedding_client=self.embedding_client if self.use_rag else None,
                use_rag=self.use_rag,
                decay_factor=self._decay_factor,
                importance_weight=self._importance_weight,
                recency_weight=self._recency_weight,
                relevance_weight=self._relevance_weight,
            )

        return self._archival_memories[key]

    async def add_memory(
        self,
        npc_id: str,
        player_id: str,
        content: str,
        importance: float = 5.0,
        memory_type: str = "observation",
        topics: Optional[list[str]] = None,
        emotional_context: Optional[str] = None,
    ) -> Memory:
        """Add a memory to archival storage with automatic embedding."""
        memory = Memory.create(
            npc_id=npc_id,
            player_id=player_id,
            content=content,
            importance=importance,
            memory_type=memory_type,
            topics=topics,
            emotional_context=emotional_context,
        )

        archival = self.get_archival(npc_id, player_id)
        await archival.add(memory)

        return memory

    def add_memory_sync(
        self,
        npc_id: str,
        player_id: str,
        content: str,
        importance: float = 5.0,
        memory_type: str = "observation",
        topics: Optional[list[str]] = None,
        emotional_context: Optional[str] = None,
    ) -> Memory:
        """Synchronous wrapper for add_memory (backwards compatibility)."""
        memory = Memory.create(
            npc_id=npc_id,
            player_id=player_id,
            content=content,
            importance=importance,
            memory_type=memory_type,
            topics=topics,
            emotional_context=emotional_context,
        )

        archival = self.get_archival(npc_id, player_id)
        archival.add_sync(memory)

        return memory

    async def search_memories(
        self,
        npc_id: str,
        player_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[Memory]:
        """Search archival memories using RAG."""
        archival = self.get_archival(npc_id, player_id)
        return await archival.search(query, top_k)

    def search_memories_sync(
        self,
        npc_id: str,
        player_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[Memory]:
        """Synchronous wrapper for search_memories."""
        archival = self.get_archival(npc_id, player_id)
        return archival.search_sync(query, top_k)

    # === Context Building ===

    async def get_memory_context(
        self,
        npc_id: str,
        player_id: str,
        session_id: str,
        query: str,
        max_archival: int = 5,
    ) -> MemoryContext:
        """
        Build complete memory context for a prompt using RAG.

        Combines core, recall, and relevant archival memories.
        """
        core = self.get_core(npc_id, player_id)
        recall = self.get_recall(npc_id, player_id, session_id)
        archival = self.get_archival(npc_id, player_id)

        core_text = core.to_prompt_text()
        recall_text = recall.to_prompt_text()
        archival_text = await archival.to_prompt_text(query, max_archival)

        total_tokens = (
            core.get_token_estimate() +
            recall.get_token_estimate() +
            (len(archival_text) // 4)
        )

        return MemoryContext(
            core=core_text,
            recall=recall_text,
            archival=archival_text,
            total_tokens_estimate=total_tokens,
        )

    def get_memory_context_sync(
        self,
        npc_id: str,
        player_id: str,
        session_id: str,
        query: str,
        max_archival: int = 5,
    ) -> MemoryContext:
        """Synchronous wrapper for get_memory_context."""
        core = self.get_core(npc_id, player_id)
        recall = self.get_recall(npc_id, player_id, session_id)
        archival = self.get_archival(npc_id, player_id)

        core_text = core.to_prompt_text()
        recall_text = recall.to_prompt_text()
        archival_text = archival.to_prompt_text_sync(query, max_archival)

        total_tokens = (
            core.get_token_estimate() +
            recall.get_token_estimate() +
            (len(archival_text) // 4)
        )

        return MemoryContext(
            core=core_text,
            recall=recall_text,
            archival=archival_text,
            total_tokens_estimate=total_tokens,
        )

    # === Session Management ===

    async def end_session(
        self,
        npc_id: str,
        player_id: str,
        session_id: str,
    ) -> Optional[str]:
        """
        End a session and archive important information.

        Returns summary if conversation was archived.
        """
        recall = self.get_recall(npc_id, player_id, session_id)
        summary = recall.get_summary_for_archival()

        if summary:
            await self.add_memory(
                npc_id=npc_id,
                player_id=player_id,
                content=summary,
                importance=4.0,
                memory_type="observation",
                topics=["conversation", "session"],
            )

        return summary

    def end_session_sync(
        self,
        npc_id: str,
        player_id: str,
        session_id: str,
    ) -> Optional[str]:
        """Synchronous wrapper for end_session."""
        recall = self.get_recall(npc_id, player_id, session_id)
        summary = recall.get_summary_for_archival()

        if summary:
            self.add_memory_sync(
                npc_id=npc_id,
                player_id=player_id,
                content=summary,
                importance=4.0,
                memory_type="observation",
                topics=["conversation", "session"],
            )

        return summary

    # === Stats ===

    def get_stats(self, npc_id: str, player_id: str) -> dict:
        """Get memory statistics for this NPC-player pair."""
        core = self.get_core(npc_id, player_id)
        archival = self.get_archival(npc_id, player_id)

        return {
            "player_name": core.player_info.name,
            "relationship_level": core.player_info.relationship_level,
            "interaction_count": core.player_info.interaction_count,
            "archival_memories": archival.count(),
            "known_facts": len(core.player_info.known_facts),
            "rag_enabled": self.use_rag and archival.use_rag,
            "rag_initialized": self._rag_initialized,
        }

    async def close(self) -> None:
        """Close all connections."""
        if self.vector_store:
            await self.vector_store.close()
        if self.embedding_client:
            await self.embedding_client.close()


# Factory function for creating configured MemoryManager
async def create_memory_manager(
    settings=None,
) -> MemoryManager:
    """Create a fully configured MemoryManager from settings.

    Args:
        settings: Settings object (uses global settings if None)

    Returns:
        Configured MemoryManager instance
    """
    if settings is None:
        from src.config import settings

    rag_config = settings.get_rag_config()

    # Initialize embedding client if RAG is enabled
    embedding_client = None
    vector_store = None

    if rag_config["vector_store"]["available"]:
        try:
            from src.rag.embeddings import OllamaEmbeddings
            from src.rag.vector_store import QdrantVectorStore

            embedding_client = OllamaEmbeddings(
                base_url=rag_config["embeddings"]["url"],
                model=rag_config["embeddings"]["model"],
                batch_size=rag_config["embeddings"]["batch_size"],
            )

            # Check if embeddings are available
            if await embedding_client.is_available():
                vector_store = QdrantVectorStore(
                    url=rag_config["vector_store"]["url"],
                    api_key=rag_config["vector_store"]["api_key"],
                )

                # Check if Qdrant is available
                if not await vector_store.is_available():
                    logger.warning("Qdrant not available, falling back to in-memory")
                    from src.rag.vector_store import InMemoryVectorStore
                    vector_store = InMemoryVectorStore()
            else:
                logger.warning("Ollama embeddings not available, RAG disabled")
                embedding_client = None
                vector_store = None

        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}")
            embedding_client = None
            vector_store = None

    # Create manager
    manager = MemoryManager(
        vector_store=vector_store,
        embedding_client=embedding_client,
        use_rag=embedding_client is not None and vector_store is not None,
    )

    # Configure scoring
    manager.configure_scoring(
        decay_factor=rag_config["scoring"]["recency_decay"],
        importance_weight=rag_config["scoring"]["importance_weight"],
        recency_weight=rag_config["scoring"]["recency_weight"],
        relevance_weight=rag_config["scoring"]["relevance_weight"],
    )

    # Initialize RAG if available
    if manager.use_rag:
        await manager.initialize_rag(
            collection_name=rag_config["vector_store"]["collection"]
        )

    return manager
