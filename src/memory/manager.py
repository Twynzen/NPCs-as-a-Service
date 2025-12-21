"""Memory Manager - Orchestrates all memory systems.

Provides a unified interface for the NPC engine to interact with
all memory layers: core, recall, and archival.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel

from .core_memory import CoreMemory
from .recall_memory import RecallMemory, ConversationTurn
from .archival_memory import ArchivalMemory, Memory


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
    Unified memory management for NPCs.

    Handles:
    - Core memory (always in context)
    - Recall memory (recent conversation)
    - Archival memory (long-term semantic)

    Provides memory editing capabilities following MemGPT pattern.
    """

    def __init__(self, vector_store=None, kv_store=None):
        self.vector_store = vector_store  # For archival (Qdrant, etc.)
        self.kv_store = kv_store  # For core/recall (Redis, etc.)

        # In-memory fallbacks
        self._core_memories: dict[str, CoreMemory] = {}
        self._recall_memories: dict[str, RecallMemory] = {}
        self._archival_memories: dict[str, ArchivalMemory] = {}

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
                vector_store=self.vector_store,
            )

        return self._archival_memories[key]

    def add_memory(
        self,
        npc_id: str,
        player_id: str,
        content: str,
        importance: float = 5.0,
        memory_type: str = "observation",
        topics: Optional[list[str]] = None,
        emotional_context: Optional[str] = None,
    ) -> Memory:
        """Add a memory to archival storage."""
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
        archival.add(memory)

        return memory

    def search_memories(
        self,
        npc_id: str,
        player_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[Memory]:
        """Search archival memories."""
        archival = self.get_archival(npc_id, player_id)
        return archival.search(query, top_k)

    # === Context Building ===

    def get_memory_context(
        self,
        npc_id: str,
        player_id: str,
        session_id: str,
        query: str,
        max_archival: int = 5,
    ) -> MemoryContext:
        """
        Build complete memory context for a prompt.

        Combines core, recall, and relevant archival memories.
        """
        core = self.get_core(npc_id, player_id)
        recall = self.get_recall(npc_id, player_id, session_id)
        archival = self.get_archival(npc_id, player_id)

        core_text = core.to_prompt_text()
        recall_text = recall.to_prompt_text()
        archival_text = archival.to_prompt_text(query, max_archival)

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

    def end_session(
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
            self.add_memory(
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
        }
