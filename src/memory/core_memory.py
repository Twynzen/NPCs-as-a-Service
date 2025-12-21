"""Core Memory - Always in context, editable by the NPC itself.

This is the NPC's "working memory" - facts about the current player
and the NPC's current emotional state. Limited size, always included
in every prompt.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class PlayerInfo(BaseModel):
    """What the NPC knows about this specific player."""
    name: Optional[str] = None
    known_facts: list[str] = Field(default_factory=list)
    relationship_level: int = Field(default=0, ge=-10, le=10)  # -10 hostile, +10 trusted
    first_met: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    interaction_count: int = 0


class NPCState(BaseModel):
    """Current state of the NPC for this conversation."""
    current_emotion: str = "neutral"
    current_goal: Optional[str] = None
    active_topics: list[str] = Field(default_factory=list)
    flags: dict[str, bool] = Field(default_factory=dict)  # Game-specific flags


class CoreMemory(BaseModel):
    """
    Core memory that's always included in the NPC's context.

    Follows MemGPT pattern: small, always present, editable.
    Target size: ~500 tokens max.
    """
    npc_id: str
    player_id: str
    player_info: PlayerInfo = Field(default_factory=PlayerInfo)
    npc_state: NPCState = Field(default_factory=NPCState)
    custom_data: dict = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=datetime.now)

    def update_player_name(self, name: str) -> None:
        """Update player's name."""
        self.player_info.name = name
        self._touch()

    def add_player_fact(self, fact: str) -> None:
        """Add a fact about the player (max 10 facts, oldest removed)."""
        if fact not in self.player_info.known_facts:
            self.player_info.known_facts.append(fact)
            if len(self.player_info.known_facts) > 10:
                self.player_info.known_facts.pop(0)
        self._touch()

    def update_relationship(self, delta: int) -> None:
        """Adjust relationship level."""
        new_level = self.player_info.relationship_level + delta
        self.player_info.relationship_level = max(-10, min(10, new_level))
        self._touch()

    def set_emotion(self, emotion: str) -> None:
        """Update NPC's current emotion."""
        self.npc_state.current_emotion = emotion
        self._touch()

    def record_interaction(self) -> None:
        """Record that an interaction happened."""
        now = datetime.now()
        if self.player_info.first_met is None:
            self.player_info.first_met = now
        self.player_info.last_seen = now
        self.player_info.interaction_count += 1
        self._touch()

    def _touch(self) -> None:
        """Update timestamp."""
        self.updated_at = datetime.now()

    def to_prompt_text(self) -> str:
        """Convert to text for inclusion in prompt."""
        lines = ["## What you know about this person:"]

        if self.player_info.name:
            lines.append(f"- Name: {self.player_info.name}")

        if self.player_info.known_facts:
            for fact in self.player_info.known_facts:
                lines.append(f"- {fact}")

        # Relationship description
        level = self.player_info.relationship_level
        if level <= -7:
            rel = "You deeply distrust this person"
        elif level <= -3:
            rel = "You are wary of this person"
        elif level <= 3:
            rel = "This person is a stranger or acquaintance"
        elif level <= 7:
            rel = "This person is a regular you somewhat trust"
        else:
            rel = "This person is a trusted friend"
        lines.append(f"- Relationship: {rel}")

        # Interaction history
        if self.player_info.interaction_count > 0:
            if self.player_info.interaction_count == 1:
                lines.append("- This is their first visit")
            else:
                lines.append(f"- They have visited {self.player_info.interaction_count} times")

        # Current state
        lines.append(f"\n## Your current state:")
        lines.append(f"- Mood: {self.npc_state.current_emotion}")

        if self.npc_state.current_goal:
            lines.append(f"- Current goal: {self.npc_state.current_goal}")

        return "\n".join(lines)

    def get_token_estimate(self) -> int:
        """Estimate token count (rough: 4 chars = 1 token)."""
        return len(self.to_prompt_text()) // 4
