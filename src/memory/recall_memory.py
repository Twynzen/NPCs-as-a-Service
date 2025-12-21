"""Recall Memory - Recent conversation history.

Stores the last N turns of conversation for context continuity.
Automatically managed, no semantic search needed.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """A single turn in conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    actions: list[dict] = Field(default_factory=list)


class RecallMemory(BaseModel):
    """
    Recent conversation history for context.

    Automatically truncates to stay within token limits.
    Stored per (npc_id, player_id, session_id).
    """
    npc_id: str
    player_id: str
    session_id: str
    turns: list[ConversationTurn] = Field(default_factory=list)
    max_turns: int = 20
    created_at: datetime = Field(default_factory=datetime.now)

    def add_turn(self, role: str, content: str, actions: Optional[list[dict]] = None) -> None:
        """Add a conversation turn."""
        turn = ConversationTurn(
            role=role,
            content=content,
            actions=actions or [],
        )
        self.turns.append(turn)

        # Truncate if needed (keep most recent)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_recent(self, n: int = 10) -> list[ConversationTurn]:
        """Get the N most recent turns."""
        return self.turns[-n:]

    def to_messages(self) -> list[dict]:
        """Convert to message format for LLM."""
        return [
            {"role": turn.role, "content": turn.content}
            for turn in self.turns
        ]

    def to_prompt_text(self) -> str:
        """Convert to text for inclusion in prompt."""
        if not self.turns:
            return ""

        lines = ["## Recent conversation:"]
        for turn in self.turns[-10:]:  # Last 10 turns max
            speaker = "Player" if turn.role == "user" else "You"
            lines.append(f"{speaker}: {turn.content}")

        return "\n".join(lines)

    def get_token_estimate(self) -> int:
        """Estimate token count."""
        return len(self.to_prompt_text()) // 4

    def clear(self) -> None:
        """Clear all turns (for new session)."""
        self.turns = []

    def get_summary_for_archival(self) -> Optional[str]:
        """Generate a summary of this conversation for long-term storage."""
        if len(self.turns) < 4:
            return None

        # Simple summary: first and last exchanges
        first_user = next((t for t in self.turns if t.role == "user"), None)
        last_user = next((t for t in reversed(self.turns) if t.role == "user"), None)

        if first_user and last_user and first_user != last_user:
            return f"Conversation started with '{first_user.content[:50]}...' and ended discussing '{last_user.content[:50]}...'"
        elif first_user:
            return f"Brief conversation about '{first_user.content[:100]}...'"

        return None
