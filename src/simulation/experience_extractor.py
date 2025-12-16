"""Extract memorable experiences from conversations."""

from typing import Optional
from pydantic import BaseModel, Field

from src.simulation.conversation import ConversationResult, ConversationTurn
from src.llm.ollama_client import OllamaClient


class Experience(BaseModel):
    """An extracted experience/memory from a conversation."""
    description: str
    importance: int = Field(ge=1, le=10)
    topics: list[str]
    emotional_context: Optional[str] = None
    customer_type: Optional[str] = None
    conversation_id: str


class ExtractionResult(BaseModel):
    """Result of experience extraction."""
    conversation_id: str
    experiences: list[Experience]
    summary: str


class ExperienceExtractor:
    """Extracts memorable experiences from completed conversations."""

    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client

    def extract(
        self,
        conversation: ConversationResult,
        npc_name: str = "Zamir",
    ) -> ExtractionResult:
        """
        Extract experiences from a conversation.

        Args:
            conversation: The completed conversation
            npc_name: Name of the NPC for perspective

        Returns:
            ExtractionResult with extracted experiences
        """
        # Format conversation for analysis
        conv_text = self._format_conversation(conversation.turns)

        # Extract experiences using LLM
        experiences = self._extract_experiences(
            conv_text,
            npc_name,
            conversation.customer_profile,
            conversation.id,
        )

        # Generate summary
        summary = self._generate_summary(conv_text, npc_name)

        return ExtractionResult(
            conversation_id=conversation.id,
            experiences=experiences,
            summary=summary,
        )

    def _format_conversation(self, turns: list[ConversationTurn]) -> str:
        """Format conversation turns into readable text."""
        lines = []
        for turn in turns:
            speaker = "Customer" if turn.speaker == "customer" else "NPC"
            lines.append(f"{speaker}: {turn.message}")
        return "\n\n".join(lines)

    def _extract_experiences(
        self,
        conversation_text: str,
        npc_name: str,
        customer_profile: dict,
        conv_id: str,
    ) -> list[Experience]:
        """Extract discrete experiences from conversation."""
        prompt = f"""Analyze this conversation from {npc_name}'s perspective (a bartender and information broker).
Extract 1-3 memorable experiences or observations that {npc_name} would remember.

For each experience, provide:
1. A brief description (1-2 sentences, from {npc_name}'s perspective)
2. Importance score (1-10, where 1=mundane, 10=extremely significant)
3. Key topics (2-4 keywords)
4. Emotional context (one word: neutral, tense, friendly, hostile, curious, etc.)

Customer was: {customer_profile.get('archetype', 'unknown')} ({customer_profile.get('faction', 'unknown')} faction)
Customer objective: {customer_profile.get('objective', 'unknown')}
Customer emotional state: {customer_profile.get('emotional_state', 'unknown')}

Conversation:
{conversation_text}

Format your response EXACTLY like this (one experience per block):
---
DESCRIPTION: [description here]
IMPORTANCE: [number 1-10]
TOPICS: [topic1, topic2, topic3]
EMOTION: [emotional context]
---

Provide 1-3 experiences, each in a separate block."""

        response = self.llm.generate(
            prompt,
            temperature=0.5,
            max_tokens=600,
        )

        return self._parse_experiences(
            response,
            conv_id,
            customer_profile.get("archetype"),
        )

    def _parse_experiences(
        self,
        response: str,
        conv_id: str,
        customer_type: Optional[str],
    ) -> list[Experience]:
        """Parse LLM response into Experience objects."""
        experiences = []

        # Split by separator
        blocks = response.split("---")

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            try:
                # Parse each field
                description = ""
                importance = 5
                topics = []
                emotion = "neutral"

                for line in block.split("\n"):
                    line = line.strip()
                    if line.startswith("DESCRIPTION:"):
                        description = line.replace("DESCRIPTION:", "").strip()
                    elif line.startswith("IMPORTANCE:"):
                        try:
                            importance = int(line.replace("IMPORTANCE:", "").strip())
                            importance = max(1, min(10, importance))
                        except ValueError:
                            importance = 5
                    elif line.startswith("TOPICS:"):
                        topics_str = line.replace("TOPICS:", "").strip()
                        topics = [t.strip() for t in topics_str.split(",")]
                    elif line.startswith("EMOTION:"):
                        emotion = line.replace("EMOTION:", "").strip().lower()

                if description:
                    experiences.append(Experience(
                        description=description,
                        importance=importance,
                        topics=topics,
                        emotional_context=emotion,
                        customer_type=customer_type,
                        conversation_id=conv_id,
                    ))

            except Exception:
                # Skip malformed blocks
                continue

        # If parsing failed completely, create a generic experience
        if not experiences:
            experiences.append(Experience(
                description="A customer visited the bar for conversation.",
                importance=3,
                topics=["visit", "conversation"],
                emotional_context="neutral",
                customer_type=customer_type,
                conversation_id=conv_id,
            ))

        return experiences

    def _generate_summary(self, conversation_text: str, npc_name: str) -> str:
        """Generate a brief summary of the conversation."""
        prompt = f"""Summarize this conversation in one sentence from {npc_name}'s perspective:

{conversation_text}

Summary (one sentence):"""

        response = self.llm.generate(
            prompt,
            temperature=0.3,
            max_tokens=100,
        )

        return response.strip()
