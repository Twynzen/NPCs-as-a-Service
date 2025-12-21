"""Context Builder - Assembles prompts for NPC generation.

Combines character definition, memory context, and conversation
into a complete prompt for the LLM.
"""

from typing import Optional
from pydantic import BaseModel

from src.characters.loader import Character
from src.memory.manager import MemoryContext
from src.engine.action_parser import ActionParser


class BuiltContext(BaseModel):
    """Complete context ready for LLM."""
    system_prompt: str
    messages: list[dict]
    estimated_tokens: int


class ContextBuilder:
    """
    Builds complete prompts for NPC interactions.

    Assembles:
    - Character personality and rules
    - Memory context (core, recall, archival)
    - Action definitions
    - Conversation history
    """

    def __init__(
        self,
        character: Character,
        action_parser: Optional[ActionParser] = None,
        max_context_tokens: int = 4096,
    ):
        self.character = character
        self.action_parser = action_parser or ActionParser()
        self.max_context_tokens = max_context_tokens

    def build(
        self,
        memory_context: MemoryContext,
        user_message: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> BuiltContext:
        """
        Build complete context for generation.

        Args:
            memory_context: Memory from MemoryManager
            user_message: Current user input
            conversation_history: Previous turns (if not using recall)

        Returns:
            BuiltContext ready for LLM
        """
        # Build system prompt
        system_parts = [
            self._build_character_section(),
            memory_context.core,
            memory_context.archival,
            self.action_parser.get_actions_prompt(),
            self._build_response_guidelines(),
        ]

        system_prompt = "\n\n".join(part for part in system_parts if part)

        # Build messages
        messages = []

        # Add conversation history
        if conversation_history:
            for msg in conversation_history[-10:]:  # Last 10 turns
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message,
        })

        # Estimate tokens
        total_text = system_prompt + " ".join(m["content"] for m in messages)
        estimated_tokens = len(total_text) // 4

        return BuiltContext(
            system_prompt=system_prompt,
            messages=messages,
            estimated_tokens=estimated_tokens,
        )

    def _build_character_section(self) -> str:
        """Build character identity section."""
        char = self.character
        sections = [f"# You are {char.name}\n{char.role}"]

        if char.backstory:
            sections.append(f"## Background\n{char.backstory}")

        if char.personality.traits:
            traits = ", ".join(char.personality.traits)
            sections.append(f"## Personality\n{traits}")

        if char.personality.speech_patterns:
            patterns = "\n".join(f"- {p}" for p in char.personality.speech_patterns)
            sections.append(f"## How you speak\n{patterns}")

        if char.knowledge_domains:
            kd = char.knowledge_domains
            knowledge = []
            if kd.expert:
                knowledge.append(f"Expert in: {', '.join(kd.expert)}")
            if kd.familiar:
                knowledge.append(f"Know about: {', '.join(kd.familiar)}")
            if kd.ignorant:
                knowledge.append(f"Don't know: {', '.join(kd.ignorant)}")
            sections.append(f"## Your knowledge\n" + "\n".join(knowledge))

        if char.behavioral_boundaries:
            bb = char.behavioral_boundaries
            rules = []
            if bb.never:
                rules.extend(f"NEVER: {n}" for n in bb.never)
            if bb.always:
                rules.extend(f"ALWAYS: {a}" for a in bb.always)
            sections.append(f"## Absolute rules\n" + "\n".join(f"- {r}" for r in rules))

        return "\n\n".join(sections)

    def _build_response_guidelines(self) -> str:
        """Build response format guidelines."""
        return """## Response Guidelines
- Stay in character at all times
- Keep responses 2-4 sentences unless more is needed
- Use *asterisks* for actions and body language
- Never break character or mention being an AI
- Never speak for the player
- Use actions when appropriate (see Available Actions above)
- React naturally to what the player says and does"""

    def build_first_message(self) -> Optional[str]:
        """Get character's opening message if defined."""
        return self.character.first_message

    def get_few_shot_examples(self) -> list[dict]:
        """Get example dialogues for few-shot prompting."""
        if not self.character.example_dialogues:
            return []

        examples = []
        for ex in self.character.example_dialogues:
            examples.append({
                "context": ex.context,
                "user": ex.customer,
                "assistant": ex.npc,
            })
        return examples
