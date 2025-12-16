"""Character card loader supporting JSON and YAML formats."""

import json
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class BigFivePersonality(BaseModel):
    """Big Five personality traits (OCEAN model)."""
    openness: float = Field(default=0.5, ge=0, le=1)
    conscientiousness: float = Field(default=0.5, ge=0, le=1)
    extraversion: float = Field(default=0.5, ge=0, le=1)
    agreeableness: float = Field(default=0.5, ge=0, le=1)
    neuroticism: float = Field(default=0.5, ge=0, le=1)


class Personality(BaseModel):
    """Character personality definition."""
    traits: list[str] = Field(default_factory=list)
    speech_patterns: list[str] = Field(default_factory=list)
    big_five: Optional[BigFivePersonality] = None


class KnowledgeDomains(BaseModel):
    """Knowledge areas for the character."""
    expert: list[str] = Field(default_factory=list)
    familiar: list[str] = Field(default_factory=list)
    ignorant: list[str] = Field(default_factory=list)


class BehavioralBoundaries(BaseModel):
    """Behavioral constraints for the character."""
    never: list[str] = Field(default_factory=list)
    always: list[str] = Field(default_factory=list)


class ExampleDialogue(BaseModel):
    """Few-shot example dialogue."""
    context: str
    customer: str
    npc: str


class Character(BaseModel):
    """Complete character card definition."""
    name: str
    role: str
    personality: Personality
    backstory: str = ""
    knowledge_domains: Optional[KnowledgeDomains] = None
    behavioral_boundaries: Optional[BehavioralBoundaries] = None
    example_dialogues: list[ExampleDialogue] = Field(default_factory=list)
    first_message: Optional[str] = None

    def build_system_prompt(self) -> str:
        """Build the system prompt for the LLM from the character card."""
        sections = []

        # Identity
        sections.append(f"You are {self.name}, {self.role}.")

        # Backstory
        if self.backstory:
            sections.append(f"\n## Background\n{self.backstory}")

        # Personality traits
        if self.personality.traits:
            traits_str = ", ".join(self.personality.traits)
            sections.append(f"\n## Core Personality Traits\n{traits_str}")

        # Speech patterns
        if self.personality.speech_patterns:
            patterns = "\n".join(f"- {p}" for p in self.personality.speech_patterns)
            sections.append(f"\n## Speech Patterns\n{patterns}")

        # Knowledge domains
        if self.knowledge_domains:
            kd = self.knowledge_domains
            knowledge_parts = []
            if kd.expert:
                knowledge_parts.append(f"Expert in: {', '.join(kd.expert)}")
            if kd.familiar:
                knowledge_parts.append(f"Familiar with: {', '.join(kd.familiar)}")
            if kd.ignorant:
                knowledge_parts.append(f"Has no knowledge of: {', '.join(kd.ignorant)}")
            if knowledge_parts:
                sections.append(f"\n## Knowledge\n" + "\n".join(knowledge_parts))

        # Behavioral boundaries
        if self.behavioral_boundaries:
            bb = self.behavioral_boundaries
            boundaries = []
            if bb.never:
                never_list = "\n".join(f"- NEVER: {n}" for n in bb.never)
                boundaries.append(never_list)
            if bb.always:
                always_list = "\n".join(f"- ALWAYS: {a}" for a in bb.always)
                boundaries.append(always_list)
            if boundaries:
                sections.append(f"\n## Behavioral Rules\n" + "\n".join(boundaries))

        # Response format instructions
        sections.append("""
## Response Guidelines
- Stay in character at all times
- Respond naturally as this character would
- Express emotions through actions in *asterisks* when appropriate
- Keep responses 2-4 sentences unless the situation demands more
- Never break character or acknowledge being an AI
- Never speak for the customer""")

        return "\n".join(sections)

    def get_few_shot_examples(self) -> str:
        """Format example dialogues for few-shot prompting."""
        if not self.example_dialogues:
            return ""

        examples = []
        for ex in self.example_dialogues:
            examples.append(f"[Context: {ex.context}]")
            examples.append(f"Customer: {ex.customer}")
            examples.append(f"{self.name}: {ex.npc}")
            examples.append("")

        return "\n## Example Interactions\n" + "\n".join(examples)


def load_character(name: str, templates_dir: Optional[Path] = None) -> Character:
    """
    Load a character card by name.

    Args:
        name: Character name (filename without extension)
        templates_dir: Directory containing character templates.
                       Defaults to src/characters/templates/

    Returns:
        Character instance

    Raises:
        FileNotFoundError: If character card not found
        ValueError: If character card is invalid
    """
    if templates_dir is None:
        templates_dir = Path(__file__).parent / "templates"

    # Try YAML first, then JSON
    yaml_path = templates_dir / f"{name}.yaml"
    json_path = templates_dir / f"{name}.json"

    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        available = [p.stem for p in templates_dir.glob("*.yaml")] + \
                    [p.stem for p in templates_dir.glob("*.json")]
        raise FileNotFoundError(
            f"Character '{name}' not found. "
            f"Available characters: {', '.join(available) or 'none'}"
        )

    return Character.model_validate(data)


def list_characters(templates_dir: Optional[Path] = None) -> list[str]:
    """List all available character names."""
    if templates_dir is None:
        templates_dir = Path(__file__).parent / "templates"

    characters = set()
    for ext in ["yaml", "json"]:
        for path in templates_dir.glob(f"*.{ext}"):
            characters.add(path.stem)

    return sorted(characters)
