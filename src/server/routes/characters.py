"""Character management endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.characters.loader import load_character, list_characters, Character

router = APIRouter()


class CharacterSummary(BaseModel):
    """Brief character info."""
    name: str
    role: str
    traits: list[str]


class CharacterListResponse(BaseModel):
    """List of available characters."""
    characters: list[CharacterSummary]
    total: int


class CharacterDetailResponse(BaseModel):
    """Detailed character information."""
    name: str
    role: str
    backstory: str
    traits: list[str]
    speech_patterns: list[str]
    knowledge_domains: Optional[dict]
    behavioral_boundaries: Optional[dict]
    has_first_message: bool
    example_dialogues_count: int


@router.get("", response_model=CharacterListResponse)
async def get_characters():
    """
    List all available NPC characters.

    Returns a summary of each character that can be used in sessions.
    """
    character_names = list_characters()

    characters = []
    for name in character_names:
        try:
            char = load_character(name)
            characters.append(CharacterSummary(
                name=char.name,
                role=char.role,
                traits=char.personality.traits[:3],  # First 3 traits
            ))
        except Exception:
            # Skip characters that fail to load
            continue

    return CharacterListResponse(
        characters=characters,
        total=len(characters),
    )


@router.get("/{character_id}", response_model=CharacterDetailResponse)
async def get_character(character_id: str):
    """
    Get detailed information about a character.

    Use the character_id (filename without extension) from the list endpoint.
    """
    try:
        char = load_character(character_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return CharacterDetailResponse(
        name=char.name,
        role=char.role,
        backstory=char.backstory,
        traits=char.personality.traits,
        speech_patterns=char.personality.speech_patterns,
        knowledge_domains=char.knowledge_domains.model_dump() if char.knowledge_domains else None,
        behavioral_boundaries=char.behavioral_boundaries.model_dump() if char.behavioral_boundaries else None,
        has_first_message=bool(char.first_message),
        example_dialogues_count=len(char.example_dialogues),
    )


@router.get("/{character_id}/system-prompt")
async def get_character_system_prompt(character_id: str):
    """
    Get the generated system prompt for a character.

    Useful for debugging and understanding how the character is configured.
    """
    try:
        char = load_character(character_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "character_id": character_id,
        "name": char.name,
        "system_prompt": char.build_system_prompt(),
        "few_shot_examples": char.get_few_shot_examples(),
    }


@router.get("/{character_id}/first-message")
async def get_character_first_message(character_id: str):
    """
    Get the character's opening message.

    This is what the NPC says when a conversation starts.
    """
    try:
        char = load_character(character_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "character_id": character_id,
        "name": char.name,
        "first_message": char.first_message,
        "has_first_message": bool(char.first_message),
    }
