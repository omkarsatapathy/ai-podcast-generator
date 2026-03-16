"""Pydantic models for Phase 2: Character Designer."""

from pydantic import BaseModel, Field
from typing import List, Literal


class CharacterPersona(BaseModel):
    """A single speaker persona for the podcast."""
    name: str = Field(..., description="Full character name, e.g. 'Dr. Sarah Chen'")
    role: Literal["host", "expert", "skeptic"] = Field(..., description="Speaker role")
    expertise_area: str = Field(..., description="What this character knows about, tailored to topic")
    speaking_style: str = Field(..., description="How they talk — analogies, technical depth, etc.")
    vocabulary_level: Literal["casual", "moderate", "technical"]
    filler_patterns: List[str] = Field(..., min_length=2, description="Speech fillers like 'you know', 'like'")
    reaction_patterns: List[str] = Field(..., min_length=2, description="Reactions like 'Oh interesting!'")
    disagreement_style: str = Field(..., description="How they push back or challenge")
    laugh_frequency: Literal["rare", "moderate", "frequent"]
    catchphrases: List[str] = Field(..., min_length=2, description="Signature phrases")
    emotional_range: str = Field(..., description="What excites, concerns, or amuses them")
    tts_voice_id: str = Field(..., description="Voice name from the available TTS voices")
    gender: Literal["male", "female"] = Field(..., description="Gender matching the chosen voice")


class CharacterRoster(BaseModel):
    """LLM response: all speaker personas for the podcast."""
    characters: List[CharacterPersona] = Field(..., min_length=2, max_length=3)
