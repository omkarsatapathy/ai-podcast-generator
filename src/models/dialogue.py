"""Pydantic models for Phase 3: Dialogue Generation."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal

_VALID_EMOTIONS = {"curious", "excited", "skeptical", "thoughtful", "amused", "neutral"}
_VALID_INTENTS = {"question", "answer", "reaction", "challenge", "summary", "transition"}


class RawUtterance(BaseModel):
    """Single utterance from LLM dialogue generation."""
    speaker: str
    text: str
    intent: Literal["question", "answer", "reaction", "challenge", "summary", "transition"]
    emotion: Literal["curious", "excited", "skeptical", "thoughtful", "amused", "neutral"]
    grounding_chunk_ids: List[int] = []

    @field_validator("emotion", mode="before")
    @classmethod
    def coerce_emotion(cls, v: str) -> str:
        if v in _VALID_EMOTIONS:
            return v
        return "neutral"

    @field_validator("intent", mode="before")
    @classmethod
    def coerce_intent(cls, v: str) -> str:
        if v in _VALID_INTENTS:
            return v
        return "answer"


class BeatDialogue(BaseModel):
    """LLM structured output for a beat's dialogue."""
    utterances: List[RawUtterance]


class FactCheckItemResult(BaseModel):
    """Verification result for a single claim."""
    claim_index: int
    verdict: Literal["supported", "partially_supported", "unsupported"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    source_quote: str = ""
    correction: Optional[str] = None


class BatchFactCheckResult(BaseModel):
    """LLM output for batch fact checking."""
    results: List[FactCheckItemResult]


class QAReviewResult(BaseModel):
    """LLM output for chapter quality review."""
    overall_pass: bool
    issues_found: List[dict] = []
    strengths: List[str] = []
    listener_experience_score: float = Field(ge=1.0, le=10.0)
    reasoning: str = ""
