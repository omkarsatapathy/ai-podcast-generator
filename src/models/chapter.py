"""Pydantic models for Phase 2: Chapter Planner."""

from pydantic import BaseModel, Field
from typing import List, Literal


# --- Step 1: Chunk Analysis (LLM structured output) ---

class ChunkAnalysis(BaseModel):
    """Metadata extracted from a single chunk by LLM."""
    chunk_id: str
    topic: str = Field(..., description="Main topic in max 5 words")
    subtopics: List[str] = Field(..., description="3-5 subtopics, each max 8 words")
    summary: str = Field(..., description="Summary in max 15 words")
    tone: Literal["factual", "opinion", "debate", "technical", "narrative"]


class BatchChunkAnalysis(BaseModel):
    """LLM response for a batch of chunks."""
    analyses: List[ChunkAnalysis]


# --- Step 3: Narrative Sequence (LLM structured output) ---

class ChapterSequenceItem(BaseModel):
    """A single chapter in the narrative sequence."""
    chapter_number: int
    act: Literal["setup", "explore", "resolve"]
    energy_level: Literal["high", "medium", "low"]
    cluster_ids: List[int] = Field(..., description="Which cluster indices belong to this chapter")
    estimated_duration_minutes: float = Field(..., ge=2.0, le=5.0)


class NarrativeSequence(BaseModel):
    """LLM response for 3-act narrative ordering."""
    chapters: List[ChapterSequenceItem]


# --- Step 4: Chapter Outline (LLM structured output) ---

class ChapterOutlineGenerated(BaseModel):
    """LLM-generated creative content for a single chapter."""
    chapter_number: int
    title: str = Field(..., description="Engaging title, 5-8 words")
    key_points: List[str] = Field(..., description="3-5 key discussion points")
    transition_hook: str = Field(..., description="One sentence teasing next chapter")


class BatchChapterOutlines(BaseModel):
    """LLM response for all chapter outlines."""
    outlines: List[ChapterOutlineGenerated]


# --- Final Output ---

class ChapterOutline(BaseModel):
    """Complete chapter outline combining LLM output + deterministic data."""
    chapter_number: int
    title: str
    act: Literal["setup", "explore", "resolve"]
    energy_level: Literal["high", "medium", "low"]
    key_points: List[str]
    source_chunk_ids: List[str]
    transition_hook: str
    estimated_duration_minutes: float
