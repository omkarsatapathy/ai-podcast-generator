"""Pydantic models for Phase 5: Audio Post-Processing."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class OverlapOperation(BaseModel):
    """Single timing directive resolved to a concrete audio operation."""

    model_config = ConfigDict(extra="ignore")

    op_type: Literal["interrupt", "backchannel", "laugh"]
    chapter_number: int
    target_utterance_id: str
    trigger_utterance_id: str
    speaker: str
    duration_ms: Optional[int] = None
    insertion_offset_ms: Optional[int] = None


class UtteranceTimestamp(BaseModel):
    """Entry in the utterance timestamp map."""

    utterance_id: str
    speaker: str
    start_ms: int
    end_ms: int
    chapter_number: int


class ChapterMixReport(BaseModel):
    """Per-chapter Overlap Engine output report."""

    chapter_number: int
    input_clip_count: int = 0
    interrupts_applied: int = 0
    backchannels_applied: int = 0
    laughs_applied: int = 0
    total_duration_ms: int = 0
    output_path: str = ""


class MasteringReport(BaseModel):
    """Per-chapter Post-Processor output report."""

    chapter_number: int
    steps_applied: List[str] = Field(default_factory=list)
    input_duration_ms: int = 0
    output_duration_ms: int = 0
    output_path: str = ""


class ColdOpenReport(BaseModel):
    """Cold Open Generator output report."""

    selected_chapter_number: int = 0
    start_utterance_id: str = ""
    end_utterance_id: str = ""
    duration_ms: int = 0
    framing_strategy: Literal["tts", "pre_recorded", "none"] = "none"
    llm_used: bool = False
    fallback_used: bool = False


class ColdOpenCandidate(BaseModel):
    """LLM-selected cold open candidate."""

    model_config = ConfigDict(extra="ignore")

    candidate_rank: int
    chapter_number: int
    start_utterance_id: str
    end_utterance_id: str
    reason: str = ""


class Phase5Output(BaseModel):
    """Final Phase 5 output contract."""

    phase5_contract_version: str = "1.0"
    episode_id: str = ""
    final_podcast_path: str = ""
    total_duration_seconds: float = 0.0
    file_size_bytes: int = 0
    chapter_count: int = 0
    degraded_chapters: List[int] = Field(default_factory=list)
    cold_open_included: bool = False
    loudness_target_lufs: float = -16.0
    mp3_bitrate_kbps: int = 128
    chapter_markers: Dict[str, Any] = Field(default_factory=dict)
    ready: bool = False
