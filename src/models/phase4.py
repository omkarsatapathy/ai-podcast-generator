"""Pydantic models for Phase 4: Voice Synthesis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Phase4AudioMetadata(BaseModel):
    """Timing metadata extracted from Phase 3 naturalness markers."""

    interrupt_duration: Optional[str] = None
    backchannel_speaker: Optional[str] = None


class Phase4Utterance(BaseModel):
    """Minimal utterance contract Phase 4 expects from Phase 3."""

    model_config = ConfigDict(extra="ignore")

    utterance_id: str
    speaker: str
    role: str = "unknown"
    beat: int
    text_clean: str
    text_with_naturalness: str = ""
    text_ssml: str
    estimated_duration_seconds: float = Field(ge=0.0)
    tts_voice_id: str = ""
    audio_metadata: Phase4AudioMetadata = Field(default_factory=Phase4AudioMetadata)


class Phase4ChapterScript(BaseModel):
    """Chapter-level Phase 3 output consumed by Phase 4."""

    model_config = ConfigDict(extra="ignore")

    chapter_number: int
    utterances: List[Phase4Utterance] = Field(default_factory=list)


class SpeakerVoiceAssignment(BaseModel):
    """Resolved voice policy for a single speaker."""

    speaker: str
    role: str
    provider: str
    model: str
    voice_id: str
    source: str
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None
    fallback_voice_id: Optional[str] = None


class TTSJob(BaseModel):
    """Single utterance-level synthesis job."""

    model_config = ConfigDict(extra="allow")

    job_id: str
    episode_id: str
    chapter_number: int
    utterance_id: str
    lineage_utterance_id: str
    order_index: int
    segment_index: int = 0
    segment_count: int = 1
    speaker: str
    role: str
    provider: str
    model: str
    voice_id: str
    text_clean: str
    text_with_naturalness: str = ""
    text_ssml: str = ""
    estimated_duration_seconds: float = Field(ge=0.0)
    audio_metadata: Phase4AudioMetadata = Field(default_factory=Phase4AudioMetadata)
    output_path: str
    retry_count: int = 0
    repair_mode: bool = False
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None
    fallback_voice_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class AudioClip(BaseModel):
    """Synthesis output for a single job."""

    model_config = ConfigDict(extra="allow")

    job_id: str
    utterance_id: str
    lineage_utterance_id: str
    chapter_number: int
    order_index: int
    speaker: str
    role: str
    provider: str
    model: str
    voice_id: str
    path: str
    duration_seconds: float = Field(ge=0.0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    size_bytes: int = Field(ge=0)
    audio_metadata: Phase4AudioMetadata = Field(default_factory=Phase4AudioMetadata)
    qc_passed: bool = True


class ChapterAudioManifest(BaseModel):
    """Ordered chapter audio handoff for Phase 5."""

    manifest_version: str
    chapter_number: int
    utterance_count: int = Field(ge=0)
    complete: bool
    clip_checksum: str
    clips: List[Dict[str, Any]] = Field(default_factory=list)
    timing_directives: List[Dict[str, Any]] = Field(default_factory=list)


class Phase4Output(BaseModel):
    """Official Phase 4 handoff contract."""

    phase4_contract_version: str
    episode_id: str
    provider: str
    model: str
    ready_for_phase5: bool
    audio_files: List[Dict[str, Any]] = Field(default_factory=list)
    chapter_manifests: List[Dict[str, Any]] = Field(default_factory=list)
    voice_metadata: Dict[str, Any] = Field(default_factory=dict)
    timing_metadata: Dict[str, Any] = Field(default_factory=dict)
    quality_reports: Dict[str, Any] = Field(default_factory=dict)
    failed_jobs: List[Dict[str, Any]] = Field(default_factory=list)
