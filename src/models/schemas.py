"""Pydantic schemas for API request/response validation."""
from pydantic import BaseModel, Field, validator
from typing import Optional
from enum import Enum


class PodcastStatus(str, Enum):
    """Podcast generation status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PodcastRequest(BaseModel):
    """Request model for podcast generation."""

    topic: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The topic for the podcast",
        examples=["The impact of AI on healthcare"]
    )

    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Optional detailed description to guide content generation",
        examples=["Focus on recent breakthroughs and ethical considerations"]
    )

    num_speakers: Optional[int] = Field(
        2,
        ge=2,
        le=3,
        description="Number of speakers (2-3)"
    )

    @validator('topic')
    def topic_not_empty(cls, v):
        """Validate topic is not just whitespace."""
        if not v or not v.strip():
            raise ValueError('Topic cannot be empty or whitespace')
        return v.strip()


class PodcastResponse(BaseModel):
    """Response model for podcast generation."""

    job_id: str = Field(..., description="Unique identifier for the podcast generation job")
    status: PodcastStatus = Field(..., description="Current status of the job")
    message: str = Field(..., description="Human-readable status message")
    audio_url: Optional[str] = Field(None, description="URL to download the generated podcast MP3")
    duration_seconds: Optional[float] = Field(None, description="Duration of the podcast in seconds")
    created_at: str = Field(..., description="ISO timestamp of job creation")


class JobStatusResponse(BaseModel):
    """Response model for job status check."""

    job_id: str = Field(..., description="Unique identifier for the job")
    status: PodcastStatus = Field(..., description="Current status of the job")
    message: str = Field(..., description="Status message")
    progress_percent: Optional[int] = Field(None, ge=0, le=100, description="Progress percentage")
    audio_url: Optional[str] = Field(None, description="URL to download the generated podcast")
    error: Optional[str] = Field(None, description="Error message if failed")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")
