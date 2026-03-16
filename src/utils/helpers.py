"""Utility helper functions."""
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


def generate_job_id() -> str:
    """Generate a unique job ID."""
    return f"podcast_{uuid.uuid4().hex[:12]}"


def get_timestamp_iso() -> str:
    """Get current timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


def get_output_path(job_id: str, output_dir: Path, extension: str = "mp3") -> Path:
    """Generate output file path for a job."""
    return output_dir / f"{job_id}.{extension}"


def validate_audio_file(file_path: Path) -> bool:
    """Validate that an audio file exists and has content."""
    return file_path.exists() and file_path.stat().st_size > 0
