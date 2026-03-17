"""Audio helpers for Phase 4 synthesis and QC."""

from __future__ import annotations

import os
import tempfile
import wave
from array import array
from pathlib import Path
from typing import Any, Dict


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write bytes atomically to avoid partial file corruption."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def write_audio_bytes_atomic(
    path: str | Path,
    audio_bytes: bytes,
    sample_rate: int,
    channels: int = 1,
    sample_width: int = 2,
) -> None:
    """Write provider output to a WAV file, wrapping raw PCM when needed."""

    path = Path(path)
    if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        _write_bytes_atomic(path, audio_bytes)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as raw_handle:
            with wave.open(raw_handle, "wb") as wav_handle:
                wav_handle.setnchannels(channels)
                wav_handle.setsampwidth(sample_width)
                wav_handle.setframerate(sample_rate)
                wav_handle.writeframes(audio_bytes)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def inspect_wav_file(path: str | Path) -> Dict[str, Any]:
    """Read basic technical metadata from a WAV file."""

    path = Path(path)
    with wave.open(str(path), "rb") as handle:
        frame_count = handle.getnframes()
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        duration_seconds = frame_count / float(sample_rate) if sample_rate else 0.0

    return {
        "path": str(path),
        "frame_count": frame_count,
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": sample_width,
        "duration_seconds": duration_seconds,
        "size_bytes": path.stat().st_size,
    }


def validate_wav_file(
    path: str | Path,
    min_duration_seconds: float,
    silence_peak_threshold: int,
    clipping_sample_threshold: int,
    clipping_ratio_threshold: float,
    expected_sample_rate: int | None = None,
    expected_channels: int | None = None,
) -> Dict[str, Any]:
    """Validate that a WAV file is decodable and usable for Phase 5."""

    metadata = inspect_wav_file(path)
    report: Dict[str, Any] = {
        **metadata,
        "passed": True,
        "reasons": [],
    }

    if metadata["duration_seconds"] < min_duration_seconds:
        report["passed"] = False
        report["reasons"].append("duration_below_minimum")

    if expected_sample_rate and metadata["sample_rate"] != expected_sample_rate:
        report["passed"] = False
        report["reasons"].append("sample_rate_mismatch")

    if expected_channels and metadata["channels"] != expected_channels:
        report["passed"] = False
        report["reasons"].append("channel_mismatch")

    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        sample_width = handle.getsampwidth()

    if not frames:
        report["passed"] = False
        report["reasons"].append("empty_audio")
        report["peak_amplitude"] = 0
        report["clipping_ratio"] = 0.0
        return report

    if sample_width != 2:
        # Phase 4 currently only writes 16-bit PCM WAV output.
        report["passed"] = False
        report["reasons"].append("unsupported_sample_width")
        report["peak_amplitude"] = 0
        report["clipping_ratio"] = 0.0
        return report

    samples = array("h")
    samples.frombytes(frames)
    peak_amplitude = max((abs(sample) for sample in samples), default=0)
    clipping_ratio = (
        sum(1 for sample in samples if abs(sample) >= clipping_sample_threshold)
        / len(samples)
    ) if samples else 0.0

    report["peak_amplitude"] = peak_amplitude
    report["clipping_ratio"] = clipping_ratio

    if peak_amplitude < silence_peak_threshold:
        report["passed"] = False
        report["reasons"].append("audio_too_quiet")

    if clipping_ratio > clipping_ratio_threshold:
        report["passed"] = False
        report["reasons"].append("clipping_detected")

    return report
