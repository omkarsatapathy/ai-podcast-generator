"""Phase 5: Audio Post-Processor — mastering chain via ffmpeg."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

from pydub import AudioSegment

from config.settings import settings
from src.models.phase5 import MasteringReport
from src.tools.audio_tools import (
    convert_to_pipeline_format,
    export_wav_atomic,
    get_file_duration_ms,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _run_ffmpeg(args: list[str], description: str) -> str:
    """Run an ffmpeg command, return stderr for diagnostics.

    Automatically injects -ar 44100 before the output file to prevent
    ffmpeg filters (especially loudnorm) from resampling to 192 kHz.
    """

    cmd = ["ffmpeg", "-y", "-loglevel", "error"] + args
    # Force output sample rate to pipeline standard (44100 Hz)
    # Insert -ar 44100 before the last argument (the output path)
    if len(cmd) > 1 and not cmd[-1].startswith("-"):
        cmd.insert(-1, "-ar")
        cmd.insert(-1, str(settings.PHASE5_TARGET_SAMPLE_RATE))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        logger.error("ffmpeg %s failed: %s", description, result.stderr)
        raise RuntimeError(f"ffmpeg {description} failed: {result.stderr}")
    return result.stderr


def _verify_ffmpeg() -> None:
    """Ensure ffmpeg is available on PATH."""

    try:
        subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, timeout=10, check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "ffmpeg not found. Install via: brew install ffmpeg (macOS) "
            "or apt-get install ffmpeg (Linux)"
        ) from exc


def _run_noise_gate(input_path: str, output_path: str) -> None:
    """Strip silence artifacts below threshold."""

    threshold = settings.PHASE5_NOISE_GATE_THRESHOLD_DB
    duration = settings.PHASE5_NOISE_GATE_SILENCE_DURATION
    _run_ffmpeg(
        [
            "-i", input_path,
            "-af", f"silenceremove=stop_periods=-1:stop_duration={duration}:stop_threshold={threshold}dB",
            output_path,
        ],
        "noise_gate",
    )


def _run_equalisation(input_path: str, output_path: str) -> None:
    """Apply EQ: boost presence, cut rumble."""

    pf = settings.PHASE5_EQ_PRESENCE_FREQ
    pg = settings.PHASE5_EQ_PRESENCE_GAIN
    rf = settings.PHASE5_EQ_RUMBLE_FREQ
    rg = settings.PHASE5_EQ_RUMBLE_GAIN
    _run_ffmpeg(
        [
            "-i", input_path,
            "-af", f"equalizer=f={pf}:t=h:w=2000:g={pg},equalizer=f={rf}:t=h:w=100:g={rg}",
            output_path,
        ],
        "equalisation",
    )


def _run_compression(input_path: str, output_path: str) -> None:
    """Apply dynamic compression."""

    t = settings.PHASE5_COMP_THRESHOLD_DB
    r = settings.PHASE5_COMP_RATIO
    a = settings.PHASE5_COMP_ATTACK_MS
    rel = settings.PHASE5_COMP_RELEASE_MS
    m = settings.PHASE5_COMP_MAKEUP_GAIN_DB
    _run_ffmpeg(
        [
            "-i", input_path,
            "-af", f"acompressor=threshold={t}dB:ratio={r}:attack={a}:release={rel}:makeup={m}",
            output_path,
        ],
        "compression",
    )


def _run_loudness_normalisation(input_path: str, output_path: str) -> None:
    """Two-pass loudnorm to -16 LUFS."""

    target_i = settings.PHASE5_LOUDNESS_TARGET_LUFS
    target_tp = settings.PHASE5_LOUDNESS_TRUE_PEAK_DB

    # Pass 1: analysis
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", input_path,
            "-af", f"loudnorm=I={target_i}:TP={target_tp}:LRA=11:print_format=json",
            "-f", "null", "-",
        ],
        capture_output=True, text=True, timeout=120,
    )

    # Parse loudnorm JSON from stderr
    measured = _parse_loudnorm_json(result.stderr)
    if not measured:
        # Fallback: single-pass
        logger.warning("loudnorm pass 1 parse failed, using single-pass")
        _run_ffmpeg(
            ["-i", input_path, "-af", f"loudnorm=I={target_i}:TP={target_tp}:LRA=11", output_path],
            "loudnorm_single",
        )
        return

    # Guard against -inf or invalid measured values (happens with very quiet audio)
    input_i = str(measured.get("input_i", ""))
    if input_i in ("-inf", "inf", "", "nan"):
        logger.warning("loudnorm measured -inf LUFS, using single-pass")
        _run_ffmpeg(
            ["-i", input_path, "-af", f"loudnorm=I={target_i}:TP={target_tp}:LRA=11", output_path],
            "loudnorm_single",
        )
        return

    # Pass 2: apply measured values
    _run_ffmpeg(
        [
            "-i", input_path,
            "-af", (
                f"loudnorm=I={target_i}:TP={target_tp}:LRA=11"
                f":measured_I={measured['input_i']}"
                f":measured_TP={measured['input_tp']}"
                f":measured_LRA={measured['input_lra']}"
                f":measured_thresh={measured['input_thresh']}"
                f":offset={measured['target_offset']}"
                ":linear=true"
            ),
            output_path,
        ],
        "loudnorm_pass2",
    )


def _parse_loudnorm_json(stderr: str) -> dict | None:
    """Extract the loudnorm JSON block from ffmpeg stderr."""

    # ffmpeg outputs the JSON at the end of stderr
    try:
        # Find the last JSON object in stderr
        brace_start = stderr.rfind("{")
        brace_end = stderr.rfind("}") + 1
        if brace_start < 0 or brace_end <= brace_start:
            return None
        return json.loads(stderr[brace_start:brace_end])
    except (json.JSONDecodeError, ValueError):
        return None


def _apply_room_tone(input_path: str, output_path: str) -> bool:
    """Overlay ambient room tone. Returns True if applied."""

    if not settings.PHASE5_ENABLE_ROOM_TONE:
        return False

    tone_path = Path(settings.BASE_DIR) / "data/audio/assets/room_tone/room_tone_default.wav"
    if not tone_path.exists():
        logger.warning("Room tone asset not found at %s, skipping", tone_path)
        return False

    try:
        chapter = AudioSegment.from_wav(input_path)
        room_tone = AudioSegment.from_wav(str(tone_path))
        room_tone = convert_to_pipeline_format(room_tone)

        # Loop to cover chapter duration
        repeats = (len(chapter) // len(room_tone)) + 1
        room_tone = (room_tone * repeats)[:len(chapter)]
        room_tone = room_tone + settings.PHASE5_ROOM_TONE_LEVEL_DB

        mixed = chapter.overlay(room_tone)
        export_wav_atomic(mixed, output_path)
        return True
    except Exception as exc:
        logger.warning("Room tone application failed: %s", exc)
        return False


def run_mastering_chain(
    input_path: str,
    output_path: str,
    chapter_number: int,
) -> MasteringReport:
    """Run the full mastering chain on a chapter WAV file."""

    _verify_ffmpeg()

    report = MasteringReport(chapter_number=chapter_number)
    report.input_duration_ms = get_file_duration_ms(input_path)

    # Create temp directory for intermediate files
    temp_dir = tempfile.mkdtemp(prefix="phase5_master_")
    step_files = []

    try:
        # Step 1: Noise gate
        step1 = os.path.join(temp_dir, "step1_noisegate.wav")
        _run_noise_gate(input_path, step1)
        step_files.append(step1)
        report.steps_applied.append("noise_gate")

        # Verify duration didn't drop too much
        step1_dur = get_file_duration_ms(step1)
        if report.input_duration_ms > 0 and step1_dur < report.input_duration_ms * 0.95:
            logger.warning(
                "Chapter %d noise gate removed >5%% duration (%d→%d ms)",
                chapter_number, report.input_duration_ms, step1_dur,
            )

        # Step 2: EQ
        step2 = os.path.join(temp_dir, "step2_eq.wav")
        _run_equalisation(step1, step2)
        step_files.append(step2)
        report.steps_applied.append("eq")

        # Step 3: Compression
        step3 = os.path.join(temp_dir, "step3_compressed.wav")
        _run_compression(step2, step3)
        step_files.append(step3)
        report.steps_applied.append("compression")

        # Step 4: Loudness normalization
        step4 = os.path.join(temp_dir, "step4_normalized.wav")
        if report.input_duration_ms > 5000:
            _run_loudness_normalisation(step3, step4)
            report.steps_applied.append("loudnorm")
        else:
            # Skip loudnorm for very short chapters
            logger.warning("Chapter %d too short for loudnorm, skipping", chapter_number)
            step4 = step3

        # Step 5: Room tone (optional)
        step5 = os.path.join(temp_dir, "step5_ambience.wav")
        if _apply_room_tone(step4, step5):
            report.steps_applied.append("room_tone")
            final_step = step5
            step_files.append(step5)
        else:
            final_step = step4

        # Export to final path
        export_wav_atomic(AudioSegment.from_wav(final_step), output_path)
        report.output_path = output_path
        report.output_duration_ms = get_file_duration_ms(output_path)

        logger.info(
            "Chapter %d mastered: %d ms → %d ms, steps=%s",
            chapter_number, report.input_duration_ms, report.output_duration_ms,
            report.steps_applied,
        )
    finally:
        # Clean up temp files
        for f in step_files:
            try:
                if os.path.exists(f):
                    os.unlink(f)
            except OSError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

    return report
