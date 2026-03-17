"""Phase 5: Chapter Stitcher — final episode assembly, MP3 export, ID3 tagging."""

from __future__ import annotations

import datetime
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydub import AudioSegment

from config.settings import settings
from src.tools.audio_tools import convert_to_pipeline_format, export_wav_atomic
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_optional_asset(path: str | Path) -> AudioSegment | None:
    """Load and normalize an optional audio asset. Returns None if missing."""

    path = Path(path)
    if not path.exists():
        logger.debug("Optional asset not found: %s", path)
        return None
    try:
        seg = AudioSegment.from_wav(str(path))
        return convert_to_pipeline_format(seg)
    except Exception as exc:
        logger.warning("Failed to load asset %s: %s", path, exc)
        return None


def _build_intro_crossfade(
    cold_open: AudioSegment | None,
    intro_music: AudioSegment | None,
) -> AudioSegment:
    """Build the intro section with cold open + music crossfade."""

    crossfade_ms = settings.PHASE5_COLD_OPEN_INTRO_CROSSFADE_MS
    music_dur = settings.PHASE5_INTRO_MUSIC_DURATION_MS

    if intro_music and len(intro_music) > music_dur:
        intro_music = intro_music[:music_dur]

    if cold_open and intro_music and len(cold_open) > crossfade_ms:
        # Crossfade: intro music fades in under cold open tail
        cold_open_tail = cold_open[-crossfade_ms:]
        intro_fade = (intro_music[:crossfade_ms] + (-8))
        blended_tail = cold_open_tail.overlay(intro_fade)
        intro_section = cold_open[:-crossfade_ms] + blended_tail
        if len(intro_music) > crossfade_ms:
            intro_section += intro_music[crossfade_ms:]
        return intro_section
    elif cold_open and intro_music:
        return cold_open + intro_music
    elif cold_open:
        return cold_open
    elif intro_music:
        return intro_music
    else:
        return AudioSegment.silent(duration=1000)


def _run_final_loudness(input_path: str, output_path: str) -> None:
    """Apply final loudness normalization to assembled episode."""

    target_i = settings.PHASE5_LOUDNESS_TARGET_LUFS
    target_tp = settings.PHASE5_LOUDNESS_TRUE_PEAK_DB

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", input_path,
                "-af", f"loudnorm=I={target_i}:TP={target_tp}:LRA=11",
                "-ar", str(settings.PHASE5_TARGET_SAMPLE_RATE),
                output_path,
            ],
            capture_output=True, text=True, timeout=300, check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.warning("Final loudness pass failed, using source: %s", exc)
        # Copy source as-is
        if input_path != output_path:
            import shutil
            shutil.copy2(input_path, output_path)


def _encode_mp3(input_wav: str, output_mp3: str) -> None:
    """Encode WAV to MP3 via ffmpeg."""

    bitrate = settings.PHASE5_MP3_BITRATE_KBPS
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", input_wav,
            "-codec:a", "libmp3lame",
            "-b:a", f"{bitrate}k",
            "-q:a", "2",
            output_mp3,
        ],
        capture_output=True, text=True, timeout=300, check=True,
    )


def _embed_id3_tags(
    mp3_path: str,
    topic: str,
    episode_id: str,
    chapter_markers: Dict[str, Any],
) -> None:
    """Write ID3 metadata tags to the MP3 file."""

    try:
        from mutagen.id3 import ID3, TIT2, TPE1, TALB, TDRC, TCON, COMM, CHAP, CTOC, TIT2 as ChapTitle
        from mutagen.mp3 import MP3

        audio = MP3(mp3_path)
        if audio.tags is None:
            audio.add_tags()

        tags = audio.tags
        tags.add(TIT2(encoding=3, text=f"{topic} — Episode {episode_id}"))
        tags.add(TPE1(encoding=3, text="AI Podcast Generator"))
        tags.add(TALB(encoding=3, text="AI Podcast Generator"))
        tags.add(TDRC(encoding=3, text=str(datetime.date.today().year)))
        tags.add(TCON(encoding=3, text="Podcast"))
        tags.add(COMM(
            encoding=3, lang="eng", desc="",
            text=f"An AI-generated podcast episode on the topic: {topic}.",
        ))

        # Chapter markers
        if chapter_markers:
            chapter_ids = []
            total_ms = int(audio.info.length * 1000)
            sorted_markers = sorted(chapter_markers.items(), key=lambda x: x[1].get("start_ms", 0))

            for label, info in sorted_markers:
                chap_id = f"chap_{label}".replace(" ", "_")[:20]
                chapter_ids.append(chap_id)
                start = info.get("start_ms", 0)
                end = info.get("end_ms", total_ms)
                tags.add(CHAP(
                    element_id=chap_id,
                    start_time=start,
                    end_time=end,
                    sub_frames=[TIT2(encoding=3, text=label)],
                ))

            if chapter_ids:
                tags.add(CTOC(
                    element_id="toc",
                    flags=3,
                    child_element_ids=chapter_ids,
                    sub_frames=[TIT2(encoding=3, text="Table of Contents")],
                ))

        audio.save()
        logger.info("ID3 tags written to %s", mp3_path)
    except ImportError:
        logger.warning("mutagen not installed, skipping ID3 tags")
    except Exception as exc:
        logger.warning("ID3 tagging failed: %s", exc)


def run_chapter_stitcher(
    cold_open_path: str | None,
    cold_open_failed: bool,
    chapter_mastered_paths: Dict[int, str],
    chapter_dialogues: List[Dict[str, Any]],
    topic: str,
    episode_id: str,
    output_dir: str,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Assemble all components into the final podcast MP3."""

    chapter_markers: Dict[str, Any] = {}
    report: Dict[str, Any] = {"chapters_assembled": 0, "cold_open_included": False}
    base = Path(settings.BASE_DIR)

    # Load optional assets
    cold_open = None
    if not cold_open_failed and cold_open_path and Path(cold_open_path).exists():
        try:
            cold_open = AudioSegment.from_wav(cold_open_path)
            cold_open = convert_to_pipeline_format(cold_open)
            report["cold_open_included"] = True
        except Exception as exc:
            logger.warning("Failed to load cold open: %s", exc)

    intro_music = _load_optional_asset(base / "data/audio/assets/music/intro_music.wav")
    outro_music = _load_optional_asset(base / "data/audio/assets/music/outro_music.wav")
    chapter_sting = _load_optional_asset(base / "data/audio/assets/transitions/chapter_sting.wav")
    if chapter_sting is None:
        chapter_sting = AudioSegment.silent(duration=500)

    # Build intro section
    intro_section = _build_intro_crossfade(cold_open, intro_music)

    # Start assembling episode
    episode = intro_section
    chapter_markers["Intro"] = {"start_ms": 0, "end_ms": len(episode)}
    episode += AudioSegment.silent(duration=500)

    # Host intro placeholder (silence if no pre-rendered clip)
    host_intro_path = Path(output_dir) / "host_intro.wav"
    if host_intro_path.exists():
        try:
            host_intro = AudioSegment.from_wav(str(host_intro_path))
            host_intro = convert_to_pipeline_format(host_intro)
        except Exception:
            host_intro = AudioSegment.silent(duration=2000)
    else:
        host_intro = AudioSegment.silent(duration=2000)

    intro_start = len(episode)
    episode += host_intro
    chapter_markers["Introduction"] = {"start_ms": intro_start, "end_ms": len(episode)}
    episode += AudioSegment.silent(duration=300)

    # Append each chapter
    sorted_chapters = sorted(chapter_mastered_paths.keys())
    for i, ch_num in enumerate(sorted_chapters):
        ch_path = chapter_mastered_paths[ch_num]
        if not Path(ch_path).exists():
            logger.warning("Chapter %d mastered file missing, skipping", ch_num)
            continue

        try:
            ch_audio = AudioSegment.from_wav(ch_path)
            ch_audio = convert_to_pipeline_format(ch_audio)
        except Exception as exc:
            logger.warning("Failed to load chapter %d: %s", ch_num, exc)
            continue

        # Get chapter title
        ch_title = f"Chapter {ch_num}"
        ch_data = next(
            (c for c in chapter_dialogues if c.get("chapter_number") == ch_num), None
        )
        if ch_data and ch_data.get("title"):
            ch_title = ch_data["title"]

        ch_start = len(episode)
        episode += ch_audio
        ch_end = len(episode)
        chapter_markers[ch_title] = {"start_ms": ch_start, "end_ms": ch_end}
        report["chapters_assembled"] = report.get("chapters_assembled", 0) + 1

        # Transition sting between chapters (not after last)
        if i < len(sorted_chapters) - 1:
            episode += chapter_sting + AudioSegment.silent(duration=200)

    # Host outro placeholder
    host_outro_path = Path(output_dir) / "host_outro.wav"
    if host_outro_path.exists():
        try:
            host_outro = AudioSegment.from_wav(str(host_outro_path))
            host_outro = convert_to_pipeline_format(host_outro)
        except Exception:
            host_outro = AudioSegment.silent(duration=2000)
    else:
        host_outro = AudioSegment.silent(duration=2000)

    outro_start = len(episode)
    episode += AudioSegment.silent(duration=500) + host_outro
    chapter_markers["Outro"] = {"start_ms": outro_start, "end_ms": len(episode)}

    # Outro music crossfade
    if outro_music:
        outro_tail_ms = min(3000, len(host_outro))
        if len(episode) > outro_tail_ms:
            fade_music = (outro_music[:outro_tail_ms] + (-6))
            tail = episode[-outro_tail_ms:]
            blended = tail.overlay(fade_music)
            episode = episode[:-outro_tail_ms] + blended
        if len(outro_music) > outro_tail_ms:
            episode += outro_music[outro_tail_ms:]

    # Final loudness normalization + MP3 encoding
    final_dir = Path(output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    temp_assembled = str(final_dir / "episode_assembled_temp.wav")
    temp_normalized = str(final_dir / "episode_final_normalized.wav")
    final_mp3 = str(final_dir / "podcast_episode_final.mp3")

    export_wav_atomic(episode, temp_assembled)
    _run_final_loudness(temp_assembled, temp_normalized)

    try:
        _encode_mp3(temp_normalized, final_mp3)
    except subprocess.CalledProcessError as exc:
        logger.error("MP3 encoding failed: %s", exc)
        return "", chapter_markers, report

    # ID3 tags
    _embed_id3_tags(final_mp3, topic, episode_id, chapter_markers)

    # Validate
    if Path(final_mp3).exists() and Path(final_mp3).stat().st_size > 0:
        report["file_size_bytes"] = Path(final_mp3).stat().st_size
        report["total_duration_ms"] = len(episode)
        report["bitrate_kbps"] = settings.PHASE5_MP3_BITRATE_KBPS
        logger.info(
            "Episode assembled: %d ms, %d bytes, %s",
            len(episode), report["file_size_bytes"], final_mp3,
        )
    else:
        logger.error("Final MP3 missing or empty")
        return "", chapter_markers, report

    # Cleanup temp files
    for tmp in [temp_assembled, temp_normalized]:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass

    return final_mp3, chapter_markers, report
