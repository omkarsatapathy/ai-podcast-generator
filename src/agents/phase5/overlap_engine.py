"""Phase 5: Audio Overlap Engine — mixes sequential clips into natural conversation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from pydub import AudioSegment

from config.settings import settings
from src.models.phase5 import ChapterMixReport
from src.tools.audio_tools import convert_to_pipeline_format, export_wav_atomic
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_and_normalize_clips(
    clips: List[Dict[str, Any]],
) -> Dict[str, AudioSegment]:
    """Load WAV clips and normalize to pipeline format (44100 Hz, stereo, 16-bit)."""

    loaded: Dict[str, AudioSegment] = {}
    for clip in sorted(clips, key=lambda c: c.get("order_index", 0)):
        uid = clip["utterance_id"]
        path = clip["path"]
        try:
            seg = AudioSegment.from_wav(path)
            seg = convert_to_pipeline_format(
                seg,
                settings.PHASE5_TARGET_SAMPLE_RATE,
                settings.PHASE5_TARGET_CHANNELS,
                settings.PHASE5_TARGET_SAMPLE_WIDTH,
            )
            loaded[uid] = seg
        except Exception as exc:
            logger.warning("Failed to load clip %s: %s", uid, exc)
    return loaded


def _build_sequential_timeline(
    clips: List[Dict[str, Any]],
    loaded: Dict[str, AudioSegment],
    gap_ms: int,
) -> Tuple[AudioSegment, Dict[str, Dict[str, int]]]:
    """Build baseline timeline by concatenating clips with gaps."""

    timeline = AudioSegment.empty()
    timestamp_map: Dict[str, Dict[str, int]] = {}

    sorted_clips = sorted(clips, key=lambda c: c.get("order_index", 0))
    for i, clip in enumerate(sorted_clips):
        uid = clip["utterance_id"]
        seg = loaded.get(uid)
        if seg is None:
            continue

        # Add gap before clip (except first)
        if i > 0 and gap_ms > 0:
            timeline += AudioSegment.silent(duration=gap_ms)

        start_ms = len(timeline)
        timeline += seg
        end_ms = len(timeline)

        timestamp_map[uid] = {"start_ms": start_ms, "end_ms": end_ms}

    return timeline, timestamp_map


def _resolve_timing_directives(
    directives: List[Dict[str, Any]],
    clips: List[Dict[str, Any]],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Classify timing directives into interrupt, backchannel, and laugh ops."""

    clip_ids = {c["utterance_id"] for c in clips}
    interrupt_ops, backchannel_ops, laugh_ops = [], [], []

    for d in directives:
        d_type = d.get("type", "").upper()
        uid = d.get("utterance_id", "")
        if uid not in clip_ids:
            logger.warning("Directive references unknown utterance %s, skipping", uid)
            continue

        if d_type == "INTERRUPT":
            interrupt_ops.append(d)
        elif d_type == "BACKCHANNEL":
            backchannel_ops.append(d)
        elif d_type == "LAUGH":
            laugh_ops.append(d)

    return interrupt_ops, backchannel_ops, laugh_ops


def _apply_interrupt_ops(
    timeline: AudioSegment,
    timestamp_map: Dict[str, Dict[str, int]],
    interrupt_ops: List[Dict[str, Any]],
    loaded: Dict[str, AudioSegment],
    clips: List[Dict[str, Any]],
) -> Tuple[AudioSegment, Dict[str, Dict[str, int]], Set[str]]:
    """Apply INTERRUPT directives — superimpose both speakers for up to
    PHASE5_INTERRUPT_MAX_DUAL_PLAY_MS, then fade out the interrupted speaker.

    Returns (timeline, timestamp_map, interrupted_uids) where
    *interrupted_uids* marks boundaries that crossfades should skip.
    """

    sorted_clips = sorted(clips, key=lambda c: c.get("order_index", 0))
    clip_order = [c["utterance_id"] for c in sorted_clips]
    uid_to_order = {c["utterance_id"]: c.get("order_index", 0) for c in clips}

    max_dual_ms = settings.PHASE5_INTERRUPT_MAX_DUAL_PLAY_MS
    fade_out_ms = settings.PHASE5_INTERRUPT_FADE_OUT_MS
    interrupted_uids: Set[str] = set()

    # Process ops in clip order so timestamp shifts propagate correctly
    ops_sorted = sorted(
        interrupt_ops,
        key=lambda op: uid_to_order.get(op.get("utterance_id", ""), 0),
    )

    for op in ops_sorted:
        target_uid = op.get("utterance_id", "")
        overlap_ms = int(op.get("duration_ms", 200))

        if target_uid not in clip_order:
            continue
        idx = clip_order.index(target_uid)
        if idx + 1 >= len(clip_order):
            continue
        next_uid = clip_order[idx + 1]

        target_ts = timestamp_map.get(target_uid)
        next_ts = timestamp_map.get(next_uid)
        if not target_ts or not next_ts:
            continue

        next_clip = loaded.get(next_uid)
        if next_clip is None:
            continue

        # --- Where the interrupter begins (shifted backwards) ---
        overlap_start = max(
            next_ts["start_ms"] - overlap_ms,
            target_ts["start_ms"] + 100,
        )
        actual_overlap = next_ts["start_ms"] - overlap_start
        if actual_overlap <= 0:
            continue

        # Cap simultaneous playback to max_dual_ms
        dual_play_ms = min(actual_overlap, max_dual_ms)

        # --- Build the superimposed section ---
        # Interrupted speaker's continuing audio during dual-play window
        old_audio = timeline[overlap_start: overlap_start + dual_play_ms]
        # Interrupter's audio for the same window
        new_audio = next_clip[:dual_play_ms]

        # Trim to the shorter of the two to prevent length mismatch
        seg_len = min(len(old_audio), len(new_audio))
        if seg_len <= 0:
            continue
        old_audio = old_audio[:seg_len]
        new_audio = new_audio[:seg_len]

        # Fade out the interrupted speaker at the tail of dual play
        actual_fade = min(fade_out_ms, seg_len // 2)
        if actual_fade > 0:
            old_audio = old_audio.fade_out(actual_fade)

        # Slight volume reduction on interrupted speaker so interrupter cuts through
        old_audio = old_audio + settings.PHASE5_INTERRUPT_VOLUME_REDUCTION_DB

        # Superimpose: both speakers audible simultaneously
        mixed = old_audio.overlay(new_audio)

        # --- Rebuild timeline without duplication ---
        before = timeline[:overlap_start]
        remaining_interrupter = next_clip[seg_len:]  # rest of interrupter after dual section
        after = timeline[next_ts["end_ms"]:]  # everything past original next-clip end

        old_len = len(timeline)
        timeline = before + mixed + remaining_interrupter + after
        new_len = len(timeline)
        shift = new_len - old_len

        # --- Update timestamp map ---
        new_next_end = overlap_start + seg_len + len(remaining_interrupter)
        timestamp_map[next_uid] = {
            "start_ms": overlap_start,
            "end_ms": new_next_end,
        }

        for subsequent_uid in clip_order[idx + 2:]:
            if subsequent_uid in timestamp_map:
                timestamp_map[subsequent_uid]["start_ms"] += shift
                timestamp_map[subsequent_uid]["end_ms"] += shift

        interrupted_uids.add(target_uid)
        logger.debug(
            "Interrupt: %s overlapped by %s — dual play %d ms (cap %d ms)",
            target_uid, next_uid, seg_len, max_dual_ms,
        )

    return timeline, timestamp_map, interrupted_uids


def _apply_backchannel_ops(
    timeline: AudioSegment,
    timestamp_map: Dict[str, Dict[str, int]],
    backchannel_ops: List[Dict[str, Any]],
) -> AudioSegment:
    """Apply BACKCHANNEL directives — overlay short sounds during speech."""

    for op in backchannel_ops:
        during_uid = op.get("utterance_id", "")
        ts = timestamp_map.get(during_uid)
        if not ts:
            continue

        asset_path = Path(settings.BASE_DIR) / "data/audio/assets/backchannels/mmhm_neutral.wav"
        if not asset_path.exists():
            logger.debug("No backchannel asset at %s, skipping", asset_path)
            continue

        try:
            bc_clip = AudioSegment.from_wav(str(asset_path))
            bc_clip = convert_to_pipeline_format(bc_clip)
            bc_clip = bc_clip + settings.PHASE5_BACKCHANNEL_VOLUME_DB
        except Exception as exc:
            logger.warning("Failed to load backchannel asset: %s", exc)
            continue

        # Insert at 30% into the utterance
        utterance_dur = ts["end_ms"] - ts["start_ms"]
        insertion_ms = ts["start_ms"] + int(utterance_dur * 0.3)
        timeline = timeline.overlay(bc_clip, position=insertion_ms)

    return timeline


def _apply_laugh_ops(
    timeline: AudioSegment,
    timestamp_map: Dict[str, Dict[str, int]],
    laugh_ops: List[Dict[str, Any]],
) -> AudioSegment:
    """Apply LAUGH directives — overlay laughter after utterance."""

    for op in laugh_ops:
        after_uid = op.get("utterance_id", "")
        ts = timestamp_map.get(after_uid)
        if not ts:
            continue

        laugh_type = op.get("laugh_type", "light")
        asset_name = f"laugh_{laugh_type}.wav"
        asset_path = Path(settings.BASE_DIR) / f"data/audio/assets/laughs/{asset_name}"
        if not asset_path.exists():
            asset_path = Path(settings.BASE_DIR) / "data/audio/assets/laughs/laugh_light.wav"
        if not asset_path.exists():
            logger.debug("No laugh asset found, skipping")
            continue

        try:
            laugh_clip = AudioSegment.from_wav(str(asset_path))
            laugh_clip = convert_to_pipeline_format(laugh_clip)
            laugh_clip = laugh_clip + settings.PHASE5_LAUGH_VOLUME_DB
        except Exception as exc:
            logger.warning("Failed to load laugh asset: %s", exc)
            continue

        timeline = timeline.overlay(laugh_clip, position=ts["end_ms"])

    return timeline


def _apply_crossfades(
    timeline: AudioSegment,
    clips: List[Dict[str, Any]],
    timestamp_map: Dict[str, Dict[str, int]],
    fade_ms: int,
    skip_uids: Set[str] | None = None,
) -> AudioSegment:
    """Apply crossfades at speaker-turn boundaries.

    Boundaries whose *current* utterance ID is in *skip_uids* are skipped
    because an interrupt already handled the transition there.
    """

    sorted_clips = sorted(clips, key=lambda c: c.get("order_index", 0))
    if skip_uids is None:
        skip_uids = set()

    for i in range(len(sorted_clips) - 1):
        curr = sorted_clips[i]
        nxt = sorted_clips[i + 1]
        if curr.get("speaker") == nxt.get("speaker"):
            continue
        # Skip boundaries already handled by interrupt superimposition
        if curr["utterance_id"] in skip_uids:
            continue

        curr_ts = timestamp_map.get(curr["utterance_id"])
        nxt_ts = timestamp_map.get(nxt["utterance_id"])
        if not curr_ts or not nxt_ts:
            continue

        boundary = curr_ts["end_ms"]
        fade = min(fade_ms, boundary, len(timeline) - boundary)
        if fade <= 0:
            continue

        before = timeline[:boundary - fade]
        fading_out = timeline[boundary - fade:boundary].fade_out(fade)
        fading_in = timeline[boundary:boundary + fade].fade_in(fade)
        rest = timeline[boundary + fade:]
        timeline = before + fading_out + fading_in + rest

    return timeline


def run_overlap_engine(
    manifest: Dict[str, Any],
    timing_directives: List[Dict[str, Any]],
    output_dir: str,
    chapter_number: int,
) -> Tuple[str, Dict[str, Dict[str, int]], ChapterMixReport]:
    """Process a single chapter: load clips, apply overlaps, export mixed WAV."""

    clips = manifest.get("clips", [])
    report = ChapterMixReport(chapter_number=chapter_number, input_clip_count=len(clips))

    if not clips:
        logger.warning("Chapter %d has no clips", chapter_number)
        return "", {}, report

    # Load and normalize
    loaded = _load_and_normalize_clips(clips)
    if not loaded:
        logger.error("Chapter %d: no clips could be loaded", chapter_number)
        return "", {}, report

    # Build sequential timeline
    timeline, timestamp_map = _build_sequential_timeline(
        clips, loaded, settings.PHASE5_TURN_GAP_MS
    )

    # Resolve and apply directives
    interrupt_ops, backchannel_ops, laugh_ops = _resolve_timing_directives(
        timing_directives, clips
    )

    interrupted_uids: Set[str] = set()
    if interrupt_ops:
        timeline, timestamp_map, interrupted_uids = _apply_interrupt_ops(
            timeline, timestamp_map, interrupt_ops, loaded, clips
        )
        report.interrupts_applied = len(interrupt_ops)

    if backchannel_ops:
        timeline = _apply_backchannel_ops(timeline, timestamp_map, backchannel_ops)
        report.backchannels_applied = len(backchannel_ops)

    if laugh_ops:
        timeline = _apply_laugh_ops(timeline, timestamp_map, laugh_ops)
        report.laughs_applied = len(laugh_ops)

    # Crossfades (skip boundaries already handled by interrupts)
    timeline = _apply_crossfades(
        timeline, clips, timestamp_map, settings.PHASE5_CROSSFADE_MS,
        skip_uids=interrupted_uids,
    )

    # Export
    output_path = str(Path(output_dir) / f"chapter_{chapter_number}_overlap_mixed.wav")
    export_wav_atomic(timeline, output_path)

    report.total_duration_ms = len(timeline)
    report.output_path = output_path

    logger.info(
        "Chapter %d overlap mixed: %d clips, %d ms",
        chapter_number, len(loaded), len(timeline),
    )
    return output_path, timestamp_map, report
