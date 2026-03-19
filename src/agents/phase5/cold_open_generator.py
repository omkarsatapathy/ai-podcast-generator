"""Phase 5: Cold Open Generator — LLM-powered teaser extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydub import AudioSegment

from config.settings import settings
from src.models.phase5 import ColdOpenCandidate, ColdOpenReport
from src.tools.audio_tools import convert_to_pipeline_format, export_wav_atomic
from src.utils.logger import get_logger

logger = get_logger(__name__)

COLD_OPEN_SYSTEM_PROMPT = (
    "You are a podcast producer assistant. Your job is to identify the single most "
    "compelling moment from a podcast episode script that will be used as the cold open "
    "(teaser). A compelling moment is one that: creates curiosity or surprise in a "
    "listener who hasn't heard the episode, features strong emotional contrast between "
    "speakers (excitement vs. skepticism, wonder vs. challenge), or contains a memorable "
    "analogy, a surprising fact, or a moment of genuine disagreement. You must NOT select "
    "opening or introductory moments — the cold open should feel like it drops the "
    "listener into the middle of something interesting."
)

COLD_OPEN_USER_PROMPT = (
    "Below is the full transcript of the episode. Identify the 3 best candidate moments. "
    "For each candidate, return a JSON object with the following fields:\n"
    "- `candidate_rank` (1 = best, 2 = second best, 3 = third best)\n"
    "- `chapter_number` (integer)\n"
    "- `start_utterance_id` (string — the utterance_id where the excerpt should start)\n"
    "- `end_utterance_id` (string — the utterance_id where the excerpt should end, inclusive)\n"
    "- `reason` (1-2 sentences explaining why this is compelling for a cold open)\n\n"
    "Return ONLY a JSON array with exactly 3 objects. No other text.\n\n"
    "TRANSCRIPT:\n{script_text}"
)


def _build_script_text(chapter_dialogues: List[Dict[str, Any]]) -> str:
    """Build compact script representation for LLM scanning."""

    lines = []
    for ch in chapter_dialogues:
        ch_num = ch.get("chapter_number", 0)
        for utt in ch.get("utterances", []):
            uid = utt.get("utterance_id", "")
            speaker = utt.get("speaker", "")
            text = utt.get("text_clean", "")
            lines.append(f"[CH{ch_num} | {speaker} | utterance_id={uid}]\n{text}")
    return "\n\n".join(lines)


def _scan_script_for_candidates(
    script_text: str,
) -> List[ColdOpenCandidate]:
    """Call LLM to identify cold open candidates."""

    try:
        from src.api_factory.llm import get_llm

        llm = get_llm(
            tier=settings.PHASE5_COLD_OPEN_LLM_MODEL,
            temperature=0.3,
        )
        prompt = COLD_OPEN_USER_PROMPT.format(script_text=script_text)
        response = llm.invoke(
            [
                {"role": "system", "content": COLD_OPEN_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        raw = response.content if hasattr(response, "content") else str(response)

        # Parse JSON array from response
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        candidates_data = json.loads(raw)

        candidates = []
        for item in candidates_data:
            candidates.append(ColdOpenCandidate(**item))
        return sorted(candidates, key=lambda c: c.candidate_rank)
    except Exception as exc:
        logger.warning("LLM cold open scan failed: %s", exc)
        return []


def _fallback_candidate(
    chapter_dialogues: List[Dict[str, Any]],
) -> ColdOpenCandidate | None:
    """Deterministic fallback: pick beat 3 (Deep Dive) from longest chapter."""

    best_ch = max(chapter_dialogues, key=lambda c: len(c.get("utterances", [])), default=None)
    if not best_ch:
        return None

    beat3_utts = [u for u in best_ch.get("utterances", []) if u.get("beat") == 3]
    if len(beat3_utts) < 2:
        # Fall back to middle utterances
        utts = best_ch.get("utterances", [])
        mid = len(utts) // 2
        beat3_utts = utts[max(0, mid - 1):mid + 2]

    if not beat3_utts:
        return None

    return ColdOpenCandidate(
        candidate_rank=1,
        chapter_number=best_ch.get("chapter_number", 1),
        start_utterance_id=beat3_utts[0].get("utterance_id", ""),
        end_utterance_id=beat3_utts[-1].get("utterance_id", ""),
        reason="Fallback: beat 3 from longest chapter",
    )


def _validate_candidate(
    candidate: ColdOpenCandidate,
    timestamp_maps: Dict[int, Dict[str, Dict[str, int]]],
) -> bool:
    """Check that candidate utterance IDs exist in timestamp maps."""

    ch_map = timestamp_maps.get(candidate.chapter_number, {})
    return (
        candidate.start_utterance_id in ch_map
        and candidate.end_utterance_id in ch_map
    )


def _adjust_excerpt_duration(
    candidate: ColdOpenCandidate,
    chapter_dialogues: List[Dict[str, Any]],
    timestamp_maps: Dict[int, Dict[str, Dict[str, int]]],
) -> ColdOpenCandidate:
    """Adjust excerpt to fit within 12-25 second window."""

    ch_map = timestamp_maps.get(candidate.chapter_number, {})
    start_ts = ch_map.get(candidate.start_utterance_id, {})
    end_ts = ch_map.get(candidate.end_utterance_id, {})
    duration = end_ts.get("end_ms", 0) - start_ts.get("start_ms", 0)

    if settings.PHASE5_COLD_OPEN_MIN_MS <= duration <= settings.PHASE5_COLD_OPEN_MAX_MS:
        return candidate

    # Get ordered utterance IDs for this chapter
    ch_data = next(
        (c for c in chapter_dialogues if c.get("chapter_number") == candidate.chapter_number),
        None,
    )
    if not ch_data:
        return candidate

    utt_ids = [u["utterance_id"] for u in ch_data.get("utterances", []) if u["utterance_id"] in ch_map]
    if not utt_ids:
        return candidate

    start_idx = utt_ids.index(candidate.start_utterance_id) if candidate.start_utterance_id in utt_ids else 0
    end_idx = utt_ids.index(candidate.end_utterance_id) if candidate.end_utterance_id in utt_ids else len(utt_ids) - 1

    if duration < settings.PHASE5_COLD_OPEN_MIN_MS:
        # Extend end forward
        while end_idx < len(utt_ids) - 1:
            end_idx += 1
            new_end = ch_map.get(utt_ids[end_idx], {})
            duration = new_end.get("end_ms", 0) - start_ts.get("start_ms", 0)
            if duration >= settings.PHASE5_COLD_OPEN_MIN_MS:
                break
        candidate.end_utterance_id = utt_ids[end_idx]
    elif duration > settings.PHASE5_COLD_OPEN_MAX_MS:
        # Trim end backward
        while end_idx > start_idx:
            end_idx -= 1
            new_end = ch_map.get(utt_ids[end_idx], {})
            duration = new_end.get("end_ms", 0) - start_ts.get("start_ms", 0)
            if duration <= settings.PHASE5_COLD_OPEN_MAX_MS:
                break
        candidate.end_utterance_id = utt_ids[end_idx]

    return candidate


def generate_cold_open(
    chapter_dialogues: List[Dict[str, Any]],
    chapter_mastered_paths: Dict[int, str],
    timestamp_maps: Dict[int, Dict[str, Dict[str, int]]],
    output_dir: str,
) -> Tuple[str, ColdOpenReport]:
    """Generate the cold open teaser clip."""

    report = ColdOpenReport()

    # Build script text for LLM
    script_text = _build_script_text(chapter_dialogues)

    # LLM scan
    candidates = _scan_script_for_candidates(script_text)
    report.llm_used = len(candidates) > 0

    # Select best valid candidate
    selected = None
    for c in candidates:
        if _validate_candidate(c, timestamp_maps):
            selected = c
            break

    # Fallback if no valid candidate
    if not selected:
        selected = _fallback_candidate(chapter_dialogues)
        if selected and _validate_candidate(selected, timestamp_maps):
            report.fallback_used = True
        else:
            logger.error("Cold open generation failed: no valid candidate found")
            return "", report

    # Adjust duration
    selected = _adjust_excerpt_duration(selected, chapter_dialogues, timestamp_maps)

    # Extract audio slice
    ch_map = timestamp_maps.get(selected.chapter_number, {})
    start_ms = ch_map[selected.start_utterance_id]["start_ms"]
    end_ms = ch_map[selected.end_utterance_id]["end_ms"]

    mastered_path = chapter_mastered_paths.get(selected.chapter_number)
    if not mastered_path or not Path(mastered_path).exists():
        logger.error("Mastered audio for chapter %d not found", selected.chapter_number)
        return "", report

    chapter_audio = AudioSegment.from_wav(mastered_path)
    excerpt = chapter_audio[start_ms:end_ms]
    excerpt = excerpt.fade_in(200).fade_out(300)

    # Try framing line
    framing_line = _load_framing_line()
    if framing_line:
        report.framing_strategy = "pre_recorded"
    else:
        report.framing_strategy = "none"

    # Assemble cold open
    cold_open = AudioSegment.empty()
    if framing_line:
        cold_open += framing_line + AudioSegment.silent(duration=100)
    cold_open += excerpt

    # Transition sound
    transition_path = Path(settings.BASE_DIR) / "data/audio/assets/transitions/cold_open_end.wav"
    if transition_path.exists():
        try:
            transition = AudioSegment.from_wav(str(transition_path))
            transition = convert_to_pipeline_format(transition)
            cold_open += AudioSegment.silent(duration=500) + transition.fade_in(100)
        except Exception:
            cold_open += AudioSegment.silent(duration=1000)
    else:
        cold_open += AudioSegment.silent(duration=1000)

    # Export
    output_path = str(Path(output_dir) / "cold_open.wav")
    export_wav_atomic(cold_open, output_path)

    report.selected_chapter_number = selected.chapter_number
    report.start_utterance_id = selected.start_utterance_id
    report.end_utterance_id = selected.end_utterance_id
    report.duration_ms = len(cold_open)

    logger.info(
        "Cold open generated: chapter=%d, duration=%d ms",
        selected.chapter_number, len(cold_open),
    )
    return output_path, report


def _load_framing_line() -> AudioSegment | None:
    """Load pre-recorded framing line if available."""

    path = Path(settings.BASE_DIR) / "data/audio/assets/framing/later_in_this_episode.wav"
    if not path.exists():
        return None
    try:
        seg = AudioSegment.from_wav(str(path))
        return convert_to_pipeline_format(seg)
    except Exception as exc:
        logger.warning("Failed to load framing line: %s", exc)
        return None
