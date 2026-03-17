"""Phase 5: Audio Post-Processing subgraph."""

from __future__ import annotations

import os
import wave
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from config.settings import settings
from src.agents.phase5.chapter_stitcher import run_chapter_stitcher
from src.agents.phase5.cold_open_generator import generate_cold_open
from src.agents.phase5.overlap_engine import run_overlap_engine
from src.agents.phase5.post_processor import run_mastering_chain
from src.models.phase5 import Phase5Output
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Phase5State(TypedDict, total=False):
    """State for Phase 5: Audio Post-Processing."""

    # Inherited from Phase 4
    phase4_output: Dict[str, Any]
    chapter_audio_manifests: List[Dict[str, Any]]
    timing_metadata: Dict[str, Any]
    audio_files: List[Dict[str, Any]]
    voice_metadata: Dict[str, Any]
    episode_id: str
    ready_for_phase5: bool

    # Inherited from earlier phases
    topic: str
    character_personas: List[Dict[str, Any]]
    chapter_dialogues: List[Dict[str, Any]]

    # Step 1: Input validation
    validated_manifests: List[Dict[str, Any]]
    degraded_chapter_numbers: List[int]
    validated_timing_directives: List[Dict[str, Any]]
    input_audio_format: Dict[str, Any]
    phase5_output_paths: Dict[str, str]
    phase5_blocked: bool
    phase5_validation_report: Dict[str, Any]

    # Step 2: Overlap Engine
    chapter_mixed_audio_paths: Dict[int, str]
    utterance_timestamp_maps: Dict[int, Dict[str, Dict[str, int]]]
    overlap_engine_reports: List[Dict[str, Any]]

    # Step 3: Post-Processor
    chapter_mastered_audio_paths: Dict[int, str]
    post_processor_reports: List[Dict[str, Any]]

    # Step 4: Cold Open Generator
    cold_open_path: str
    cold_open_failed: bool
    cold_open_report: Dict[str, Any]

    # Step 5: Chapter Stitcher
    final_podcast_mp3_path: str
    chapter_markers: Dict[str, Any]
    stitcher_report: Dict[str, Any]

    # Step 6: Output
    phase5_output: Dict[str, Any]
    final_podcast_mp3: str
    phase5_complete: bool


# ==================== NODE FUNCTIONS ====================


def validate_phase5_input(state: Phase5State) -> Phase5State:
    """Validate Phase 4 output before any audio processing."""

    print("\n🔍 PHASE 5 - INPUT CONTRACT VALIDATION")

    report: Dict[str, Any] = {"errors": [], "warnings": [], "stats": {}}
    degraded = []
    validated_manifests = []
    validated_directives = []

    # Check readiness flag
    p4_output = state.get("phase4_output", {})
    if not state.get("ready_for_phase5") and not p4_output.get("ready_for_phase5"):
        report["errors"].append("Phase 4 did not signal readiness for Phase 5")
        state["phase5_blocked"] = True
        state["phase5_validation_report"] = report
        print("   ❌ Blocked: Phase 4 not ready")
        return state

    # Get manifests from phase4_output or state
    manifests = state.get("chapter_audio_manifests") or p4_output.get("chapter_manifests", [])
    episode_id = state.get("episode_id") or p4_output.get("episode_id", "episode_001")
    state["episode_id"] = episode_id

    if not manifests:
        report["errors"].append("No chapter audio manifests found")
        state["phase5_blocked"] = True
        state["phase5_validation_report"] = report
        print("   ❌ Blocked: no manifests")
        return state

    # Validate each manifest
    total_clips = 0
    for manifest in manifests:
        ch_num = manifest.get("chapter_number", 0)
        clips = manifest.get("clips", [])

        if not clips:
            report["warnings"].append(f"Chapter {ch_num}: no clips")
            degraded.append(ch_num)
            continue

        # Check physical file existence
        missing = 0
        for clip in clips:
            path = clip.get("path", "")
            if not path or not Path(path).exists() or Path(path).stat().st_size == 0:
                missing += 1

        missing_ratio = missing / len(clips) if clips else 1.0
        if missing_ratio > 0.2:
            report["warnings"].append(f"Chapter {ch_num}: {missing}/{len(clips)} clips missing")
            degraded.append(ch_num)
        else:
            validated_manifests.append(manifest)
            total_clips += len(clips) - missing

        # Collect timing directives
        for d in manifest.get("timing_directives", []):
            d_type = d.get("type", "").upper()
            if d_type in ("INTERRUPT", "BACKCHANNEL", "LAUGH"):
                validated_directives.append(d)

    # Also collect from timing_metadata
    timing_meta = state.get("timing_metadata") or p4_output.get("timing_metadata", {})
    for ch_key, ch_directives in timing_meta.items():
        if isinstance(ch_directives, list):
            validated_directives.extend(ch_directives)

    if not validated_manifests:
        report["errors"].append("All chapters failed validation")
        state["phase5_blocked"] = True
        state["phase5_validation_report"] = report
        print("   ❌ Blocked: all chapters invalid")
        return state

    # Detect input audio format from first available clip
    input_format = {"sample_rate": 24000, "channels": 1, "sample_width": 2}
    for m in validated_manifests:
        for clip in m.get("clips", []):
            path = clip.get("path", "")
            if path and Path(path).exists():
                try:
                    with wave.open(path, "rb") as wf:
                        input_format = {
                            "sample_rate": wf.getframerate(),
                            "channels": wf.getnchannels(),
                            "sample_width": wf.getsampwidth(),
                        }
                    break
                except Exception:
                    pass
        if input_format["sample_rate"] != 24000:
            break

    # Resolve output paths
    base = Path(settings.BASE_DIR) / settings.PHASE5_OUTPUT_BASE_DIR / episode_id
    output_paths = {
        "overlap": str(base / "overlap"),
        "mastered": str(base / "mastered"),
        "final": str(base / "final"),
        "base": str(base),
    }
    for p in output_paths.values():
        Path(p).mkdir(parents=True, exist_ok=True)

    report["stats"] = {
        "manifests_validated": len(validated_manifests),
        "total_clips": total_clips,
        "degraded_chapters": len(degraded),
        "timing_directives": len(validated_directives),
    }

    state["validated_manifests"] = validated_manifests
    state["degraded_chapter_numbers"] = degraded
    state["validated_timing_directives"] = validated_directives
    state["input_audio_format"] = input_format
    state["phase5_output_paths"] = output_paths
    state["phase5_blocked"] = False
    state["phase5_validation_report"] = report

    print(
        f"   ✅ Validated: {len(validated_manifests)} chapters, "
        f"{total_clips} clips, {len(degraded)} degraded"
    )
    return state


def run_overlap_engine_node(state: Phase5State) -> Phase5State:
    """Mix sequential clips into natural conversation per chapter."""

    if state.get("phase5_blocked"):
        return state

    print("\n🔊 OVERLAP ENGINE")

    output_paths = state.get("phase5_output_paths", {})
    overlap_dir = output_paths.get("overlap", "")
    manifests = state.get("validated_manifests", [])
    directives = state.get("validated_timing_directives", [])

    mixed_paths: Dict[int, str] = {}
    timestamp_maps: Dict[int, Dict[str, Dict[str, int]]] = {}
    reports = []

    for manifest in manifests:
        ch_num = manifest.get("chapter_number", 0)
        # Filter directives for this chapter
        ch_directives = [
            d for d in directives
            if d.get("chapter_number") == ch_num
            or any(
                c["utterance_id"] == d.get("utterance_id")
                for c in manifest.get("clips", [])
            )
        ]

        try:
            path, ts_map, report = run_overlap_engine(
                manifest, ch_directives, overlap_dir, ch_num
            )
            if path:
                mixed_paths[ch_num] = path
                timestamp_maps[ch_num] = ts_map
            reports.append(report.model_dump())
        except Exception as exc:
            logger.error("Overlap engine failed for chapter %d: %s", ch_num, exc)
            reports.append({"chapter_number": ch_num, "error": str(exc)})

    state["chapter_mixed_audio_paths"] = mixed_paths
    state["utterance_timestamp_maps"] = timestamp_maps
    state["overlap_engine_reports"] = reports

    print(f"   ✅ Mixed {len(mixed_paths)} chapter(s)")
    return state


def run_post_processor_node(state: Phase5State) -> Phase5State:
    """Apply mastering chain to each chapter's mixed audio."""

    if state.get("phase5_blocked"):
        return state

    print("\n🎛️ POST-PROCESSOR (MASTERING)")

    mixed_paths = state.get("chapter_mixed_audio_paths", {})
    output_paths = state.get("phase5_output_paths", {})
    mastered_dir = output_paths.get("mastered", "")

    mastered_paths: Dict[int, str] = {}
    reports = []

    for ch_num, input_path in sorted(mixed_paths.items()):
        output_path = str(Path(mastered_dir) / f"chapter_{ch_num}_mastered.wav")
        try:
            report = run_mastering_chain(input_path, output_path, ch_num)
            mastered_paths[ch_num] = output_path
            reports.append(report.model_dump())
        except Exception as exc:
            logger.error("Mastering failed for chapter %d: %s", ch_num, exc)
            # Fall back to unmixed version
            mastered_paths[ch_num] = input_path
            reports.append({"chapter_number": ch_num, "error": str(exc)})

    state["chapter_mastered_audio_paths"] = mastered_paths
    state["post_processor_reports"] = reports

    print(f"   ✅ Mastered {len(mastered_paths)} chapter(s)")
    return state


def generate_cold_open_node(state: Phase5State) -> Phase5State:
    """Generate cold open teaser (best-effort, never blocks)."""

    if state.get("phase5_blocked"):
        return state

    print("\n🎬 COLD OPEN GENERATOR")

    output_paths = state.get("phase5_output_paths", {})
    base_dir = output_paths.get("base", "")

    try:
        path, report = generate_cold_open(
            chapter_dialogues=state.get("chapter_dialogues", []),
            chapter_mastered_paths=state.get("chapter_mastered_audio_paths", {}),
            timestamp_maps=state.get("utterance_timestamp_maps", {}),
            output_dir=base_dir,
        )
        state["cold_open_path"] = path
        state["cold_open_failed"] = not bool(path)
        state["cold_open_report"] = report.model_dump()

        if path:
            print(f"   ✅ Cold open: {report.duration_ms} ms")
        else:
            print("   ⚠️ Cold open generation failed (non-blocking)")
    except Exception as exc:
        logger.error("Cold open generation error: %s", exc)
        state["cold_open_path"] = ""
        state["cold_open_failed"] = True
        state["cold_open_report"] = {"error": str(exc)}
        print("   ⚠️ Cold open failed (non-blocking)")

    return state


def run_chapter_stitcher_node(state: Phase5State) -> Phase5State:
    """Assemble final episode MP3."""

    if state.get("phase5_blocked"):
        return state

    print("\n🧵 CHAPTER STITCHER")

    output_paths = state.get("phase5_output_paths", {})
    base_dir = output_paths.get("base", "")

    try:
        mp3_path, markers, report = run_chapter_stitcher(
            cold_open_path=state.get("cold_open_path"),
            cold_open_failed=state.get("cold_open_failed", True),
            chapter_mastered_paths=state.get("chapter_mastered_audio_paths", {}),
            chapter_dialogues=state.get("chapter_dialogues", []),
            topic=state.get("topic", "AI Podcast"),
            episode_id=state.get("episode_id", "episode_001"),
            output_dir=base_dir,
        )
        state["final_podcast_mp3_path"] = mp3_path
        state["chapter_markers"] = markers
        state["stitcher_report"] = report

        if mp3_path:
            print(f"   ✅ Episode assembled: {mp3_path}")
        else:
            print("   ❌ Episode assembly failed")
    except Exception as exc:
        logger.error("Chapter stitcher error: %s", exc)
        state["final_podcast_mp3_path"] = ""
        state["chapter_markers"] = {}
        state["stitcher_report"] = {"error": str(exc)}
        print(f"   ❌ Stitcher failed: {exc}")

    return state


def package_phase5_output(state: Phase5State) -> Phase5State:
    """Emit the Phase 5 output contract."""

    print("\n📦 PHASE 5 OUTPUT PACKAGING")

    mp3_path = state.get("final_podcast_mp3_path", "")
    stitcher = state.get("stitcher_report", {})
    degraded = state.get("degraded_chapter_numbers", [])
    mastered = state.get("chapter_mastered_audio_paths", {})

    file_size = 0
    duration_seconds = 0.0
    if mp3_path and Path(mp3_path).exists():
        file_size = Path(mp3_path).stat().st_size
        duration_seconds = stitcher.get("total_duration_ms", 0) / 1000.0

    output = Phase5Output(
        episode_id=state.get("episode_id", ""),
        final_podcast_path=mp3_path,
        total_duration_seconds=duration_seconds,
        file_size_bytes=file_size,
        chapter_count=len(mastered),
        degraded_chapters=degraded,
        cold_open_included=not state.get("cold_open_failed", True),
        mp3_bitrate_kbps=settings.PHASE5_MP3_BITRATE_KBPS,
        chapter_markers=state.get("chapter_markers", {}),
        ready=bool(mp3_path and Path(mp3_path).exists()),
    )

    state["phase5_output"] = output.model_dump()
    state["final_podcast_mp3"] = mp3_path
    state["phase5_complete"] = output.ready

    # Summary
    mins = int(duration_seconds // 60)
    secs = int(duration_seconds % 60)
    size_mb = file_size / (1024 * 1024) if file_size else 0

    print("=" * 40)
    print("PHASE 5 COMPLETE")
    print("=" * 40)
    print(f"Episode ID : {output.episode_id}")
    print(f"MP3 Path   : {mp3_path}")
    print(f"Duration   : {mins}:{secs:02d}")
    print(f"File Size  : {size_mb:.1f} MB")
    print(f"Chapters   : {output.chapter_count} ({len(degraded)} degraded)")
    print(f"Cold Open  : {'Yes' if output.cold_open_included else 'No'}")
    print(f"Loudness   : {output.loudness_target_lufs} LUFS")
    print(f"Bitrate    : {output.mp3_bitrate_kbps} kbps")
    print(f"Ready      : {'Yes' if output.ready else 'No'}")
    print("=" * 40)

    return state


# ==================== ROUTING ====================


def route_after_validation(state: Phase5State) -> str:
    """Fail fast when input is broken."""

    if state.get("phase5_blocked"):
        return "package_phase5_output"
    return "run_overlap_engine"


# ==================== GRAPH ====================


def create_phase5_graph():
    """Create and compile the Phase 5 audio post-processing graph."""

    workflow = StateGraph(Phase5State)

    workflow.add_node("validate_phase5_input", validate_phase5_input)
    workflow.add_node("run_overlap_engine", run_overlap_engine_node)
    workflow.add_node("run_post_processor", run_post_processor_node)
    workflow.add_node("generate_cold_open", generate_cold_open_node)
    workflow.add_node("run_chapter_stitcher", run_chapter_stitcher_node)
    workflow.add_node("package_phase5_output", package_phase5_output)

    workflow.set_entry_point("validate_phase5_input")
    workflow.add_conditional_edges(
        "validate_phase5_input",
        route_after_validation,
        {
            "run_overlap_engine": "run_overlap_engine",
            "package_phase5_output": "package_phase5_output",
        },
    )
    workflow.add_edge("run_overlap_engine", "run_post_processor")
    workflow.add_edge("run_post_processor", "generate_cold_open")
    workflow.add_edge("generate_cold_open", "run_chapter_stitcher")
    workflow.add_edge("run_chapter_stitcher", "package_phase5_output")
    workflow.add_edge("package_phase5_output", END)

    return workflow.compile()
