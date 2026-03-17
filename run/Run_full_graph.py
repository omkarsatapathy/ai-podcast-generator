"""Full podcast pipeline runner with per-phase caching and auto-resume.

Runs Phase 1 → 2 → 3 → 4 → 5 end-to-end, producing a final podcast MP3.
Caches state after each phase so a crash resumes from the last checkpoint.

Cache layout:
    data/cache/<topic_slug>_<hash>/
        manifest.json
        phase{1..5}_state.json

Output:
    data/audio/phase5/<episode_id>/final/podcast_episode_final.mp3

Usage:
    python run/Run_full_graph.py "Topic here" --speakers 2 --duration 20

    # Resume from cache (automatic):
    python run/Run_full_graph.py "Topic here"

    # Force restart from scratch:
    python run/Run_full_graph.py "Topic here" --fresh

    # Restart from a specific phase:
    python run/Run_full_graph.py "Topic here" --from-phase 3
"""

import os
import sys
import json
import shutil
import hashlib
import time
import logging
import argparse
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=False)

from config.settings import settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("podcast_pipeline")

PHASES = (1, 2, 3, 4, 5)
REQUIRED_ENV_KEYS = {
    "OPENAI_API_KEY": "OpenAI (research + dialogue)",
    "GOOGLE_SEARCH_API_KEY": "Google Custom Search",
    "GOOGLE_SEARCH_ENGINE_ID": "Google Search Engine ID",
    "GEMINI_API_KEY": "Gemini TTS (Phase 4)",
}

# Data subdirectories to clean before a fresh run
DATA_SUBDIRS = ("audio", "cache", "input", "output", "temp")


# ============================= Utilities ==================================

def _topic_cache_dir(topic: str) -> Path:
    """Deterministic cache directory for a topic string."""
    norm = topic.strip().lower()
    slug = norm.replace(" ", "_")[:40]
    digest = hashlib.sha256(norm.encode()).hexdigest()[:16]
    return settings.CACHE_DIR / f"{slug}_{digest}"


def _verify_env() -> bool:
    """Check that all required API keys are set. Returns True if OK."""
    ok = True
    for key, desc in REQUIRED_ENV_KEYS.items():
        val = os.getenv(key)
        if val and val != "your-openai-api-key-here":
            log.info("  [ok]  %-30s %s...", key, val[:16])
        else:
            log.error("  [!!]  %-30s NOT SET (%s)", key, desc)
            ok = False
    return ok


def _clean_data_folders() -> None:
    """Remove all files inside data/ subdirectories, keeping folders."""
    for name in DATA_SUBDIRS:
        folder = settings.DATA_DIR / name
        if not folder.exists():
            continue
        for child in folder.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        log.info("  Cleaned data/%s/", name)


# ============================= Cache ======================================

def _serialize_state(state: dict) -> dict:
    """Make state JSON-serialisable (handles LangChain message objects)."""
    out = {}
    for key, val in state.items():
        if key == "messages" and isinstance(val, list):
            out[key] = [
                m if isinstance(m, dict)
                else {"role": getattr(m, "type", "unknown"), "content": str(m.content)}
                for m in val
            ]
        else:
            out[key] = val
    return out


def save_phase(state: dict, cache_dir: Path, phase: int) -> None:
    """Persist state and update manifest after a phase completes."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    filepath = cache_dir / f"phase{phase}_state.json"
    with open(filepath, "w") as f:
        json.dump(_serialize_state(state), f, indent=2, default=str)

    # Build manifest entry
    manifest_file = cache_dir / "manifest.json"
    manifest = {}
    if manifest_file.exists():
        with open(manifest_file) as f:
            manifest = json.load(f)

    entry = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "state_keys": list(state.keys()),
        "ranked_chunks": len(state.get("ranked_chunks", [])),
        "chapters": len(state.get("chapter_outlines", [])),
        "personas": len(state.get("character_personas", [])),
        "dialogues": len(state.get("chapter_dialogues", [])),
    }
    if phase == 4:
        p4 = state.get("phase4_output") or {}
        entry["audio_files"] = len(p4.get("audio_files", []))
        entry["chapter_manifests"] = len(p4.get("chapter_manifests", []))
        entry["ready_for_phase5"] = p4.get("ready_for_phase5", False)
    elif phase == 5:
        p5 = state.get("phase5_output") or {}
        entry["final_podcast_path"] = p5.get("final_podcast_path", "")
        entry["total_duration_seconds"] = p5.get("total_duration_seconds", 0.0)
        entry["file_size_bytes"] = p5.get("file_size_bytes", 0)

    manifest[f"phase{phase}"] = entry
    manifest["topic"] = state.get("topic", "")
    manifest["latest_phase"] = max(manifest.get("latest_phase", 0), phase)

    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("  Cache saved → %s", filepath.name)


def load_phase(cache_dir: Path, phase: int) -> dict | None:
    """Load cached state for a phase. Returns None if missing/corrupt."""
    fp = cache_dir / f"phase{phase}_state.json"
    if not fp.exists():
        return None
    try:
        with open(fp) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Cache corrupt for phase %d: %s", phase, exc)
        return None


def latest_cached_phase(cache_dir: Path) -> int:
    """Return highest phase with a valid cache file, or 0."""
    for p in reversed(PHASES):
        if load_phase(cache_dir, p) is not None:
            return p
    return 0


# ============================= Phase helpers ==============================

def _trim_dialogues(dialogues: list, cap_minutes: float) -> tuple[list, float]:
    """Return dialogues fitting within *cap_minutes*. Never returns empty."""
    trimmed, total = [], 0.0
    for ch in dialogues:
        dur = float(ch.get("estimated_chapter_duration", 0))
        if trimmed and total + dur > cap_minutes:
            break
        trimmed.append(ch)
        total += dur
    return trimmed, total


# ============================= Phase runners ==============================

def _run_phase1(topic: str) -> dict:
    from src.pipeline.phases import create_phase1_graph

    graph = create_phase1_graph()
    state = graph.invoke({
        "topic": topic,
        "freshness": "",
        "seed_results": [],
        "seed_context": "",
        "queries": [],
        "current_date": "",
        "messages": [],
        "scraped_pages": [],
        "merged_research_text": "",
        "scrape_failure_rate": 0.0,
        "query_rewrite_count": 0,
    })

    scraped = state.get("scraped_pages", [])
    ok = sum(1 for p in scraped if p.get("success"))
    log.info("Phase 1 | topic=%s  freshness=%s  queries=%d  scraped=%d/%d  chunks=%d",
             state["topic"], state.get("freshness", "?"),
             len(state.get("queries", [])), ok, len(scraped),
             len(state.get("ranked_chunks", [])))
    return state


def _run_phase2(state: dict, num_speakers: int) -> dict:
    from src.pipeline.phases import create_phase2_graph

    state["num_speakers"] = num_speakers
    state = create_phase2_graph().invoke(state)

    log.info("Phase 2 | chapters=%d  duration=%.1f min  personas=%d",
             len(state.get("chapter_outlines", [])),
             state.get("total_estimated_duration", 0),
             len(state.get("character_personas", [])))
    return state


def _run_phase3(state: dict) -> dict:
    from src.pipeline.phases import create_phase3_graph

    state = create_phase3_graph().invoke(state)

    dialogues = state.get("chapter_dialogues", [])
    total_utts = sum(len(d.get("utterances", [])) for d in dialogues)
    log.info("Phase 3 | chapters=%d  utterances=%d", len(dialogues), total_utts)
    return state


def _run_phase4(state: dict, cap_minutes: float) -> dict:
    from src.pipeline.phases import create_phase4_graph

    all_dialogues = state.get("ssml_annotated_scripts") or state.get("chapter_dialogues") or []
    trimmed, trimmed_dur = _trim_dialogues(all_dialogues, cap_minutes)
    full_dur = sum(float(d.get("estimated_chapter_duration", 0)) for d in all_dialogues)

    log.info("Phase 4 | full=%.1f min → trimmed=%d ch / %.1f min  provider=%s",
             full_dur, len(trimmed), trimmed_dur, settings.TTS_PROVIDER)

    phase4_input = {
        "topic": state.get("topic", ""),
        "episode_id": state.get("episode_id", ""),
        "character_personas": state.get("character_personas", []),
        "chapter_dialogues": trimmed,
        "ssml_annotated_scripts": trimmed,
    }

    result = create_phase4_graph().invoke(phase4_input)

    metrics = result.get("phase4_summary_metrics") or {}
    log.info("Phase 4 | clips=%d  failed=%d  ready=%s",
             metrics.get("audio_files", 0),
             metrics.get("failed_jobs", 0),
             metrics.get("ready_for_phase5", False))

    merged = dict(state)
    merged.update(result)
    return merged


def _run_phase5(state: dict) -> dict:
    from src.pipeline.phases import create_phase5_graph

    p4 = state.get("phase4_output") or {}
    phase5_input = {
        "topic": state.get("topic", ""),
        "episode_id": state.get("episode_id") or p4.get("episode_id", ""),
        "character_personas": state.get("character_personas", []),
        "chapter_dialogues": state.get("chapter_dialogues", []),
        "chapter_audio_manifests": state.get("chapter_audio_manifests") or p4.get("chapter_manifests", []),
        "timing_metadata": state.get("timing_metadata") or p4.get("timing_metadata", {}),
        "audio_files": state.get("audio_files") or p4.get("audio_files", []),
        "voice_metadata": state.get("voice_metadata") or p4.get("voice_metadata", {}),
        "ready_for_phase5": state.get("ready_for_phase5") or p4.get("ready_for_phase5", False),
        "phase4_output": p4,
    }

    result = create_phase5_graph().invoke(phase5_input)

    p5 = result.get("phase5_output") or {}
    dur = p5.get("total_duration_seconds", 0)
    size_mb = p5.get("file_size_bytes", 0) / (1024 * 1024)
    log.info("Phase 5 | mp3=%s  duration=%d:%02d  size=%.1f MB  ready=%s",
             p5.get("final_podcast_path", "N/A"),
             int(dur // 60), int(dur % 60), size_mb,
             p5.get("ready", False))

    merged = dict(state)
    merged.update(result)
    return merged


# ============================= Pipeline ===================================

def run_pipeline(
    topic: str,
    num_speakers: int,
    duration_minutes: float,
    fresh: bool = False,
    from_phase: int | None = None,
) -> dict:
    """Execute the full 5-phase podcast pipeline with caching.

    Args:
        topic: Podcast topic string.
        num_speakers: Number of podcast participants (2-3).
        duration_minutes: Target podcast duration in minutes (caps Phase 4 TTS).
        fresh: If True, wipe data folders and ignore cache.
        from_phase: Force restart from this phase number (1-5).

    Returns:
        Final pipeline state dict.
    """
    cache_dir = _topic_cache_dir(topic)

    # Override global settings for this run
    settings.NUM_SPEAKERS = num_speakers
    settings.TARGET_DURATION_MINUTES = duration_minutes

    log.info("=" * 60)
    log.info("PODCAST PIPELINE")
    log.info("=" * 60)
    log.info("  Topic      : %s", topic)
    log.info("  Speakers   : %d", num_speakers)
    log.info("  Duration   : %.0f min", duration_minutes)
    log.info("  TTS        : %s", settings.TTS_PROVIDER)
    log.info("  Cache      : %s", cache_dir)
    log.info("  Fresh      : %s", fresh)
    log.info("  From-phase : %s", from_phase or "auto")

    # Preflight: env vars
    if not _verify_env():
        log.error("Missing API keys — aborting. Set them in .env")
        sys.exit(1)

    # Clean data folders on fresh run
    if fresh:
        log.info("Fresh run — cleaning data/ folders...")
        _clean_data_folders()

    # Determine resume point
    if fresh:
        resume_after = 0
    elif from_phase:
        resume_after = from_phase - 1
    else:
        resume_after = latest_cached_phase(cache_dir)

    # Load cached state
    state = None
    if resume_after > 0:
        state = load_phase(cache_dir, resume_after)
        if state is None:
            log.warning("Cache for phase %d unreadable — starting from scratch", resume_after)
            resume_after = 0
        else:
            log.info("Resuming after phase %d (phases 1..%d cached — $0 API cost)",
                     resume_after, resume_after)
    else:
        log.info("Starting from Phase 1 (no usable cache)")

    # Execute phases sequentially
    pipeline_t0 = time.time()
    for phase in PHASES:
        if phase <= resume_after:
            continue

        log.info("-" * 60)
        log.info("PHASE %d START", phase)
        log.info("-" * 60)
        t0 = time.time()

        try:
            if phase == 1:
                state = _run_phase1(topic)
            elif phase == 2:
                state = _run_phase2(state, num_speakers)
            elif phase == 3:
                state = _run_phase3(state)
            elif phase == 4:
                state = _run_phase4(state, duration_minutes)
            elif phase == 5:
                state = _run_phase5(state)
        except Exception:
            log.error("PHASE %d FAILED", phase)
            traceback.print_exc()
            if state and phase > 1:
                log.info("Previous phases are cached — rerun to resume from phase %d", phase)
            sys.exit(1)

        elapsed = time.time() - t0
        log.info("PHASE %d DONE in %.1fs", phase, elapsed)
        save_phase(state, cache_dir, phase)

    total_elapsed = time.time() - pipeline_t0
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE in %.1fs", total_elapsed)

    # Report final output
    p5 = (state or {}).get("phase5_output") or {}
    final_path = p5.get("final_podcast_path", "")
    if final_path:
        log.info("Final podcast: %s", final_path)
    log.info("=" * 60)

    return state


# ============================= CLI ========================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate a complete podcast from a topic.",
    )
    parser.add_argument("topic", help="Podcast topic")
    parser.add_argument(
        "--speakers", type=int, default=settings.NUM_SPEAKERS,
        help=f"Number of speakers (default: {settings.NUM_SPEAKERS})",
    )
    parser.add_argument(
        "--duration", type=float, default=settings.PHASE4_SYNTHESIS_MINUTES_CAP,
        help=f"Target duration in minutes (default: {settings.PHASE4_SYNTHESIS_MINUTES_CAP})",
    )
    parser.add_argument("--fresh", action="store_true", help="Clean data and ignore cache")
    parser.add_argument(
        "--from-phase", type=int, default=None, dest="from_phase",
        choices=[1, 2, 3, 4, 5], help="Force restart from this phase",
    )
    args = parser.parse_args()

    run_pipeline(
        topic=args.topic,
        num_speakers=args.speakers,
        duration_minutes=args.duration,
        fresh=args.fresh,
        from_phase=args.from_phase,
    )


if __name__ == "__main__":
    main()
