"""Full pipeline test with per-phase caching and automatic resume.

Runs Phase 1 → Phase 2 → Phase 3 → Phase 4, caching state after each phase.
On subsequent runs, automatically resumes from the latest cached phase.

Phase 4 note: when resuming from a Phase 3 cache, only the FIRST
--minutes of content (default 15) are synthesised to cap TTS cost.
The trimmed Phase 4 output is cached as phase4_state.json for Phase 5.

Cache structure:
    data/cache/<topic_slug>_<hash>/
        manifest.json          # Quick summary of what's cached
        phase1_state.json      # State after Phase 1
        phase2_state.json      # State after Phase 2
        phase3_state.json      # State after Phase 3
        phase4_state.json      # State after Phase 4 (trimmed to --minutes)

Usage:
    # Run Phases 1-3 then Phase 4 on first 15 min:
    python tests/test_graph_with_cached.py

    # Custom topic:
    python tests/test_graph_with_cached.py "AI regulation in Europe"

    # Force re-run from scratch (ignore all cache):
    python tests/test_graph_with_cached.py --fresh

    # Force re-run starting at a specific phase:
    python tests/test_graph_with_cached.py --from-phase 4

    # Change the Phase 4 content slice (e.g. 20 minutes):
    python tests/test_graph_with_cached.py --minutes 20
"""

import os
import sys
import json
import hashlib
import time
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env", override=False)

from config.settings import settings

CACHE_DIR = project_root / "data" / "cache"
PHASES = [1, 2, 3, 4]
PHASE4_DEFAULT_MINUTES = 15.0


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path_for_topic(topic: str) -> Path:
    """Return a deterministic cache directory for *topic*."""
    topic_norm = topic.strip().lower()
    topic_hash = hashlib.sha256(topic_norm.encode()).hexdigest()[:16]
    slug = topic_norm.replace(" ", "_")[:40]
    return CACHE_DIR / f"{slug}_{topic_hash}"


def _serialize_state(state: dict) -> dict:
    """Make a state dict JSON-safe (handles LangChain message objects)."""
    out = {}
    for key, value in state.items():
        if key == "messages" and isinstance(value, list):
            out[key] = [
                m if isinstance(m, dict)
                else {"role": getattr(m, "type", "unknown"), "content": str(m.content)}
                for m in value
            ]
        else:
            out[key] = value
    return out


def save_phase_state(state: dict, cache_path: Path, phase: int) -> None:
    """Persist state to ``cache_path/phaseN_state.json``."""
    cache_path.mkdir(parents=True, exist_ok=True)

    filepath = cache_path / f"phase{phase}_state.json"
    with open(filepath, "w") as f:
        json.dump(_serialize_state(state), f, indent=2, default=str)

    # Update manifest
    manifest_file = cache_path / "manifest.json"
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
        entry["failed_jobs"] = len(p4.get("failed_jobs", []))

    manifest[f"phase{phase}"] = entry
    manifest["topic"] = state.get("topic", "")
    manifest["latest_phase"] = max(manifest.get("latest_phase", 0), phase)

    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"   >> Cache saved: {filepath.name}")


def load_phase_state(cache_path: Path, phase: int) -> dict | None:
    """Load cached state for *phase*. Returns ``None`` when missing/corrupt."""
    filepath = cache_path / f"phase{phase}_state.json"
    if not filepath.exists():
        return None
    try:
        with open(filepath) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"   !! Cache corrupt for phase {phase}: {exc}")
        return None


def find_latest_cached_phase(cache_path: Path) -> int:
    """Return the highest phase number with a valid cache, or 0 if none."""
    for phase in reversed(PHASES):
        if load_phase_state(cache_path, phase) is not None:
            return phase
    return 0


# ---------------------------------------------------------------------------
# Phase 4: content trimmer
# ---------------------------------------------------------------------------

def _trim_dialogues_to_minutes(
    dialogues: list, target_minutes: float
) -> tuple[list, float]:
    """Return the first chapters whose cumulative duration fits within target.

    Never returns an empty list — if the very first chapter already exceeds
    the target we still include it so there is always something to synthesise.
    """
    trimmed = []
    accumulated = 0.0
    for chapter in dialogues:
        dur = float(chapter.get("estimated_chapter_duration", 0.0))
        if trimmed and accumulated + dur > target_minutes:
            break
        trimmed.append(chapter)
        accumulated += dur
    return trimmed, accumulated


# ---------------------------------------------------------------------------
# Env check
# ---------------------------------------------------------------------------

def check_env_vars() -> bool:
    """Verify required API keys are present."""
    print("Checking environment variables...\n")
    required = {
        "OPENAI_API_KEY": "OpenAI API key",
        "GOOGLE_SEARCH_API_KEY": "Google Search API key",
        "GOOGLE_SEARCH_ENGINE_ID": "Google Search Engine ID",
        "GEMINI_API_KEY": "Gemini API key (Phase 4 TTS)",
    }
    ok = True
    for key, desc in required.items():
        val = os.getenv(key)
        if val and val != "your-openai-api-key-here":
            print(f"  [ok]  {key}: {val[:20]}...")
        else:
            print(f"  [!!]  {key}: NOT SET ({desc})")
            ok = False
    print()
    return ok


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_phase1(topic: str) -> dict:
    """Execute Phase 1: Research & Ingestion."""
    from src.pipeline.phases import create_phase1_graph

    print("\n" + "=" * 70)
    print("  PHASE 1: RESEARCH & INGESTION")
    print("=" * 70 + "\n")

    graph = create_phase1_graph()

    initial_state = {
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
    }

    state = graph.invoke(initial_state)

    ranked = state.get("ranked_chunks", [])
    dedup = state.get("dedup_stats", {})
    scraped = state.get("scraped_pages", [])
    succeeded = sum(1 for p in scraped if p.get("success"))
    merged_words = len(state.get("merged_research_text", "").split())

    print(f"\n--- Phase 1 Summary ---")
    print(f"  Topic:           {state['topic']}")
    print(f"  Freshness:       {state.get('freshness', '?')}")
    print(f"  Queries:         {len(state.get('queries', []))}")
    print(f"  Query rewrites:  {state.get('query_rewrite_count', 0)}")
    print(f"  Pages scraped:   {succeeded}/{len(scraped)}")
    print(f"  Merged text:     {merged_words} words")
    print(f"  Ranked chunks:   {len(ranked)}")
    if dedup:
        print(f"  Dedup stats:     total={dedup.get('total_chunks')}, "
              f"removed={dedup.get('duplicates_removed')}, "
              f"top_k={dedup.get('top_k_selected')}")
    print()
    return state


def run_phase2(state: dict) -> dict:
    """Execute Phase 2: Content Planning."""
    from src.pipeline.phases import create_phase2_graph

    print("\n" + "=" * 70)
    print("  PHASE 2: CONTENT PLANNING")
    print("=" * 70 + "\n")

    state["num_speakers"] = settings.NUM_SPEAKERS

    graph = create_phase2_graph()
    state = graph.invoke(state)

    outlines = state.get("chapter_outlines", [])
    personas = state.get("character_personas", [])

    print(f"\n--- Phase 2 Summary ---")
    print(f"  Chapters:        {len(outlines)}")
    print(f"  Duration:        {state.get('total_estimated_duration', 0):.1f} min")
    print(f"  Personas:        {len(personas)}")

    for ch in outlines:
        print(f"    Ch {ch['chapter_number']}: {ch['title']}  "
              f"({ch['estimated_duration_minutes']:.1f} min, {ch['energy_level']} energy)")

    for p in personas:
        print(f"    {p['role'].upper()}: {p['name']} — {p['expertise_area']}")

    print()
    return state


def run_phase3(state: dict) -> dict:
    """Execute Phase 3: Dialogue Generation."""
    from src.pipeline.phases import create_phase3_graph

    print("\n" + "=" * 70)
    print("  PHASE 3: DIALOGUE GENERATION")
    print("=" * 70 + "\n")

    graph = create_phase3_graph()
    state = graph.invoke(state)

    dialogues = state.get("chapter_dialogues", [])
    total_utts = sum(len(cd.get("utterances", [])) for cd in dialogues)
    total_dur = sum(cd.get("estimated_chapter_duration", 0) for cd in dialogues)

    print(f"\n--- Phase 3 Summary ---")
    print(f"  Chapters:        {len(dialogues)}")
    print(f"  Utterances:      {total_utts}")
    print(f"  Duration:        {total_dur:.1f} min")

    for cd in dialogues:
        passed = cd.get("quality_checks_passed", False)
        score = cd.get("qa_review", {}).get("listener_experience_score", "?")
        issues = len(cd.get("fact_check_issues", []))
        tag = "[PASS]" if passed else "[WARN]"
        print(f"    {tag} Ch {cd['chapter_number']}: "
              f"{len(cd.get('utterances', []))} utts, "
              f"score={score}, fact_issues={issues}")

    print()
    return state


def run_phase4(state: dict, target_minutes: float) -> dict:
    """Execute Phase 4: Voice Synthesis on the first *target_minutes* of content."""
    from src.pipeline.phases import create_phase4_graph

    print("\n" + "=" * 70)
    print("  PHASE 4: VOICE SYNTHESIS")
    print("=" * 70 + "\n")

    # Grab whichever field Phase 3 produced
    all_dialogues = (
        state.get("ssml_annotated_scripts")
        or state.get("chapter_dialogues")
        or []
    )
    full_dur = sum(float(cd.get("estimated_chapter_duration", 0)) for cd in all_dialogues)

    trimmed, trimmed_dur = _trim_dialogues_to_minutes(all_dialogues, target_minutes)

    print(f"  Full Phase 3 content : {len(all_dialogues)} chapters, {full_dur:.1f} min")
    print(f"  Slice for this run   : {len(trimmed)} chapters, {trimmed_dur:.1f} min "
          f"(target ≤ {target_minutes} min)")
    print(f"  TTS provider         : {settings.TTS_PROVIDER}")
    print(f"  TTS model            : {settings.GOOGLE_TTS_MODEL}")
    print(f"  Max workers          : {settings.PHASE4_MAX_WORKERS}")
    print(f"  Max retries/job      : {settings.PHASE4_MAX_RETRIES}")
    print(f"  Failure ratio cap    : {settings.PHASE4_MAX_FAILURE_RATIO * 100:.0f}%")
    print()

    # Build a clean Phase 4 input state using only the trimmed slice
    phase4_input = {
        "topic": state.get("topic", ""),
        "episode_id": state.get("episode_id", ""),
        "character_personas": state.get("character_personas", []),
        "chapter_dialogues": trimmed,
        "ssml_annotated_scripts": trimmed,
    }

    graph = create_phase4_graph()
    result = graph.invoke(phase4_input)

    p4_output = result.get("phase4_output") or {}
    manifests = p4_output.get("chapter_manifests", [])
    failed = p4_output.get("failed_jobs", [])
    metrics = result.get("phase4_summary_metrics") or {}

    print(f"\n--- Phase 4 Summary ---")
    print(f"  Chapters synthesised : {metrics.get('chapters', 0)}")
    print(f"  Audio clips produced : {metrics.get('audio_files', 0)}")
    print(f"  Failed jobs          : {metrics.get('failed_jobs', 0)}")
    print(f"  Ready for Phase 5    : {metrics.get('ready_for_phase5', False)}")

    for manifest in manifests:
        ch = manifest.get("chapter_number", "?")
        complete = "[COMPLETE]" if manifest.get("complete") else "[INCOMPLETE]"
        clips = len(manifest.get("clips", []))
        print(f"    {complete} Ch {ch}: {clips} clip(s)")

    if failed:
        print(f"\n  Failed jobs ({len(failed)}):")
        for job in failed[:5]:
            print(f"    - {job.get('job_id', '?')}: {job.get('error', '?')}")
        if len(failed) > 5:
            print(f"    ... and {len(failed) - 5} more (see phase4_state.json)")

    print()

    # Merge Phase 4 result back onto the full state for caching
    # (keeps personas, topic, etc. intact for Phase 5)
    merged = dict(state)
    merged.update(result)
    return merged


# ---------------------------------------------------------------------------
# Save final results
# ---------------------------------------------------------------------------

def save_results(state: dict) -> None:
    """Write Phase 3 and Phase 4 outputs for inspection."""
    out_dir = settings.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    dialogues = state.get("chapter_dialogues", [])
    result_file = out_dir / "phase3_results.json"
    with open(result_file, "w") as f:
        json.dump(dialogues, f, indent=2, default=str)
    print(f"  Phase 3 results  : {result_file}")

    p4_output = state.get("phase4_output")
    if p4_output:
        p4_file = out_dir / "phase4_results.json"
        with open(p4_file, "w") as f:
            json.dump(p4_output, f, indent=2, default=str)
        print(f"  Phase 4 results  : {p4_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PHASE_RUNNERS = {
    1: run_phase1,
    2: run_phase2,
    3: run_phase3,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run podcast pipeline (Phases 1-4) with per-phase caching.",
    )
    parser.add_argument("topic", nargs="*",
                        default=["Why Gold price is falling, and the price forecast by end of year"])
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore all cache and re-run from Phase 1")
    parser.add_argument("--from-phase", type=int, default=None, dest="from_phase",
                        help="Force re-run starting at this phase (1–4)")
    parser.add_argument("--minutes", type=float, default=PHASE4_DEFAULT_MINUTES,
                        dest="minutes",
                        help=f"Minutes of Phase 3 content to synthesise in Phase 4 "
                             f"(default: {PHASE4_DEFAULT_MINUTES})")
    args = parser.parse_args()

    topic = " ".join(args.topic)
    cache_path = _cache_path_for_topic(topic)

    print("\n" + "=" * 70)
    print("  PODCAST PIPELINE (with per-phase caching)")
    print("=" * 70)
    print(f"  Topic:           {topic}")
    print(f"  Cache dir:       {cache_path}")
    print(f"  TTS provider:    {settings.TTS_PROVIDER}")
    print(f"  Phase 4 slice:   {args.minutes} min")
    print(f"  Fresh:           {args.fresh}")
    print(f"  From-phase:      {args.from_phase or 'auto'}")
    print()

    if not check_env_vars():
        print("Please set missing API keys in .env")
        return

    # Determine where to resume from
    if args.fresh:
        resume_after = 0
    elif args.from_phase:
        resume_after = args.from_phase - 1
    else:
        resume_after = find_latest_cached_phase(cache_path)

    # Load cached state (if resuming)
    if resume_after > 0:
        state = load_phase_state(cache_path, resume_after)
        if state is None:
            print(f"  !! Could not load phase {resume_after} cache, starting from scratch")
            resume_after = 0
            state = None
        else:
            print(f"  >> Resuming after phase {resume_after} (cached)")
            print(f"     Phases 1..{resume_after} skipped — $0 API cost")
            print()
    else:
        state = None
        print("  >> Starting from Phase 1 (no usable cache)")
        print()

    # Run remaining phases
    for phase in PHASES:
        if phase <= resume_after:
            continue

        t0 = time.time()
        try:
            if phase == 1:
                state = PHASE_RUNNERS[phase](topic)
            elif phase == 4:
                state = run_phase4(state, args.minutes)
            else:
                state = PHASE_RUNNERS[phase](state)
        except Exception as exc:
            print(f"\n{'=' * 70}")
            print(f"  !! PHASE {phase} FAILED")
            print(f"{'=' * 70}")
            print(f"  Error: {exc}\n")
            import traceback
            traceback.print_exc()
            return

        elapsed = time.time() - t0
        print(f"  Phase {phase} completed in {elapsed:.1f}s")

        # Cache immediately after each phase
        save_phase_state(state, cache_path, phase)

    # Save final results
    save_results(state)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
