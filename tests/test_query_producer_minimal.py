"""Full pipeline test with per-phase caching and automatic resume.

Runs Phase 1 → Phase 2 → Phase 3, caching state after each phase.
On subsequent runs, automatically resumes from the latest cached phase.

Cache structure:
    data/cache/<topic_slug>_<hash>/
        manifest.json          # Quick summary of what's cached
        phase1_state.json      # State after Phase 1
        phase2_state.json      # State after Phase 2
        phase3_state.json      # State after Phase 3

Usage:
    # Run everything (caches after each phase):
    python tests/test_query_producer_minimal.py

    # Custom topic:
    python tests/test_query_producer_minimal.py "AI regulation in Europe"

    # Force re-run from scratch (ignore all cache):
    python tests/test_query_producer_minimal.py --fresh

    # Force re-run from a specific phase (e.g. re-run Phase 2 and 3):
    python tests/test_query_producer_minimal.py --from-phase 2
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
PHASES = [1, 2, 3]


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

    manifest[f"phase{phase}"] = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "state_keys": list(state.keys()),
        "ranked_chunks": len(state.get("ranked_chunks", [])),
        "chapters": len(state.get("chapter_outlines", [])),
        "personas": len(state.get("character_personas", [])),
        "dialogues": len(state.get("chapter_dialogues", [])),
    }
    manifest["topic"] = state.get("topic", "")
    manifest["latest_phase"] = phase
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
# Env check
# ---------------------------------------------------------------------------

def check_env_vars() -> bool:
    """Verify required API keys are present."""
    print("Checking environment variables...\n")
    required = {
        "OPENAI_API_KEY": "OpenAI API key",
        "GOOGLE_SEARCH_API_KEY": "Google Search API key",
        "GOOGLE_SEARCH_ENGINE_ID": "Google Search Engine ID",
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

    # Print summary
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

    # Print summary
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

    # Print summary
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


# ---------------------------------------------------------------------------
# Save final results
# ---------------------------------------------------------------------------

def save_results(state: dict) -> None:
    """Write Phase 3 dialogue output for inspection."""
    out_dir = settings.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    result_file = out_dir / "phase3_results.json"
    dialogues = state.get("chapter_dialogues", [])

    with open(result_file, "w") as f:
        json.dump(dialogues, f, indent=2, default=str)

    print(f"  Results saved: {result_file}")


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
        description="Run podcast pipeline with per-phase caching.",
    )
    parser.add_argument("topic", nargs="*",
                        default=["Why Gold price is falling, and the price forecast by end of year"])
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore all cache and re-run from Phase 1")
    parser.add_argument("--from-phase", type=int, default=None, dest="from_phase",
                        help="Force re-run starting at this phase (1, 2, or 3)")
    args = parser.parse_args()

    topic = " ".join(args.topic)
    cache_path = _cache_path_for_topic(topic)

    print("\n" + "=" * 70)
    print("  PODCAST PIPELINE (with per-phase caching)")
    print("=" * 70)
    print(f"  Topic:       {topic}")
    print(f"  Cache dir:   {cache_path}")
    print(f"  TTS:         {settings.TTS_PROVIDER}")
    print(f"  Fresh:       {args.fresh}")
    print(f"  From-phase:  {args.from_phase or 'auto'}")
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