#!/usr/bin/env python3
"""Pre-flight check: verify all imports, API keys, and graph compilation before running."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=False)

import os

errors = []

# Check API keys
for key in ["OPENAI_API_KEY", "GEMINI_API_KEY"]:
    val = os.environ.get(key, "")
    if val:
        print(f"  OK  {key} is set")
    else:
        errors.append(f"{key} not found in environment")
        print(f"  FAIL {key} missing")

# Phase 1
try:
    from src.agents.phase1.query_producer import classify_freshness, web_search, web_fetch
    from src.agents.phase1.web_scraper import scrape_all_pages
    from src.agents.phase1.dedup_relevance_scorer import process as dedup_process
    print("  OK  Phase 1 imports")
except Exception as e:
    errors.append(f"Phase 1 import: {e}")
    print(f"  FAIL Phase 1 imports: {e}")

# Phase 2
try:
    from src.agents.phase2.chapter_planner import process as chapter_process
    from src.agents.phase2.character_designer import design_characters
    print("  OK  Phase 2 imports")
except Exception as e:
    errors.append(f"Phase 2 import: {e}")
    print(f"  FAIL Phase 2 imports: {e}")

# Phase 3
try:
    from src.agents.phase3.dialogue_engine import generate_chapter_dialogue, expand_expert_utterances
    from src.agents.phase3.naturalness_injector import inject_naturalness
    from src.agents.phase3.fact_checker import check_facts
    from src.agents.phase3.qa_reviewer import review_chapter
    from src.agents.phase3.ssml_annotator import annotate_chapter
    print("  OK  Phase 3 imports")
except Exception as e:
    errors.append(f"Phase 3 import: {e}")
    print(f"  FAIL Phase 3 imports: {e}")

# Graph compilation
try:
    from src.pipeline.phases import create_phase1_graph, create_phase2_graph, create_phase3_graph
    create_phase1_graph()
    print("  OK  Phase 1 graph compiled")
    create_phase2_graph()
    print("  OK  Phase 2 graph compiled")
    create_phase3_graph()
    print("  OK  Phase 3 graph compiled")
except Exception as e:
    errors.append(f"Graph compilation: {e}")
    print(f"  FAIL Graph compilation: {e}")

# Cache logic
try:
    from test_phase3_cached import save_cache, load_cache, get_cache_path
    print("  OK  Cache functions")
except Exception as e:
    errors.append(f"Cache import: {e}")
    print(f"  FAIL Cache import: {e}")

print()
if errors:
    print(f"PREFLIGHT FAILED - {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED - safe to run")
    sys.exit(0)
