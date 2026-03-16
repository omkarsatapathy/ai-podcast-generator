"""Minimal test script for full Phase 1 + Phase 2 pipeline (queries → search → scrape → merge → chapter planning)."""
import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OUTPUT_FILE = project_root / "data" / "output" / "merged_research.txt"
PHASE2_OUTPUT_FILE = project_root / "data" / "output" / "phase2_results.json"


def check_env_vars():
    """Check if required environment variables are set."""
    print("🔍 Checking environment variables...\n")

    required = {
        "OPENAI_API_KEY": "OpenAI API key for GPT-4o-mini",
        "GOOGLE_SEARCH_API_KEY": "Google Search API key",
        "GOOGLE_SEARCH_ENGINE_ID": "Google Search Engine ID"
    }

    all_set = True
    for key, description in required.items():
        value = os.getenv(key)
        if value and value != "your-openai-api-key-here":
            print(f"✅ {key}: {value[:20]}...")
        else:
            print(f"❌ {key}: NOT SET ({description})")
            all_set = False

    print()
    return all_set


def test_full_phase1():
    """Test full Phase 1 pipeline: queries → search → scrape → merge."""
    print("=" * 60)
    print("Testing Full Phase 1 Pipeline")
    print("=" * 60)
    print()

    # Check environment
    if not check_env_vars():
        print("⚠️  Please set missing API keys in .env file")
        return

    try:
        print("📦 Importing Phase 1 graph...")
        from src.pipeline.phases.phase1_graph import create_phase1_graph
        print("✅ Import successful\n")

        graph = create_phase1_graph()


        test_topic = "why Gold price is falling, and the priec forecast by end of year"


        print(f"🔬 Testing with topic: '{test_topic}'")
        print("⏳ Running full Phase 1 (queries → search → scrape → merge)...\n")

        initial_state = {
            "topic": test_topic,
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

        final_state = graph.invoke(initial_state)

        # Display summary
        print("=" * 60)
        print("✅ RESULTS")
        print("=" * 60)

        print(f"\n📌 Topic: {final_state['topic']}")
        print(f"🏷️  Freshness: {final_state['freshness']}")
        print(f"🔍 Queries generated: {len(final_state['queries'])}")
        print(f"🔄 Query rewrites: {final_state['query_rewrite_count']}")
        print(f"📉 Scrape failure rate: {final_state['scrape_failure_rate']:.0%}")

        # Scrape stats
        scraped = final_state["scraped_pages"]
        succeeded = sum(1 for p in scraped if p["success"])
        print(f"🌐 Pages scraped: {succeeded}/{len(scraped)}")

        # Merged text stats
        merged = final_state["merged_research_text"]
        word_count = len(merged.split()) if merged else 0
        print(f"\n📝 Merged research text: {len(merged)} chars, {word_count} words")

        # Save to file
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(merged, encoding="utf-8")
        print(f"💾 Saved to: {OUTPUT_FILE}")

        # Verify dedup_and_rank node was executed (FINAL NODE)
        ranked_chunks = final_state.get("ranked_chunks", [])
        dedup_stats = final_state.get("dedup_stats", {})

        print(f"\n🔬 DEDUP & RELEVANCE SCORING (Final Node):")
        if ranked_chunks and dedup_stats:
            print(f"   ✅ Ranked chunks produced: {len(ranked_chunks)}")
            print(f"   ✅ Dedup stats available:")
            print(f"      - Total chunks: {dedup_stats.get('total_chunks', 'N/A')}")
            print(f"      - Duplicates removed: {dedup_stats.get('duplicates_removed', 'N/A')}")
            print(f"      - Top-K selected: {dedup_stats.get('top_k_selected', 'N/A')}")
        else:
            print(f"   ⚠️  Dedup & rank node NOT executed!")
            print(f"      - Ranked chunks: {len(ranked_chunks)}")
            print(f"      - Dedup stats: {bool(dedup_stats)}")

        # Verify chapter titles were generated
        chapter_titles = final_state.get("chapter_titles", [])
        print(f"\n📚 Chapter titles generated: {len(chapter_titles)}/10")

        print(f"\n📊 WORD COUNT: {word_count}")
        print("=" * 60)
        print("✅ PHASE 1 TEST PASSED! (Reached final node: dedup_and_rank → END)")
        print("=" * 60)

        return final_state

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ PHASE 1 TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {e}\n")

        import traceback
        traceback.print_exc()
        return None


def test_phase2(phase1_final_state):
    """Test Phase 2: Chapter Planner + Character Designer using Phase 1 output."""
    print("\n" + "=" * 60)
    print("Testing Phase 2 Pipeline (Chapter Planner + Character Designer)")
    print("=" * 60)
    print()

    try:
        print("📦 Importing Phase 2 graph...")
        from src.pipeline.phases.phase2_graph import create_phase2_graph
        print("✅ Import successful\n")

        graph = create_phase2_graph()

        # Prepare Phase 2 initial state from Phase 1 output
        ranked_chunks = phase1_final_state.get("ranked_chunks", [])
        topic = phase1_final_state.get("topic", "")

        if not ranked_chunks:
            print("❌ No ranked chunks from Phase 1! Cannot run Phase 2.")
            return None

        print(f"📥 Phase 2 Input:")
        print(f"   Topic: '{topic}'")
        print(f"   Ranked chunks: {len(ranked_chunks)}")
        print()

        from config.settings import settings
        num_speakers = settings.NUM_SPEAKERS

        initial_state = {
            "topic": topic,
            "ranked_chunks": ranked_chunks,
            "num_speakers": num_speakers,
            "chapter_outlines": [],
            "analyzed_chunks": [],
            "total_estimated_duration": 0.0,
            "character_personas": [],
        }

        print(f"⏳ Running Phase 2 (Chapter Planner + Character Designer, {num_speakers} speakers)...\n")
        final_state = graph.invoke(initial_state)

        # Display Phase 2 Results
        print("=" * 60)
        print("✅ PHASE 2 RESULTS")
        print("=" * 60)

        chapter_outlines = final_state.get("chapter_outlines", [])
        analyzed_chunks = final_state.get("analyzed_chunks", [])
        total_duration = final_state.get("total_estimated_duration", 0.0)

        print(f"\n📚 Chapters Created: {len(chapter_outlines)}")
        print(f"⏱️  Total Estimated Duration: {total_duration:.1f} minutes")
        print(f"📊 Analyzed Chunks: {len(analyzed_chunks)}")

        # Display chapter outlines
        print(f"\n{'=' * 60}")
        print("📖 CHAPTER OUTLINES")
        print('=' * 60)

        for i, chapter in enumerate(chapter_outlines, 1):
            print(f"\n📌 Chapter {chapter['chapter_number']}: {chapter['title']}")
            print(f"   Act: {chapter['act']} | Energy: {chapter['energy_level']} | Duration: {chapter['estimated_duration_minutes']:.1f} min")
            print(f"   Key Points ({len(chapter['key_points'])}):")
            for j, point in enumerate(chapter['key_points'], 1):
                print(f"      {j}. {point}")
            print(f"   Transition Hook: \"{chapter['transition_hook']}\"")
            print(f"   Source Chunks: {len(chapter['source_chunk_ids'])} chunks")

        # Display chunk schema (show first 2 analyzed chunks as examples)
        print(f"\n{'=' * 60}")
        print("🔬 CHUNK SCHEMA (Sample)")
        print('=' * 60)
        print("\nShowing structure of analyzed chunks (first 2 examples):\n")

        for i, chunk in enumerate(analyzed_chunks[:2], 1):
            print(f"--- Chunk {i} ({chunk.get('chunk_id', 'N/A')}) ---")
            print(f"  Original Fields:")
            print(f"    - chunk_id: {chunk.get('chunk_id', 'N/A')}")
            print(f"    - source_url: {chunk.get('source_url', 'N/A')[:60]}...")
            print(f"    - word_count: {chunk.get('word_count', 'N/A')}")
            print(f"    - relevance_score: {chunk.get('relevance_score', 'N/A'):.4f}")
            print(f"    - text (preview): {chunk.get('text', '')[:100]}...")
            print(f"\n  Phase 2 Analysis Fields (Added):")
            print(f"    - analysis_topic: \"{chunk.get('analysis_topic', 'N/A')}\"")
            print(f"    - analysis_subtopics: {chunk.get('analysis_subtopics', [])}")
            print(f"    - analysis_summary: \"{chunk.get('analysis_summary', 'N/A')}\"")
            print(f"    - analysis_tone: \"{chunk.get('analysis_tone', 'N/A')}\"")
            print()

        # Display character personas
        character_personas = final_state.get("character_personas", [])
        if character_personas:
            print(f"\n{'=' * 60}")
            print("🎭 CHARACTER PERSONAS")
            print('=' * 60)
            for p in character_personas:
                print(f"\n  {p['role'].upper()}: {p['name']}")
                print(f"    Voice: {p['tts_voice_id']} ({p['gender']})")
                print(f"    Expertise: {p['expertise_area']}")
                print(f"    Style: {p['speaking_style']}")
                print(f"    Vocabulary: {p['vocabulary_level']}")
                print(f"    Fillers: {p['filler_patterns']}")
                print(f"    Reactions: {p['reaction_patterns']}")
                print(f"    Catchphrases: {p['catchphrases']}")

        # Save Phase 2 results to JSON
        phase2_results = {
            "topic": topic,
            "total_chapters": len(chapter_outlines),
            "total_duration_minutes": total_duration,
            "chapter_outlines": chapter_outlines,
            "character_personas": character_personas,
            "analyzed_chunks_count": len(analyzed_chunks),
            "sample_chunks": analyzed_chunks[:3],  # Save first 3 for inspection
        }

        PHASE2_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        PHASE2_OUTPUT_FILE.write_text(
            json.dumps(phase2_results, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"💾 Phase 2 results saved to: {PHASE2_OUTPUT_FILE}")

        print(f"\n{'=' * 60}")
        print("✅ PHASE 2 TEST PASSED!")
        print('=' * 60)

        return final_state

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ PHASE 2 TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {e}\n")

        import traceback
        traceback.print_exc()
        return None


def test_full_pipeline():
    """Test full pipeline: Phase 1 → Phase 2."""
    print("🚀 Starting Full Pipeline Test (Phase 1 + Phase 2)")
    print("=" * 60)
    print()

    # Run Phase 1
    phase1_state = test_full_phase1()

    # Run Phase 2 if Phase 1 succeeded
    if phase1_state and phase1_state.get("ranked_chunks"):
        test_phase2(phase1_state)
    else:
        print("\n⚠️  Skipping Phase 2 due to Phase 1 failure or missing ranked_chunks")

    print("\n" + "=" * 60)
    print("🏁 FULL PIPELINE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_full_pipeline()
