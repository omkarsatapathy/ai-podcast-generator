"""Minimal test script for full Phase 1 pipeline (queries → search → scrape → merge)."""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OUTPUT_FILE = project_root / "data" / "output" / "merged_research.txt"


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

        test_topic = "the AGI in future"
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

        print(f"\n📊 WORD COUNT: {word_count}")
        print("=" * 60)
        print("✅ TEST PASSED!")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {e}\n")

        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_full_phase1()
