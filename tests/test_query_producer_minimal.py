"""Minimal test script for Query Producer Agent."""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


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


def test_query_producer():
    """Test Query Producer with a simple topic."""
    print("="*60)
    print("Testing Query Producer Agent")
    print("="*60)
    print()

    # Check environment
    if not check_env_vars():
        print("⚠️  Please set missing API keys in .env file")
        return

    try:
        # Import agent
        print("📦 Importing Query Producer Agent...")
        from src.agents.phase1.query_producer import QueryProducerAgent
        print("✅ Import successful\n")

        # Create agent
        print("🤖 Creating agent instance...")
        agent = QueryProducerAgent()
        print("✅ Agent created\n")

        # Test with simple topic
        test_topic = "the AGI in future"  # Recent topic
        print(f"🔬 Testing with topic: '{test_topic}'")
        print("⏳ Running agent (this may take 10-30 seconds)...\n")

        result = agent.run(test_topic)

        # Display results
        print("="*60)
        print("✅ RESULTS")
        print("="*60)
        print(f"\n📌 Topic: {result.topic}")
        print(f"🏷️  Freshness: {result.freshness}")
        print(f"📅 Timestamp: {result.timestamp}")

        if result.seed_context:
            print(f"📖 Seed Context: {len(result.seed_context)} characters")

        print(f"\n🔍 Generated {len(result.queries)} Queries:\n")

        for i, query in enumerate(result.queries, 1):
            print(f"{i:2d}. {query.query}")
            if query.date_filter:
                print(f"    📅 Date filter: {query.date_filter}")
            if query.results:
                print(f"    📊 Results: {len(query.results)} search results found")

        print("\n" + "="*60)
        print("✅ TEST PASSED!")
        print("="*60)

    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED!")
        print("="*60)
        print(f"\nError: {e}\n")

        import traceback
        print("Traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    test_query_producer()
