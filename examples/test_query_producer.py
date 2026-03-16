"""Test script for Query Producer Agent."""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.agents.phase1.query_producer import QueryProducerAgent

# Load environment variables
load_dotenv()

# Verify API keys are set
required_keys = ["OPENAI_API_KEY", "GOOGLE_SEARCH_API_KEY", "GOOGLE_SEARCH_ENGINE_ID"]
for key in required_keys:
    if not os.getenv(key):
        print(f"❌ Missing {key} in .env file")
        sys.exit(1)

print("✅ All required API keys found\n")


def test_query_producer():
    """Test the Query Producer Agent."""
    agent = QueryProducerAgent()

    # Test cases
    test_topics = [
        "claude co work lunch",  # Recent topic
        # "Climate change solutions",       # Evergreen topic
    ]

    for topic in test_topics:
        print(f"\n{'='*60}")
        print(f"Testing topic: {topic}")
        print(f"{'='*60}\n")

        try:
            result = agent.run(topic)

            print(f"📌 Topic: {result.topic}")
            print(f"🏷️  Freshness: {result.freshness}")
            print(f"📅 Timestamp: {result.timestamp}")

            if result.seed_context:
                print(f"\n📖 Seed Context Length: {len(result.seed_context)} chars")

            print(f"\n🔍 Generated {len(result.queries)} Queries:\n")

            for i, query in enumerate(result.queries, 1):
                print(f"{i:2d}. {query.query}")
                if query.date_filter:
                    print(f"    📅 Date filter: {query.date_filter}")
                print()

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_query_producer()
