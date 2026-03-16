"""Test individual components of Query Producer."""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()


def test_google_search():
    """Test Google Search API."""
    print("\n" + "="*60)
    print("TEST 1: Google Search API")
    print("="*60)

    try:
        from src.tools.web_tools import GoogleSearchTool

        tool = GoogleSearchTool()
        print("✅ GoogleSearchTool initialized")

        results = tool.search("Python programming", num_results=3)
        print(f"✅ Search returned {len(results)} results")

        if results:
            print("\nFirst result:")
            print(f"  Title: {results[0]['title']}")
            print(f"  Link: {results[0]['link']}")
            print(f"  Snippet: {results[0]['snippet'][:100]}...")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_web_fetch():
    """Test web content extraction."""
    print("\n" + "="*60)
    print("TEST 2: Web Fetch (Trafilatura)")
    print("="*60)

    try:
        from src.tools.web_tools import WebFetchTool

        tool = WebFetchTool()
        print("✅ WebFetchTool initialized")

        result = tool.fetch("https://en.wikipedia.org/wiki/Python_(programming_language)")
        print(f"✅ Fetch completed")

        if result["success"]:
            print(f"  Title: {result['title']}")
            print(f"  Content length: {len(result['text'])} chars")
            print(f"  Preview: {result['text'][:100]}...")
        else:
            print(f"⚠️  Fetch failed: {result.get('error')}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_openai_llm():
    """Test OpenAI LLM."""
    print("\n" + "="*60)
    print("TEST 3: OpenAI LLM (GPT-4o-mini)")
    print("="*60)

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        print("✅ ChatOpenAI initialized")

        response = llm.invoke([HumanMessage(content="Say 'Hello, Claude Code!' in one sentence.")])
        print(f"✅ LLM response: {response.content}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_freshness_classifier():
    """Test freshness classification."""
    print("\n" + "="*60)
    print("TEST 4: Freshness Classifier")
    print("="*60)

    try:
        from src.agents.phase1.query_producer import classify_freshness

        # Test recent topic
        result = classify_freshness.invoke({"topic": "OpenAI GPT-5 release 2024"})
        print(f"✅ Recent topic result: {result[:100]}...")

        # Test evergreen topic
        result2 = classify_freshness.invoke({"topic": "How photosynthesis works"})
        print(f"✅ Evergreen topic result: {result2[:100]}...")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all component tests."""
    print("\n" + "🧪 COMPONENT TESTS FOR QUERY PRODUCER" + "\n")

    # Check API keys
    print("Checking API keys...")
    keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GOOGLE_SEARCH_API_KEY": os.getenv("GOOGLE_SEARCH_API_KEY"),
        "GOOGLE_SEARCH_ENGINE_ID": os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    }

    for key, value in keys.items():
        if value and value != "your-openai-api-key-here":
            print(f"  ✅ {key}")
        else:
            print(f"  ❌ {key} NOT SET")

    # Run tests
    tests = [
        ("Google Search", test_google_search),
        ("Web Fetch", test_web_fetch),
        ("OpenAI LLM", test_openai_llm),
        ("Freshness Classifier", test_freshness_classifier),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except KeyboardInterrupt:
            print("\n\n⚠️  Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error in {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nPassed: {passed}/{total}")


if __name__ == "__main__":
    main()
