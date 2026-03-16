# Query Producer Tests

## Quick Start

### 1. Install dependencies first
```bash
pip install -e .
```

### 2. Set up .env file
Make sure your `.env` has:
```env
OPENAI_API_KEY=sk-proj-...
GOOGLE_SEARCH_API_KEY=AIza...
GOOGLE_SEARCH_ENGINE_ID=...
```

### 3. Run tests

#### Test individual components (recommended first)
```bash
python tests/test_components.py
```

This tests:
- ✅ Google Search API connectivity
- ✅ Web content extraction (Trafilatura)
- ✅ OpenAI LLM (GPT-4o-mini)
- ✅ Freshness classifier tool

#### Test full Query Producer agent
```bash
python tests/test_query_producer_minimal.py
```

This runs the complete workflow and generates 10 queries.

## Expected Output

### Component Test Success
```
🧪 COMPONENT TESTS FOR QUERY PRODUCER

Checking API keys...
  ✅ OPENAI_API_KEY
  ✅ GOOGLE_SEARCH_API_KEY
  ✅ GOOGLE_SEARCH_ENGINE_ID

============================================================
TEST 1: Google Search API
============================================================
✅ GoogleSearchTool initialized
✅ Search returned 3 results
...

============================================================
SUMMARY
============================================================
✅ PASS - Google Search
✅ PASS - Web Fetch
✅ PASS - OpenAI LLM
✅ PASS - Freshness Classifier

Passed: 4/4
```

### Full Agent Test Success
```
============================================================
Testing Query Producer Agent
============================================================

✅ Import successful
✅ Agent created

🔬 Testing with topic: 'Climate change solutions'
⏳ Running agent (this may take 10-30 seconds)...

============================================================
✅ RESULTS
============================================================

📌 Topic: Climate change solutions
🏷️  Freshness: evergreen
📅 Timestamp: 2024-03-16T...

🔍 Generated 10 Queries:

 1. Carbon capture and storage technology effectiveness
 2. Renewable energy transition challenges and solutions
...

============================================================
✅ TEST PASSED!
============================================================
```

## Troubleshooting

### ModuleNotFoundError
```
ModuleNotFoundError: No module named 'langchain'
```
**Fix**: Run `pip install -e .` from project root

### API Key Errors
```
❌ OPENAI_API_KEY NOT SET
```
**Fix**: Add valid API key to `.env` file

### Google Search Quota
```
HttpError 429: Resource exhausted
```
**Fix**: Wait 24 hours (free tier = 100 queries/day)

## Test Files

- `test_components.py` - Individual component tests
- `test_query_producer_minimal.py` - Full agent test
- `unit/` - Unit tests (coming soon)
- `integration/` - Integration tests (coming soon)
