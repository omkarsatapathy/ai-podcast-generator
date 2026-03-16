# Installation Guide

## Prerequisites

- Python 3.11 or higher
- pip package manager
- API keys (OpenAI, Google Search)

## Quick Start

### 1. Clone and Setup

```bash
cd Podcast_creator
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -e .
```

This installs all required packages:
- LangChain & LangGraph (agent framework)
- OpenAI (LLM)
- Google API Client (search)
- Trafilatura (content extraction)
- FastAPI (API server)
- And more...

### 3. Configure Environment Variables

Create/update `.env` file:

```env
# Required for Query Producer
OPENAI_API_KEY=sk-proj-...
GOOGLE_SEARCH_API_KEY=AIzaSy...
GOOGLE_SEARCH_ENGINE_ID=63e2eae...

# Optional (for future agents)
GEMINI_API_KEY=AIzaSy...
```

### 4. Get API Keys

#### OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up / Log in
3. Go to API Keys section
4. Create new secret key
5. Copy to `.env` as `OPENAI_API_KEY`

#### Google Search API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select existing)
3. Enable **Custom Search API**:
   - Navigate to "APIs & Services" → "Library"
   - Search for "Custom Search API"
   - Click "Enable"
4. Create credentials:
   - Go to "Credentials"
   - Click "Create Credentials" → "API Key"
   - Copy the API key to `.env` as `GOOGLE_SEARCH_API_KEY`
5. Set up Programmable Search Engine:
   - Go to [Programmable Search Engine](https://programmablesearchengine.google.com/)
   - Click "Add" to create new search engine
   - Configure:
     - Search the entire web
     - Name: "Podcast Research"
   - Copy the Search Engine ID (cx parameter)
   - Add to `.env` as `GOOGLE_SEARCH_ENGINE_ID`

**Note**: Google Custom Search has a free tier of 100 queries/day.

### 5. Test Installation

```bash
python examples/test_query_producer.py
```

Expected output:
```
✅ All required API keys found

============================================================
Testing topic: SpaceX Starship launch 2024
============================================================

📌 Topic: SpaceX Starship launch 2024
🏷️  Freshness: recent
📅 Timestamp: 2024-03-16T10:30:00

🔍 Generated 10 Queries:

 1. SpaceX Starship IFT-4 launch details
    📅 Date filter: m1

 2. Starship orbital test flight timeline
    📅 Date filter: m1
...
```

## Verify Installation

### Check Python Version

```bash
python --version
# Should be 3.11 or higher
```

### Check Installed Packages

```bash
pip list | grep -E "langchain|openai|google-api"
```

Expected:
```
langchain                 0.1.x
langchain-core            0.1.x
langchain-openai          0.0.x
langgraph                 0.0.x
openai                    1.x.x
google-api-python-client  2.x.x
```

### Test API Keys

```bash
python -c "
import os
from dotenv import load_dotenv
load_dotenv()

keys = ['OPENAI_API_KEY', 'GOOGLE_SEARCH_API_KEY', 'GOOGLE_SEARCH_ENGINE_ID']
for key in keys:
    value = os.getenv(key)
    if value:
        print(f'✅ {key}: {value[:20]}...')
    else:
        print(f'❌ {key}: NOT SET')
"
```

## Troubleshooting

### ModuleNotFoundError

```
ModuleNotFoundError: No module named 'langchain'
```

**Solution**: Make sure you ran `pip install -e .` from the project root.

### API Key Not Found

```
ValueError: GOOGLE_SEARCH_API_KEY must be set
```

**Solution**:
1. Verify `.env` file exists in project root
2. Check keys are set correctly (no spaces, quotes)
3. Restart your Python session after editing `.env`

### Import Errors

```
ImportError: cannot import name 'ChatOpenAI'
```

**Solution**: Update dependencies:
```bash
pip install --upgrade langchain langchain-openai
```

### Google API Quota Exceeded

```
HttpError 429: Resource has been exhausted
```

**Solution**:
- Free tier: 100 queries/day
- Wait 24 hours or upgrade to paid plan
- Reduce `num_results` in search queries

## Next Steps

1. ✅ Install dependencies
2. ✅ Configure API keys
3. ✅ Test Query Producer
4. 📝 Read [Query Producer Guide](docs/query_producer_guide.md)
5. 🚀 Start building your podcast!

## Optional: Development Setup

For development with auto-reload:

```bash
pip install -e ".[dev]"  # If dev dependencies are defined
```

Run tests:
```bash
pytest tests/
```

Start API server:
```bash
uvicorn main:app --reload
```

## Support

- 📖 [Design Document](design_doc.pdf)
- 🔍 [Query Producer Guide](docs/query_producer_guide.md)
- 📊 [Flow Diagrams](query_producer_upgraded_flow.svg)
