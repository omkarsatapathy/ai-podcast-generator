# Query Producer Agent - User Guide

## Overview

The Query Producer is a ReAct-style LLM agent that generates 10 diverse, context-aware search queries for any given topic. It uses Google Search API for web research and OpenAI GPT-4o-mini for intelligent query generation.

## Architecture

### Flow Diagram

```
Topic Input
    ↓
Step 1: Freshness Classifier
    ↓
   / \
  /   \
Recent  Evergreen
  ↓      ↓
Seed    Direct
Search  LLM
  ↓      ↓
Extract  |
Context  |
  ↓      |
Informed |
Queries  |
  ↓      ↓
   \ | /
    ↓
Date-Aware Tagging
    ↓
10 Search Queries
```

### Components

1. **Freshness Classifier**
   - Determines if topic is recent/breaking news or evergreen
   - Uses keyword heuristics + LLM reasoning
   - Output: `"recent"` or `"evergreen"`

2. **Recent Path** (for breaking news)
   - **Seed Search**: Web search the raw topic
   - **Extract Context**: Fetch top 3-5 results
   - **Informed Queries**: LLM generates queries using context

3. **Evergreen Path** (for timeless topics)
   - **Direct LLM**: Generate queries without seed search
   - Faster, no external API calls needed

4. **Date Tagging**
   - Adds date filters for recent topics (`m1` = past month)
   - No filters for evergreen topics

## Tools Available

The agent has access to these tools via LangChain:

- `web_search(query, num_results)` - Google Custom Search API
- `web_fetch(url)` - Extract content using Trafilatura
- `get_today_date()` - Returns current date
- `classify_freshness(topic)` - Classify topic freshness

## Technology Stack

- **LangGraph**: Workflow orchestration
- **LangChain**: Agent framework & tool calling
- **OpenAI GPT-4o-mini**: LLM for reasoning
- **Google Custom Search API**: Web search
- **Trafilatura**: Content extraction

## Setup

### 1. Install Dependencies

```bash
pip install -e .
```

### 2. Configure API Keys

Add to `.env`:

```env
OPENAI_API_KEY=sk-...
GOOGLE_SEARCH_API_KEY=AIza...
GOOGLE_SEARCH_ENGINE_ID=your-cx-id
```

### 3. Get Google Search API Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable "Custom Search API"
4. Create credentials → API Key
5. Go to [Programmable Search Engine](https://programmablesearchengine.google.com/)
6. Create a new search engine
7. Copy the Search Engine ID (cx parameter)

## Usage

### Basic Usage

```python
from src.agents.phase1.query_producer import QueryProducerAgent

agent = QueryProducerAgent()
result = agent.run("SpaceX Starship launch 2024")

print(f"Freshness: {result.freshness}")
for query in result.queries:
    print(f"- {query.query}")
```

### Output Structure

```python
QueryProducerOutput(
    topic="SpaceX Starship launch 2024",
    freshness="recent",
    queries=[
        SearchQuery(
            query="SpaceX Starship IFT-4 launch date",
            date_filter="m1"
        ),
        # ... 9 more queries
    ],
    seed_context="...",  # Only for recent topics
    timestamp="2024-03-16T10:30:00"
)
```

### Test Script

```bash
python examples/test_query_producer.py
```

## Examples

### Recent Topic (Breaking News)

**Input**: `"OpenAI GPT-5 release 2024"`

**Flow**: Recent → Seed Search → Extract Context → Informed Queries

**Output**:
1. OpenAI GPT-5 official announcement date
2. GPT-5 vs GPT-4 performance benchmarks
3. OpenAI Sam Altman GPT-5 interview
4. GPT-5 pricing and API access
5. ... (6 more)

All queries tagged with `date_filter="m1"`

### Evergreen Topic

**Input**: `"How photosynthesis works"`

**Flow**: Evergreen → Direct LLM → Date Tagging (skip)

**Output**:
1. Photosynthesis light-dependent reactions mechanism
2. Calvin cycle step-by-step explanation
3. Chloroplast structure and function
4. Photosynthesis efficiency factors
5. ... (6 more)

No date filters applied.

## Query Quality Principles

The agent generates queries that:

1. **Cover multiple angles** - who, what, why, when, how, impact
2. **Are specific** - include names, dates, technical terms
3. **Target authoritative sources** - research, expert opinions, official docs
4. **Avoid redundancy** - each query explores a different facet
5. **Are optimized for search engines** - clear, concise, keyword-rich

## Customization

### Adjust Number of Results

```python
# In query_producer.py, modify seed_search_node
search_results = web_search.invoke({"query": topic, "num_results": 10})
```

### Change Date Filter Logic

```python
# In date_tagging_node
if freshness == "recent":
    query_dict["date_filter"] = "w1"  # Past week instead of month
```

### Add Custom Tools

```python
@tool
def search_arxiv(query: str) -> str:
    """Search academic papers on arXiv."""
    # Implementation
    pass

# Add to workflow
```

## Troubleshooting

### API Key Errors

```
ValueError: GOOGLE_SEARCH_API_KEY must be set in .env
```

**Solution**: Verify `.env` file has all required keys.

### No Search Results

```
No results found.
```

**Possible causes**:
- Invalid search engine ID
- API quota exceeded (100 queries/day free tier)
- Network issues

### LLM Errors

```
openai.AuthenticationError: Invalid API key
```

**Solution**: Check `OPENAI_API_KEY` is valid and has credits.

## Performance

- **Evergreen path**: ~5-10 seconds (1 LLM call)
- **Recent path**: ~20-30 seconds (seed search + content fetch + LLM)

## Limitations

- Google Custom Search API: 100 free queries/day
- Content extraction may fail for paywalled sites
- LLM hallucination risk (mitigated by grounding in seed context)

## Next Steps

After query generation, queries are passed to:
1. **Web Scraper Agent** - Executes all 10 queries
2. **Dedup + Relevance Scorer** - Ranks and filters results

See: [Web Scraper Guide](./web_scraper_guide.md)
