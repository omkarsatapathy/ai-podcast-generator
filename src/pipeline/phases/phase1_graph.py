"""Phase 1: Research & Ingestion Subgraph.

This module contains the graph structure, state definition, and node functions
for the Phase 1 (Query Producer) workflow. The workflow:
1. Classifies topics by freshness (recent vs evergreen)
2. For recent topics: performs seed search and extracts context
3. Generates 10 diverse search queries
4. Adds date filters to queries (for recent topics)
5. Executes all searches and returns results

This subgraph is:
- Independent: Can be imported and run standalone by query_producer.py
- Composable: Can be imported as a node in the main orchestrator
"""

from typing import List, Dict, Any, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from config.settings import settings
from src.tools.web_tools import get_current_date


# ==================== STATE ====================

class Phase1State(TypedDict):
    """State for Phase 1: Research & Ingestion."""

    topic: str
    freshness: str  # 'recent' or 'evergreen'
    seed_results: List[Dict[str, Any]]
    seed_context: str
    queries: List[Dict[str, Any]]  # Contains query text, date_filter, and results
    current_date: str
    messages: List
    scraped_pages: List[Dict[str, Any]]  # Individual scrape results
    merged_research_text: str  # All scraped text merged into one string
    scrape_failure_rate: float  # Fraction of URLs that failed
    query_rewrite_count: int  # Times queries have been rewritten (max 3)
    chapter_titles: List[str]  # 10 suggested chapter titles for stage 2

    # Dedup + Relevance Scorer outputs (pure in-memory)
    ranked_chunks: List[Dict[str, Any]]  # Top-K chunks for Chapter Planner
    dedup_stats: Dict[str, Any]  # Processing statistics


# ==================== NODES ====================

def initialize_node(state: Phase1State) -> Phase1State:
    """Initialize the agent state."""
    state["current_date"] = get_current_date()
    state["seed_results"] = []
    state["seed_context"] = ""
    state["queries"] = []
    state["messages"] = []
    state["scraped_pages"] = []
    state["merged_research_text"] = ""
    state["scrape_failure_rate"] = 0.0
    state["query_rewrite_count"] = 0
    state["chapter_titles"] = []
    return state


def classify_freshness_node(state: Phase1State) -> Phase1State:
    """Classify topic freshness."""
    from src.agents.phase1.query_producer import classify_freshness

    topic = state["topic"]

    # Use the classify_freshness tool
    result = classify_freshness.invoke({"topic": topic})

    # Parse result
    if "recent" in result.lower():
        state["freshness"] = "recent"
    else:
        state["freshness"] = "evergreen"

    print(f"\n🏷️  FRESHNESS CLASSIFICATION: {state['freshness'].upper()}")
    print(f"   Reason: {result}\n")

    return state


def seed_search_node(state: Phase1State) -> Phase1State:
    """Execute seed search for recent topics."""
    from src.agents.phase1.query_producer import web_search

    topic = state["topic"]

    print(f"🔍 SEED SEARCH: Searching for '{topic}'...\n")

    # Search the raw topic
    search_results = web_search.invoke({"query": topic, "num_results": 5})

    state["messages"].append({
        "role": "system",
        "content": f"Seed search results:\n{search_results}"
    })

    return state


def extract_context_node(state: Phase1State) -> Phase1State:
    """Extract context from top seed results."""
    from src.agents.phase1.query_producer import web_fetch

    search_results = state["messages"][-1]["content"]

    # Parse URLs from search results
    urls = []
    for line in search_results.split("\n"):
        if "URL:" in line:
            url = line.split("URL:")[1].strip()
            urls.append(url)

    print(f"📄 FETCHING CONTENT from top {min(3, len(urls))} URLs:")
    for i, url in enumerate(urls[:3], 1):
        print(f"   {i}. {url}")
    print()

    # Fetch top 3 URLs
    contexts = []
    for i, url in enumerate(urls[:3], 1):
        print(f"   ⏳ Fetching {i}/{min(3, len(urls))}: {url[:80]}...")
        content = web_fetch.invoke({"url": url})
        contexts.append(content)

        # Check if fetch was successful
        if "Failed to fetch" in content:
            print(f"      ❌ Fetch failed")
        else:
            print(f"      ✅ Success ({len(content)} chars)")

    print()
    state["seed_context"] = "\n\n---\n\n".join(contexts)

    return state


def generate_queries_node(state: Phase1State) -> Phase1State:
    """Generate 15 diverse search queries and 10 chapter titles. On re-entry (retry loop), increments rewrite counter."""
    # If queries already exist, this is a retry — increment counter
    if state["queries"]:
        state["query_rewrite_count"] += 1
        print(f"🔄 QUERY REWRITE #{state['query_rewrite_count']}/{settings.MAX_QUERY_REWRITE_ATTEMPTS}")

    llm = ChatOpenAI(model=settings.QUERY_PRODUCER_MODEL, temperature=settings.QUERY_PRODUCER_TEMPERATURE)

    topic = state["topic"]
    freshness = state["freshness"]
    seed_context = state.get("seed_context", "")
    current_date = state["current_date"]

    print(f"🤖 GENERATING QUERIES AND CHAPTER TITLES using LLM (freshness: {freshness})...\n")

    if freshness == "recent":
        system_prompt = f"""You are a search query expert and podcast content planner. Your task has two parts:

Topic: {topic}
Current Date: {current_date}

Context from seed search:
{seed_context}

PART 1 - SEARCH QUERIES:
Generate {settings.QUERY_PRODUCE_PER_TOPIC} diverse, specific search queries for researching this recent topic.

Generate queries that:
1. Cover different angles (who, what, why, impact, reactions, background)
2. Include specific names, dates, or events from the context
3. Are specific and factual
4. Will return high-quality sources

PART 2 - CHAPTER TITLES:
Based on the topic and context, suggest 10 compelling chapter titles for a podcast episode. These should represent key aspects, themes, or segments that would make for engaging podcast content.

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:

QUERIES:
1. [first query]
2. [second query]
...
{settings.QUERY_PRODUCE_PER_TOPIC}. [last query]

CHAPTER_TITLES:
1. [first chapter title]
2. [second chapter title]
...
10. [tenth chapter title]"""

    else:  # evergreen
        system_prompt = f"""You are a search query expert and podcast content planner. Your task has two parts:

Topic: {topic}
Current Date: {current_date}

PART 1 - SEARCH QUERIES:
Generate {settings.QUERY_PRODUCE_PER_TOPIC} diverse, comprehensive search queries for researching this evergreen topic.

Generate queries that:
1. Cover foundational concepts, mechanisms, applications, debates
2. Include expert perspectives, research studies, case studies
3. Cover historical context and current state
4. Are specific and will return authoritative sources

PART 2 - CHAPTER TITLES:
Based on the topic, suggest 10 compelling chapter titles for a podcast episode. These should represent key aspects, themes, or segments that would make for engaging podcast content.

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:

QUERIES:
1. [first query]
2. [second query]
...
{settings.QUERY_PRODUCE_PER_TOPIC}. [last query]

CHAPTER_TITLES:
1. [first chapter title]
2. [second chapter title]
...
10. [tenth chapter title]"""

    response = llm.invoke([SystemMessage(content=system_prompt)])

    # Parse response into queries and chapter titles sections
    response_text = response.content

    # Split response into sections
    queries = []
    chapter_titles = []

    # Find QUERIES and CHAPTER_TITLES sections
    lines = response_text.split("\n")
    current_section = None

    for line in lines:
        line_stripped = line.strip()

        # Check for section headers
        if "QUERIES:" in line_stripped.upper():
            current_section = "queries"
            continue
        elif "CHAPTER_TITLES:" in line_stripped.upper() or "CHAPTER TITLES:" in line_stripped.upper():
            current_section = "chapter_titles"
            continue

        # Parse numbered items based on current section
        if current_section == "queries" and line_stripped:
            # Check if line starts with a number (1. to 15.)
            if any(line_stripped.startswith(f"{i}.") for i in range(1, settings.QUERY_PRODUCE_PER_TOPIC + 1)):
                query_text = line_stripped.split(".", 1)[1].strip().strip('"\'')
                queries.append({"query": query_text, "date_filter": None})

        elif current_section == "chapter_titles" and line_stripped:
            # Check if line starts with a number (1. to 10.)
            if any(line_stripped.startswith(f"{i}.") for i in range(1, 11)):
                title_text = line_stripped.split(".", 1)[1].strip().strip('"\'')
                chapter_titles.append(title_text)

    # Store results in state
    state["queries"] = queries[:settings.QUERY_PRODUCE_PER_TOPIC]  # Use setting value
    state["chapter_titles"] = chapter_titles[:10]  # Keep exactly 10 chapter titles

    print(f"✅ Generated {len(state['queries'])} queries")
    print(f"✅ Generated {len(state['chapter_titles'])} chapter titles")
    if state["chapter_titles"]:
        print(f"\n📚 CHAPTER TITLES:")
        for i, title in enumerate(state["chapter_titles"], 1):
            print(f"   {i}. {title}")
    print()

    return state


def date_tagging_node(state: Phase1State) -> Phase1State:
    """Add date filters to queries where relevant."""
    freshness = state["freshness"]
    queries = state["queries"]

    if freshness == "recent":
        # Add recent date filter to all queries
        for query_dict in queries:
            query_dict["date_filter"] = "m1"  # Past month
        print(f"📅 DATE FILTERS: Added 'm1' (past month) filter to all queries\n")
    else:
        print(f"📅 DATE FILTERS: No date filters needed (evergreen topic)\n")

    return state


def execute_searches_node(state: Phase1State) -> Phase1State:
    """Execute Google searches for all 10 queries."""
    queries = state["queries"]
    from src.tools.web_tools import GoogleSearchTool
    search_tool = GoogleSearchTool()

    print("\n🔍 Executing 10 Google searches...")

    for i, query_dict in enumerate(queries, 1):
        query_text = query_dict["query"]
        date_filter = query_dict.get("date_filter")

        print(f"   {i}/10: {query_text[:60]}...")

        # Execute search with date filter if provided
        results = search_tool.search(
            query_text,
            num_results=settings.SEARCH_RESULTS_PER_QUERY,
            date_restrict=date_filter
        )

        # Store results in query dict
        query_dict["results"] = results
        print(f"      → {len(results)} results, links: {[r.get('link','')[:50] for r in results[:3]]}")

    total_links = sum(len(q.get("results", [])) for q in queries)
    print(f"✅ All searches completed! Total results: {total_links}\n")

    return state


def scrape_pages_node(state: Phase1State) -> Phase1State:
    """Scrape all URLs from search results using BS4 + ThreadPoolExecutor."""
    from src.agents.phase1.web_scraper import scrape_all_pages

    # Collect all unique URLs from search results
    urls = set()
    for query_dict in state["queries"]:
        for result in query_dict.get("results", []):
            link = result.get("link", "")
            if link:
                urls.add(link)

    urls = list(urls)
    total = len(urls)
    print(f"\n🌐 SCRAPING {total} unique URLs with {settings.WEB_SCRAPER_MAX_WORKERS} workers...")

    if not urls:
        state["scraped_pages"] = []
        state["scrape_failure_rate"] = 1.0
        return state

    results = scrape_all_pages(urls)

    succeeded = sum(1 for r in results if r["success"])
    failed = total - succeeded
    failure_rate = failed / total if total > 0 else 0.0

    state["scraped_pages"] = results
    state["scrape_failure_rate"] = failure_rate

    print(f"   ✅ Succeeded: {succeeded}/{total}")
    print(f"   ❌ Failed: {failed}/{total} ({failure_rate:.0%})")

    return state


def evaluate_scrape_quality_node(state: Phase1State) -> Phase1State:
    """Log scrape quality. Routing is handled by conditional edges."""
    failure_rate = state["scrape_failure_rate"]
    rewrite_count = state["query_rewrite_count"]

    if failure_rate > settings.LINK_FAILURE_THRESHOLD and rewrite_count < settings.MAX_QUERY_REWRITE_ATTEMPTS:
        print(f"\n⚠️  FAILURE RATE {failure_rate:.0%} > {settings.LINK_FAILURE_THRESHOLD:.0%} threshold")
        print(f"   Will rewrite queries (attempt {rewrite_count + 1}/{settings.MAX_QUERY_REWRITE_ATTEMPTS})...\n")
    elif failure_rate > settings.LINK_FAILURE_THRESHOLD:
        print(f"\n⚠️  FAILURE RATE {failure_rate:.0%} still high but max retries ({settings.MAX_QUERY_REWRITE_ATTEMPTS}) reached. Proceeding with available data.\n")
    else:
        print(f"\n✅ SCRAPE QUALITY OK ({failure_rate:.0%} failure rate). Proceeding to merge.\n")

    return state


def merge_texts_node(state: Phase1State) -> Phase1State:
    """Merge all successfully scraped text into a single string for the next phase."""
    texts = [page["text"] for page in state["scraped_pages"] if page["success"] and page["text"]]

    merged = "\n\n---\n\n".join(texts)
    state["merged_research_text"] = merged

    print(f"📝 MERGED {len(texts)} documents into research text ({len(merged)} chars)\n")

    return state


def dedup_and_rank_node(state: Phase1State) -> Phase1State:
    """Deduplicate and rank chunks (pure in-memory processing)."""
    from src.agents.phase1.dedup_relevance_scorer import process

    print("\n🔬 DEDUP & RELEVANCE SCORING")
    print(f"   Input: {len(state['merged_research_text'])} characters\n")

    result = process(
        merged_text=state["merged_research_text"],
        topic=state["topic"],
        scraped_pages=state["scraped_pages"]
    )

    state["ranked_chunks"] = result["ranked_chunks"]
    state["dedup_stats"] = result["stats"]

    stats = result["stats"]
    print(f"   ✅ Total chunks: {stats['total_chunks']}")
    print(f"   🔄 Duplicates removed: {stats['duplicates_removed']}")
    print(f"   ⭐ Top-K selected: {stats['top_k_selected']}\n")

    return state


# ==================== ROUTING ====================

def route_scrape_quality(state: Phase1State) -> str:
    """Route based on scrape failure rate.

    If >30% failed AND retries remain -> loop back to generate_queries
    Otherwise -> proceed to merge_texts
    """
    if (state["scrape_failure_rate"] > settings.LINK_FAILURE_THRESHOLD
            and state["query_rewrite_count"] < settings.MAX_QUERY_REWRITE_ATTEMPTS):
        return "generate_queries"
    return "merge_texts"


def route_freshness(state: Phase1State) -> str:
    """Route workflow based on freshness classification.

    Recent topics -> seed_search (to gather context)
    Evergreen topics -> generate_queries (skip seed search)
    """
    if state["freshness"] == "recent":
        return "seed_search"
    else:
        return "generate_queries"


# ==================== GRAPH CONSTRUCTION ====================

def create_phase1_graph():
    """Create and compile the Phase 1 (Research & Ingestion) subgraph.

    Workflow:
    1. initialize -> Set up state
    2. classify_freshness -> recent/evergreen
    3a. Recent: seed_search -> extract_context -> generate_queries
    3b. Evergreen: generate_queries
    4. date_tagging -> execute_searches
    5. scrape_pages -> evaluate_scrape_quality
    6. If >30% fail & retries<3: loop back to generate_queries
    7. Otherwise: merge_texts -> dedup_and_rank -> END

    Returns:
        CompiledGraph: Ready-to-run Phase 1 subgraph
    """
    workflow = StateGraph(Phase1State)

    # Add nodes
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("classify_freshness", classify_freshness_node)
    workflow.add_node("seed_search", seed_search_node)
    workflow.add_node("extract_context", extract_context_node)
    workflow.add_node("generate_queries", generate_queries_node)
    workflow.add_node("date_tagging", date_tagging_node)
    workflow.add_node("execute_searches", execute_searches_node)
    workflow.add_node("scrape_pages", scrape_pages_node)
    workflow.add_node("evaluate_scrape_quality", evaluate_scrape_quality_node)
    workflow.add_node("merge_texts", merge_texts_node)
    workflow.add_node("dedup_and_rank", dedup_and_rank_node)

    # Define edges
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "classify_freshness")

    # Conditional edge based on freshness
    workflow.add_conditional_edges(
        "classify_freshness",
        route_freshness,
        {
            "seed_search": "seed_search",
            "generate_queries": "generate_queries"
        }
    )

    # Recent path
    workflow.add_edge("seed_search", "extract_context")
    workflow.add_edge("extract_context", "generate_queries")

    # Both paths converge
    workflow.add_edge("generate_queries", "date_tagging")
    workflow.add_edge("date_tagging", "execute_searches")
    workflow.add_edge("execute_searches", "scrape_pages")
    workflow.add_edge("scrape_pages", "evaluate_scrape_quality")

    # Retry loop: if >30% fail and retries remain, loop back to generate_queries
    workflow.add_conditional_edges(
        "evaluate_scrape_quality",
        route_scrape_quality,
        {
            "generate_queries": "generate_queries",
            "merge_texts": "merge_texts"
        }
    )

    # Final processing: merge -> dedup+rank -> END
    workflow.add_edge("merge_texts", "dedup_and_rank")
    workflow.add_edge("dedup_and_rank", END)

    return workflow.compile()
