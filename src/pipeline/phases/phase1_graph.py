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


# ==================== NODES ====================

def initialize_node(state: Phase1State) -> Phase1State:
    """Initialize the agent state."""
    state["current_date"] = get_current_date()
    state["seed_results"] = []
    state["seed_context"] = ""
    state["queries"] = []
    state["messages"] = []
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
    """Generate 10 diverse search queries."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    topic = state["topic"]
    freshness = state["freshness"]
    seed_context = state.get("seed_context", "")
    current_date = state["current_date"]

    print(f"🤖 GENERATING QUERIES using LLM (freshness: {freshness})...\n")

    if freshness == "recent":
        system_prompt = f"""You are a search query expert. Generate 10 diverse, specific search queries for researching this recent topic.

Topic: {topic}
Current Date: {current_date}

Context from seed search:
{seed_context}

Generate queries that:
1. Cover different angles (who, what, why, impact, reactions, background)
2. Include specific names, dates, or events from the context
3. Are specific and factual
4. Will return high-quality sources

Return ONLY a numbered list of 10 queries, one per line."""

    else:  # evergreen
        system_prompt = f"""You are a search query expert. Generate 10 diverse, comprehensive search queries for researching this evergreen topic.

Topic: {topic}
Current Date: {current_date}

Generate queries that:
1. Cover foundational concepts, mechanisms, applications, debates
2. Include expert perspectives, research studies, case studies
3. Cover historical context and current state
4. Are specific and will return authoritative sources

Return ONLY a numbered list of 10 queries, one per line."""

    response = llm.invoke([SystemMessage(content=system_prompt)])

    # Parse queries
    queries = []
    for line in response.content.split("\n"):
        line = line.strip()
        if line and any(line.startswith(f"{i}.") for i in range(1, 11)):
            # Remove number prefix
            query_text = line.split(".", 1)[1].strip()
            queries.append({"query": query_text, "date_filter": None})

    state["queries"] = queries[:10]  # Ensure exactly 10

    print(f"✅ Generated {len(state['queries'])} queries\n")

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
            num_results=5,
            date_restrict=date_filter
        )

        # Store results in query dict
        query_dict["results"] = results

    print("✅ All searches completed!\n")

    return state


# ==================== ROUTING ====================

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

    This is an independent subgraph that can be:
    1. Imported and run standalone by query_producer.py
    2. Composed as a node in the main orchestrator

    Workflow:
    1. initialize -> Set up state with default values
    2. classify_freshness -> Determine if topic is recent/evergreen
    3a. Recent path: seed_search -> extract_context -> generate_queries
    3b. Evergreen path: generate_queries (skip seed search)
    4. date_tagging -> Add date filters to queries for recent topics
    5. execute_searches -> Run Google searches for all queries

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
    workflow.add_edge("execute_searches", END)

    return workflow.compile()
