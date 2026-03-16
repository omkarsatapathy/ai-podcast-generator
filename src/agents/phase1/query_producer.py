"""Query Producer Agent - Generates diverse search queries for podcast research.

This agent uses LangChain tools and a LangGraph workflow to:
1. Classify topics by freshness (recent vs evergreen)
2. Generate 10 diverse search queries
3. Execute Google searches with appropriate date filters
"""
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from src.tools.web_tools import GoogleSearchTool, WebFetchTool, get_current_date
from src.models.query_models import QueryProducerOutput, SearchQuery
from src.pipeline.phases.phase1_graph import create_phase1_graph


# ==================== TOOLS ====================

@tool
def web_search(query: str, num_results: int = 5) -> str:
    """
    Execute a Google search and return results.

    Args:
        query: The search query string
        num_results: Number of results to return (default 5)

    Returns:
        Formatted search results with titles, links, and snippets
    """
    search_tool = GoogleSearchTool()
    results = search_tool.search(query, num_results)

    if not results:
        return "No results found."

    output = []
    for i, result in enumerate(results, 1):
        output.append(
            f"{i}. {result['title']}\n"
            f"   URL: {result['link']}\n"
            f"   {result['snippet']}\n"
        )

    return "\n".join(output)


@tool
def web_fetch(url: str) -> str:
    """
    Fetch and extract main content from a URL.

    Args:
        url: The URL to fetch

    Returns:
        Extracted text content
    """
    fetch_tool = WebFetchTool()
    result = fetch_tool.fetch(url)

    if not result["success"]:
        return f"Failed to fetch {url}: {result.get('error', 'Unknown error')}"

    return f"Title: {result['title']}\nContent: {result['text'][:2000]}..."


@tool
def get_today_date() -> str:
    """
    Get today's date.

    Returns:
        Current date in YYYY-MM-DD format
    """
    return get_current_date()


@tool
def classify_freshness(topic: str) -> str:
    """
    Classify if a topic is about recent events or an evergreen topic.


    Args:
        topic: The topic to classify

    Returns:
        Classification result: 'recent' or 'evergreen'
    """
    # Simple heuristic checks
    recent_keywords = [
        "2024", "2025", "2026", "today", "yesterday", "this week", "this month",
        "latest", "breaking", "new", "recent", "current", "just", "now"
    ]

    topic_lower = topic.lower()

    # Check for recent keywords
    has_recent_keywords = any(kw in topic_lower for kw in recent_keywords)

    # Use LLM for final decision
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""Classify this topic as either 'recent' (time-sensitive news/events) or 'evergreen' (timeless topic).

Topic: {topic}

Here is a sample Heuristic check found recent keywords: {has_recent_keywords}. Apart from this heuristic, use your understanding of the topic to classify it.
we cant solely rely on keywords, so make a holistic judgment based on the topic's nature and context. give only 30% weightage to the presence of recent keywords and 70% weightage to your understanding of the topic's nature.

Respond with ONLY the word 'recent' or 'evergreen' followed by a brief reason.
Format: <classification>: <reason>
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"🏷️  FRESHNESS CLASSIFICATION RESULT: {response.content}\n")
    return response.content


# ==================== AGENT ====================

class QueryProducerAgent:
    """Query Producer Agent - generates diverse search queries for a topic."""

    def __init__(self):
        self.graph = create_phase1_graph()

    def run(self, topic: str) -> QueryProducerOutput:
        """
        Generate 10 diverse search queries for the given topic.

        Args:
            topic: The topic to research

        Returns:
            QueryProducerOutput with 10 search queries
        """
        # Initialize state
        initial_state = {
            "topic": topic,
            "freshness": "",
            "seed_results": [],
            "seed_context": "",
            "queries": [],
            "current_date": "",
            "messages": []
        }

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        # Convert to output model
        queries = [
            SearchQuery(
                query=q["query"],
                date_filter=q.get("date_filter"),
                results=q.get("results")
            )
            for q in final_state["queries"]
        ]

        return QueryProducerOutput(
            topic=topic,
            freshness=final_state["freshness"],
            queries=queries,
            seed_context=final_state.get("seed_context", "")
        )


# ==================== USAGE ====================

if __name__ == "__main__":
    # Example usage
    agent = QueryProducerAgent()

    # Test with a recent topic
    result = agent.run("SpaceX Starship launch 2024")
    print(f"Topic: {result.topic}")
    print(f"Freshness: {result.freshness}")
    print(f"\nQueries:")
    for i, q in enumerate(result.queries, 1):
        print(f"{i}. {q.query}")
        if q.date_filter:
            print(f"   Date filter: {q.date_filter}")
