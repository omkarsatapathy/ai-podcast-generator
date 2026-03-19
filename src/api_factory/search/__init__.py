"""Search provider factory — returns a search tool with a uniform interface.

All search tools expose:
    .search(query: str, num_results: int, date_restrict: str | None) -> list[dict]

Each result dict contains: title, link, snippet, displayLink.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Protocol, runtime_checkable

from config.settings import settings


@runtime_checkable
class SearchProvider(Protocol):
    def search(
        self, query: str, num_results: int = 10, date_restrict: str | None = None
    ) -> List[Dict[str, Any]]: ...


def get_search(provider: str | None = None) -> SearchProvider:
    """Return a configured search provider instance.

    Args:
        provider: "google" or "tavily". Defaults to settings.SEARCH_PROVIDER.
    """
    provider = provider or settings.SEARCH_PROVIDER

    if provider == "google":
        from src.tools.web_tools import GoogleSearchTool

        return GoogleSearchTool()

    if provider == "tavily":
        return _TavilySearch()

    raise ValueError(f"Unsupported search provider: '{provider}'")


class _TavilySearch:
    """Tavily Search API wrapper matching GoogleSearchTool interface."""

    def __init__(self) -> None:
        self.api_key = os.environ.get("TAVILY_API_KEY", "")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY must be set in .env or settings")

    def search(
        self, query: str, num_results: int = 10, date_restrict: str | None = None
    ) -> List[Dict[str, Any]]:
        import requests  # already in requirements

        payload: dict[str, Any] = {
            "api_key": self.api_key,
            "query": query,
            "max_results": num_results,
            "include_answer": False,
        }
        # Map Google-style date_restrict to Tavily's days parameter
        if date_restrict:
            days_map = {"d1": 1, "d3": 3, "d7": 7, "w1": 7, "m1": 30}
            days = days_map.get(date_restrict)
            if days:
                payload["days"] = days

        resp = requests.post(
            "https://api.tavily.com/search",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()

        return [
            {
                "title": r.get("title", ""),
                "link": r.get("url", ""),
                "snippet": r.get("content", ""),
                "displayLink": r.get("url", ""),
            }
            for r in resp.json().get("results", [])
        ]
