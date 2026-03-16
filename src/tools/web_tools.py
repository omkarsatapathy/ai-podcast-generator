"""Web search and content extraction tools."""
import os
from datetime import datetime
from typing import List, Dict, Any
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import trafilatura
import httpx


class GoogleSearchTool:
    """Google Custom Search API tool."""

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        self.search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

        if not self.api_key or not self.search_engine_id:
            raise ValueError("GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID must be set in .env")

        self.service = build("customsearch", "v1", developerKey=self.api_key)

    def search(self, query: str, num_results: int = 10, date_restrict: str = None) -> List[Dict[str, Any]]:
        """
        Execute Google search and return results.

        Args:
            query: Search query string
            num_results: Number of results to return (max 10 per request)
            date_restrict: Date restriction (e.g., 'd7' for past week, 'm1' for past month)

        Returns:
            List of search results with title, link, snippet
        """
        try:
            params = {
                "q": query,
                "cx": self.search_engine_id,
                "num": min(num_results, 5)
            }

            if date_restrict:
                params["dateRestrict"] = date_restrict

            result = self.service.cse().list(**params).execute()

            items = result.get("items", [])

            return [
                {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "displayLink": item.get("displayLink", "")
                }
                for item in items
            ]

        except HttpError as e:
            print(f"Google Search API error: {e}")
            return []
        except Exception as e:
            print(f"Search error: {e}")
            return []


class WebFetchTool:
    """Web content extraction tool using trafilatura."""

    def __init__(self):
        self.client = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; PodcastBot/1.0)"
            }
        )

    def fetch(self, url: str) -> Dict[str, Any]:
        """
        Fetch and extract main content from URL.

        Args:
            url: URL to fetch

        Returns:
            Dict with extracted text, title, date
        """
        try:
            response = self.client.get(url)
            response.raise_for_status()

            # Extract main content using trafilatura
            extracted = trafilatura.extract(
                response.text,
                include_comments=False,
                include_tables=True,
                output_format="txt"
            )

            # Also get metadata
            metadata = trafilatura.extract_metadata(response.text)

            return {
                "url": url,
                "text": extracted or "",
                "title": metadata.title if metadata else "",
                "date": metadata.date if metadata else "",
                "success": bool(extracted)
            }

        except Exception as e:
            print(f"Fetch error for {url}: {e}")
            return {
                "url": url,
                "text": "",
                "title": "",
                "date": "",
                "success": False,
                "error": str(e)
            }

    def __del__(self):
        """Close the HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()


def get_current_date() -> str:
    """Get current date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")


def get_date_restrict_param(freshness: str) -> str:
    """
    Convert freshness category to Google Search date restrict parameter.

    Args:
        freshness: 'recent', 'week', 'month', or None

    Returns:
        Date restrict parameter string (e.g., 'd7', 'm1')
    """
    mapping = {
        "recent": "d3",   # Past 3 days
        "week": "w1",     # Past week
        "month": "m1",    # Past month
    }
    return mapping.get(freshness, "")
