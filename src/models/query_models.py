"""Pydantic models for Query Producer."""
from pydantic import BaseModel, Field
from typing import List, Literal
from datetime import datetime


class SearchQuery(BaseModel):
    """A single search query with metadata."""

    query: str = Field(..., description="The search query string")
    date_filter: str | None = Field(None, description="Optional date filter (e.g., 'd7', 'w1', 'm1')")
    rationale: str | None = Field(None, description="Why this query is useful")
    results: List[dict] | None = Field(None, description="Search results for this query")


class QueryProducerOutput(BaseModel):
    """Output from the Query Producer agent."""

    topic: str = Field(..., description="Original topic")
    freshness: Literal["recent", "evergreen"] = Field(..., description="Topic freshness classification")
    queries: List[SearchQuery] = Field(..., description="List of 15 search queries")
    chapter_titles: List[str] = Field(default_factory=list, description="List of 10 suggested chapter titles for stage 2")
    seed_context: str | None = Field(None, description="Context from seed search (if recent topic)")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class FreshnessClassification(BaseModel):
    """Freshness classification result."""

    classification: Literal["recent", "evergreen"] = Field(..., description="Is this a recent event or evergreen topic?")
    reasoning: str = Field(..., description="Why this classification was chosen")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
