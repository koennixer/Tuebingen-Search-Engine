"""Shared data models used by crawler, retrieval, and presentation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """A fetched page ready to be stored and indexed."""

    url: str
    title: str
    text: str
    html: str
    fetched_at: str
    content_type: str
    status_code: int


@dataclass(frozen=True)
class SearchResult:
    """A ranked search result returned by the retrieval pipeline."""

    rank: int
    doc_id: int
    url: str
    title: str
    score: float
    snippet: str
    bm25_score: float
    expanded_score: float

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "rank": self.rank,
            "doc_id": self.doc_id,
            "url": self.url,
            "title": self.title,
            "score": self.score,
            "snippet": self.snippet,
            "bm25_score": self.bm25_score,
            "expanded_score": self.expanded_score,
        }
