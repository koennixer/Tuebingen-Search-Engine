"""Shared data models used by crawler, retrieval, and presentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    canonical_url: str = ""
    description: str = ""
    language: str = ""


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
    semantic_score: float = 0.0
    authority_score: float = 0.0
    intent: str = "general"
    intent_label: str = "General Tübingen"
    source_domain: str = ""
    matched_terms: tuple[str, ...] = ()
    expansion_terms: tuple[str, ...] = ()
    score_components: dict[str, float] = field(default_factory=dict)
    why: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "doc_id": self.doc_id,
            "url": self.url,
            "title": self.title,
            "score": self.score,
            "snippet": self.snippet,
            "bm25_score": self.bm25_score,
            "expanded_score": self.expanded_score,
            "semantic_score": self.semantic_score,
            "authority_score": self.authority_score,
            "intent": self.intent,
            "intent_label": self.intent_label,
            "source_domain": self.source_domain,
            "matched_terms": list(self.matched_terms),
            "expansion_terms": list(self.expansion_terms),
            "score_components": dict(self.score_components),
            "why": self.why,
        }
