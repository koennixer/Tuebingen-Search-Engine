"""Transparent query-intent and page-topic signals for Tübingen search.

The lexicons are deliberately small and auditable. They are not a replacement
for lexical retrieval; they supply a weak, query-dependent signal to the
second-stage reranker and power the exploratory interface.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .text import normalize_for_matching, tokenize


@dataclass(frozen=True)
class IntentDefinition:
    label: str
    keywords: frozenset[str]
    expansion_terms: tuple[str, ...]


INTENTS: dict[str, IntentDefinition] = {
    "culture_history": IntentDefinition(
        "Culture & history",
        frozenset(
            {
                "attraction",
                "castle",
                "church",
                "culture",
                "historic",
                "history",
                "landmark",
                "museum",
                "old",
                "palace",
                "sight",
                "tour",
            }
        ),
        ("museum", "castle", "historic", "sight", "landmark"),
    ),
    "food_drink": IntentDefinition(
        "Food & drink",
        frozenset(
            {
                "bar",
                "beer",
                "breakfast",
                "brewery",
                "cafe",
                "coffee",
                "dinner",
                "drink",
                "eat",
                "food",
                "lunch",
                "menu",
                "restaurant",
                "vegan",
                "wine",
            }
        ),
        ("restaurant", "cafe", "bar", "menu", "dining"),
    ),
    "nature_outdoors": IntentDefinition(
        "Nature & outdoors",
        frozenset(
            {
                "bike",
                "cycling",
                "forest",
                "garden",
                "hike",
                "hiking",
                "nature",
                "neckar",
                "outdoor",
                "park",
                "punting",
                "river",
                "trail",
                "walk",
            }
        ),
        ("hiking", "walk", "park", "neckar", "punting"),
    ),
    "university_student": IntentDefinition(
        "University & student life",
        frozenset(
            {
                "admission",
                "campus",
                "course",
                "erasmus",
                "exchange",
                "faculty",
                "international",
                "research",
                "semester",
                "student",
                "study",
                "university",
            }
        ),
        ("university", "student", "study", "campus", "international"),
    ),
    "accommodation": IntentDefinition(
        "Stay",
        frozenset(
            {
                "accommodation",
                "apartment",
                "booking",
                "hostel",
                "hotel",
                "overnight",
                "room",
                "sleep",
                "stay",
            }
        ),
        ("hotel", "accommodation", "room", "hostel", "stay"),
    ),
    "events_nightlife": IntentDefinition(
        "Events & nightlife",
        frozenset(
            {
                "club",
                "concert",
                "event",
                "festival",
                "music",
                "night",
                "nightlife",
                "party",
                "theatre",
            }
        ),
        ("event", "concert", "festival", "nightlife", "theatre"),
    ),
    "mobility_local": IntentDefinition(
        "Getting around",
        frozenset(
            {
                "arrival",
                "bus",
                "car",
                "cycling",
                "map",
                "parking",
                "route",
                "station",
                "taxi",
                "train",
                "transport",
                "travel",
            }
        ),
        ("transport", "bus", "train", "parking", "route"),
    ),
}


def _intent_counts(text: str) -> Counter[str]:
    tokens = tokenize(text)
    token_counts = Counter(tokens)
    scores: Counter[str] = Counter()
    for key, definition in INTENTS.items():
        scores[key] = sum(token_counts[word] for word in definition.keywords)
    return scores


def infer_query_intents(query: str, *, limit: int = 2) -> list[dict[str, object]]:
    """Return the strongest explicit intents in a query.

    A general intent is returned only when no topic lexicon matches. Scores are
    normalized within the query so they remain interpretable in the UI.
    """
    counts = _intent_counts(query)
    ranked = [(key, count) for key, count in counts.most_common() if count > 0]
    if not ranked:
        return [{"key": "general", "label": "General Tübingen", "score": 1.0}]
    maximum = float(ranked[0][1])
    return [
        {
            "key": key,
            "label": INTENTS[key].label,
            "score": round(count / maximum, 3),
        }
        for key, count in ranked[:limit]
    ]


def classify_document(url: str, title: str, text: str) -> tuple[str, float]:
    """Assign one display topic using title/URL evidence more heavily."""
    title_counts = _intent_counts(f"{url} {title}")
    body_counts = _intent_counts(text[:20_000])
    scores = {
        key: 3.0 * title_counts[key] + min(float(body_counts[key]), 8.0)
        for key in INTENTS
    }
    key, score = max(scores.items(), key=lambda item: item[1])
    if score <= 0:
        return "general", 0.0
    denominator = max(1.0, sum(scores.values()))
    return key, min(1.0, score / denominator)


def intent_match_score(query: str, url: str, title: str, text: str) -> float:
    """Return a bounded query/document intent agreement signal."""
    query_intents = [
        item["key"] for item in infer_query_intents(query) if item["key"] != "general"
    ]
    if not query_intents:
        return 0.5

    haystack = normalize_for_matching(f"{url} {title} {text[:12_000]}")
    tokens = Counter(tokenize(haystack))
    per_intent: list[float] = []
    for key in query_intents:
        definition = INTENTS[str(key)]
        matches = sum(min(tokens[word], 3) for word in definition.keywords)
        per_intent.append(min(1.0, matches / 5.0))
    return sum(per_intent) / len(per_intent)


def query_intent_expansion(query: str, *, weight: float = 0.12) -> dict[str, float]:
    """Return conservative expansion weights from explicit query intents."""
    expansions: dict[str, float] = {}
    query_tokens = set(tokenize(query))
    for item in infer_query_intents(query):
        key = str(item["key"])
        if key == "general":
            continue
        strength = float(item["score"])
        for term in INTENTS[key].expansion_terms:
            if term not in query_tokens:
                expansions[term] = max(expansions.get(term, 0.0), weight * strength)
    return expansions


def intent_label(key: str) -> str:
    """Return a human-facing label for a stored topic key."""
    definition = INTENTS.get(key)
    return definition.label if definition else "General Tübingen"
