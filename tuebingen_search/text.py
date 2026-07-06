"""Text normalization, tokenization, filtering, and snippet helpers."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter


TUEBINGEN_TERMS = {
    "tubingen",
    "tuebingen",
    "tübingen",
    "universitat tubingen",
    "university of tubingen",
    "uni tubingen",
}

ENGLISH_STOPWORDS = {
    "the", "and", "of", "to", "in", "for", "with", "on", "at", "from",
    "by", "as", "is", "are", "was", "were", "be", "this", "that", "it",
    "you", "your", "we", "our", "about", "more", "can", "all", "not",
    "or", "an", "a", "their", "has", "have",
}

GERMAN_STOPWORDS = {
    "der", "die", "das", "und", "oder", "mit", "von", "für", "ist", "im",
    "in", "den", "des", "dem", "ein", "eine", "auf", "zu", "zur", "zum",
    "nicht", "auch", "sich", "als", "wir", "sie", "ihre",
}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def normalize_for_matching(text: str) -> str:
    """Lowercase text and make Tuebingen/Tübingen/Tubingen match each other."""
    text = text.lower()
    text = text.replace("tübingen", "tubingen").replace("tuebingen", "tubingen")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.replace("tuebingen", "tubingen")


def tokenize(text: str) -> list[str]:
    """Normalize text and split it into searchable tokens."""
    return TOKEN_RE.findall(normalize_for_matching(text))


def query_terms(query: str) -> list[str]:
    """Tokenize a query and remove low-value stopwords."""
    terms = tokenize(query)
    stopwords = ENGLISH_STOPWORDS | {normalize_for_matching(w) for w in GERMAN_STOPWORDS}
    return [term for term in terms if len(term) > 1 and term not in stopwords]


def is_probably_english(text: str, html_lang: str | None = None) -> bool:
    """Lightweight English-language filter for crawled pages."""
    if html_lang:
        lang = html_lang.strip().lower()
        if lang.startswith("en"):
            return True
        if lang.startswith("de"):
            return False

    words = tokenize(text[:20_000])
    if len(words) < 30:
        return False

    counts = Counter(words)
    english_score = sum(counts[w] for w in ENGLISH_STOPWORDS)
    german_score = sum(counts[normalize_for_matching(w)] for w in GERMAN_STOPWORDS)
    return english_score >= max(4, german_score * 1.7)


def is_tuebingen_related(url: str, title: str, text: str) -> bool:
    """Check whether a page is plausibly related to Tübingen."""
    haystack = normalize_for_matching(" ".join([url, title, text[:30_000]]))
    return any(normalize_for_matching(term) in haystack for term in TUEBINGEN_TERMS)


def snippet(text: str, terms: list[str], *, length: int = 260) -> str:
    """Choose a short snippet that contains at least one query term if possible."""
    if not text:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", text)
    term_set = set(terms)
    for sentence in sentences:
        if term_set & set(tokenize(sentence)):
            return trim_snippet(sentence, length)
    return trim_snippet(text, length)


def trim_snippet(text: str, length: int) -> str:
    """Trim text on a word boundary for display."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= length:
        return text
    trimmed = text[: length - 3].rsplit(" ", 1)[0]
    return f"{trimmed}..."
