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
    "neckar",
    "stocherkahn",
}

ENGLISH_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", 
    "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", 
    "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", 
    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", 
    "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", 
    "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", 
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", 
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", 
    "don", "should", "now"
}

GERMAN_STOPWORDS = {
    "aber", "alle", "allem", "allen", "aller", "alles", "als", "also", "am", "an", "ander", "andere", 
    "anderem", "anderen", "anderer", "anderes", "anderm", "andern", "anderr", "anders", "auch", "auf", 
    "aus", "bei", "bin", "bis", "bist", "da", "damit", "dann", "der", "den", "des", "dem", "die", "das", 
    "dass", "daß", "derselbe", "derselben", "denselben", "desselben", "demselben", "dieselbe", "dieselben", 
    "dasselbe", "dazu", "dein", "deine", "deinem", "deinen", "deiner", "deines", "denn", "derer", "dessen", 
    "dich", "dir", "du", "dies", "diese", "diesem", "diesen", "dieser", "dieses", "doch", "dort", "durch", 
    "ein", "eine", "einem", "einen", "einer", "eines", "einig", "einige", "einigem", "einigen", "einiger", 
    "einiges", "einmal", "er", "ihn", "ihm", "es", "etwas", "euer", "eure", "eurem", "euren", "eurer", 
    "eures", "für", "gegen", "gewesen", "hab", "habe", "haben", "hat", "hatte", "hatten", "hier", "hin", 
    "hinter", "ich", "mich", "mir", "ihr", "ihre", "ihrem", "ihren", "ihrer", "ihres", "euch", "im", "in", 
    "indem", "ins", "ist", "jede", "jedem", "jeden", "jeder", "jedes", "jene", "jenem", "jenen", "jener", 
    "jenes", "jetzt", "kann", "kein", "keine", "keinem", "keinen", "keiner", "keines", "können", "könnte", 
    "machen", "man", "manche", "manchem", "manchen", "mancher", "manches", "mein", "meine", "meinem", 
    "meinen", "meiner", "meines", "mit", "muss", "musste", "nach", "nicht", "nichts", "noch", "nun", "nur", 
    "ob", "oder", "ohne", "sehr", "sein", "seine", "seinem", "seinen", "seiner", "seines", "selbst", "sich", 
    "sie", "ihnen", "sind", "so", "solche", "solchem", "solchen", "solcher", "solches", "soll", "sollte", 
    "sondern", "sonst", "über", "um", "und", "uns", "unsere", "unserem", "unseren", "unser", "unseres", 
    "unter", "viel", "vom", "von", "vor", "während", "war", "waren", "warst", "was", "weg", "weil", "weiter", 
    "welche", "welchem", "welchen", "welcher", "welches", "wenn", "werde", "werden", "wie", "wieder", "will", 
    "wir", "wird", "wirst", "wo", "wollen", "wollte", "würde", "würden", "zu", "zum", "zur", "zwar", "zwischen"
}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def normalize_for_matching(text: str) -> str:
    """Lowercase text and make Tuebingen/Tübingen/Tubingen match each other."""
    text = text.lower()
    text = text.replace("tübingen", "tubingen").replace("tuebingen", "tubingen")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.replace("tuebingen", "tubingen")


NORMALIZED_GERMAN_STOPWORDS = {normalize_for_matching(w) for w in GERMAN_STOPWORDS}


def tokenize(text: str) -> list[str]:
    """Normalize text and split it into searchable tokens."""
    return TOKEN_RE.findall(normalize_for_matching(text))


def query_terms(query: str) -> list[str]:
    """Tokenize a query and remove low-value stopwords."""
    terms = tokenize(query)
    return [term for term in terms if len(term) > 1]


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
    german_score = sum(counts[w] for w in NORMALIZED_GERMAN_STOPWORDS)
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
        sentence_tokens = set(tokenize(sentence))
        matched_terms = term_set & sentence_tokens
        if matched_terms:
            match_idx = sentence.lower().find(matched_terms.pop())
            return trim_snippet(sentence, length, max(0, match_idx))
    return trim_snippet(text, length)


def trim_snippet(text: str, length: int, match_index: int = 0) -> str:
    """Trim text on a word boundary for display, centered around match_index."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= length:
        return text
    
    start = max(0, match_index - (length // 2))
    end = start + length
    if end > len(text):
        end = len(text)
        start = max(0, end - length)
        
    trimmed = text[start:end]
    if start > 0:
        trimmed = "..." + trimmed.split(" ", 1)[-1]
    if end < len(text):
        trimmed = trimmed.rsplit(" ", 1)[0] + "..."
    return trimmed
