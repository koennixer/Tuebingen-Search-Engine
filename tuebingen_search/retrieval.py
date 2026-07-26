"""Query processing with BM25 and pseudo-relevance-feedback reranking."""

from __future__ import annotations

import math
import sqlite3
from collections import Counter
from itertools import batched
from pathlib import Path
from typing import Iterable

from .models import SearchResult
from .storage import connect
from .text import (
    ENGLISH_STOPWORDS,
    NORMALIZED_GERMAN_STOPWORDS,
    normalize_for_matching,
    query_terms,
    snippet,
    tokenize,
)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None


BM25_K1 = 1.2
BM25_B = 0.75
RM3_FEEDBACK_DOCS = 10
RM3_EXPANSION_TERMS = 12
RESULT_POOL_SIZE = 750
ML_RERANK_POOL_SIZE = 250


def _placeholders(values: Iterable[object]) -> str:
    return ",".join("?" for _ in values)


def _collection_stats(conn: sqlite3.Connection) -> tuple[int, float]:
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(AVG(token_count), 0.0) FROM documents"
    ).fetchone()
    document_count = int(row[0])
    avg_doc_len = float(row[1]) if row[1] else 1.0
    return document_count, max(avg_doc_len, 1.0)


def _document_frequencies(conn: sqlite3.Connection, terms: Iterable[str]) -> dict[str, int]:
    unique_terms = sorted(set(terms))
    if not unique_terms:
        return {}

    result = {}
    for chunk in batched(unique_terms, 900):
        rows = conn.execute(
            f"""
            SELECT term, document_frequency
            FROM terms
            WHERE term IN ({_placeholders(chunk)})
            """,
            chunk,
        ).fetchall()
        for term, df in rows:
            result[term] = int(df)
    return result


def _bm25_idf(document_count: int, document_frequency: int) -> float:
    return math.log(1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))


def bm25_scores(
    conn: sqlite3.Connection,
    weighted_terms: dict[str, float],
    *,
    limit: int = RESULT_POOL_SIZE,
) -> dict[int, float]:
    """Score candidate documents with a self-implemented BM25 formula."""
    weighted_terms = {term: weight for term, weight in weighted_terms.items() if weight > 0}
    if not weighted_terms:
        return {}

    document_count, avg_doc_len = _collection_stats(conn)
    if document_count == 0:
        return {}

    terms = sorted(weighted_terms)
    dfs = _document_frequencies(conn, terms)
    if not dfs:
        return {}

    scores: Counter[int] = Counter()
    for chunk in batched(terms, 900):
        rows = conn.execute(
            f"""
            SELECT p.term, p.doc_id, p.term_frequency, d.token_count
            FROM postings p
            JOIN documents d ON d.id = p.doc_id
            WHERE p.term IN ({_placeholders(chunk)})
            """,
            chunk,
        )

        for term, doc_id, term_frequency, doc_len in rows:
            df = dfs.get(term)
            if not df:
                continue
            idf = _bm25_idf(document_count, df)
            length_norm = 1.0 - BM25_B + BM25_B * (float(doc_len) / avg_doc_len)
            tf_component = (term_frequency * (BM25_K1 + 1.0)) / (
                term_frequency + BM25_K1 * length_norm
            )
            scores[int(doc_id)] += weighted_terms[term] * idf * tf_component

    return dict(scores.most_common(limit))


def _rm3_expansion_weights(
    conn: sqlite3.Connection,
    query_term_set: set[str],
    first_stage_scores: dict[int, float],
    *,
    feedback_docs: int = RM3_FEEDBACK_DOCS,
    expansion_terms: int = RM3_EXPANSION_TERMS,
) -> dict[str, float]:
    top_doc_ids = list(first_stage_scores)[:feedback_docs]
    if not top_doc_ids:
        return {}

    document_count, _avg_doc_len = _collection_stats(conn)
    stopwords = ENGLISH_STOPWORDS | NORMALIZED_GERMAN_STOPWORDS
    rank_weights = {
        doc_id: 1.0 / math.log(rank + 2.0)
        for rank, doc_id in enumerate(top_doc_ids, start=1)
    }

    term_scores: Counter[str] = Counter()
    for chunk in batched(top_doc_ids, 900):
        rows = conn.execute(
            f"""
            SELECT p.term, p.doc_id, p.term_frequency, d.token_count, t.document_frequency
            FROM postings p
            JOIN documents d ON d.id = p.doc_id
            JOIN terms t ON t.term = p.term
            WHERE p.doc_id IN ({_placeholders(chunk)})
            """,
            chunk,
        )

        for term, doc_id, term_frequency, doc_len, document_frequency in rows:
            if term in query_term_set or term in stopwords or len(term) < 3 or term.isdigit():
                continue
            idf = _bm25_idf(document_count, int(document_frequency))
            normalized_tf = float(term_frequency) / max(float(doc_len), 1.0)
            term_scores[term] += normalized_tf * idf * rank_weights[int(doc_id)]

    if not term_scores:
        return {}

    best = term_scores.most_common(expansion_terms)
    max_score = max(score for _term, score in best) or 1.0
    return {term: 0.35 * (score / max_score) for term, score in best}


def _metadata_for_docs(conn: sqlite3.Connection, doc_ids: Iterable[int]) -> dict[int, dict[str, str]]:
    ids = list(doc_ids)
    if not ids:
        return {}

    result = {}
    for chunk in batched(ids, 900):
        rows = conn.execute(
            f"""
            SELECT id, url, title, text
            FROM documents
            WHERE id IN ({_placeholders(chunk)})
            """,
            chunk,
        ).fetchall()
        for doc_id, url, title, text in rows:
            result[int(doc_id)] = {
                "url": url or "",
                "title": title or "",
                "text": text or "",
            }
    return result


def _metadata_boost(query: str, terms: list[str], metadata: dict[str, str]) -> float:
    title_tokens = set(tokenize(metadata["title"]))
    url_tokens = set(tokenize(metadata["url"]))
    text_normalized = normalize_for_matching(metadata["text"])
    query_normalized = normalize_for_matching(query)
    term_set = set(terms)

    if not term_set:
        return 0.0

    title_overlap = len(term_set & title_tokens) / len(term_set)
    url_overlap = len(term_set & url_tokens) / len(term_set)
    phrase_match = 1.0 if query_normalized and query_normalized in text_normalized else 0.0
    title_phrase = 1.0 if query_normalized and query_normalized in normalize_for_matching(metadata["title"]) else 0.0
    return 0.10 * title_overlap + 0.04 * url_overlap + 0.04 * phrase_match + 0.07 * title_phrase


def _normalize_score_map(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    max_score = max(scores.values()) or 1.0
    return {doc_id: score / max_score for doc_id, score in scores.items()}


def _ml_similarity_scores(
    query: str,
    candidate_ids: list[int],
    metadata: dict[int, dict[str, str]],
) -> dict[int, float]:
    """Use optional scikit-learn TF-IDF cosine similarity as a second-stage signal."""
    if TfidfVectorizer is None or cosine_similarity is None or not candidate_ids:
        return {}

    documents = [
        f"{metadata[doc_id]['title']} {metadata[doc_id]['text'][:6000]}"
        for doc_id in candidate_ids
        if doc_id in metadata
    ]
    valid_doc_ids = [doc_id for doc_id in candidate_ids if doc_id in metadata]
    if not documents:
        return {}

    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 2),
            max_features=8000,
        )
        matrix = vectorizer.fit_transform([query, *documents])
        similarities = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    except Exception:
        return {}

    return {doc_id: float(score) for doc_id, score in zip(valid_doc_ids, similarities)}


def retrieve(query: str, index: str | Path, *, top_k: int = 100) -> list[dict[str, int | float | str]]:
    """Return the top documents for one query.

    Pipeline:
    1. BM25 first-stage retrieval over the inverted index.
    2. RM3-style pseudo-relevance feedback from the top-ranked documents.
    3. Lightweight reranking using title, URL, and exact phrase evidence.
    """
    terms = query_terms(query)
    if not terms:
        return []

    original_weights = dict(Counter(terms))
    conn = connect(index)
    try:
        first_stage = bm25_scores(conn, original_weights, limit=RESULT_POOL_SIZE)
        if not first_stage:
            return []

        expansion_weights = _rm3_expansion_weights(conn, set(terms), first_stage)
        expanded_weights = dict(original_weights)
        for term, weight in expansion_weights.items():
            expanded_weights[term] = expanded_weights.get(term, 0.0) + weight

        expanded_stage = bm25_scores(conn, expanded_weights, limit=RESULT_POOL_SIZE)

        first_norm = _normalize_score_map(first_stage)
        expanded_norm = _normalize_score_map(expanded_stage)
        candidate_ids = set(first_norm) | set(expanded_norm)
        metadata = _metadata_for_docs(conn, candidate_ids)

        metadata_scores: dict[int, float] = {}
        preliminary_scores: dict[int, float] = {}
        for doc_id in candidate_ids:
            metadata_scores[doc_id] = _metadata_boost(
                query,
                terms,
                metadata.get(doc_id, {"url": "", "title": "", "text": ""}),
            )
            preliminary_scores[doc_id] = (
                0.25 * first_norm.get(doc_id, 0.0)
                + 0.70 * expanded_norm.get(doc_id, 0.0)
                + metadata_scores[doc_id]
            )

        rerank_pool = sorted(preliminary_scores, key=preliminary_scores.get, reverse=True)[
            :ML_RERANK_POOL_SIZE
        ]
        ml_norm = _normalize_score_map(_ml_similarity_scores(query, rerank_pool, metadata))

        final_scores: dict[int, float] = {}
        for doc_id in candidate_ids:
            if ml_norm:
                final_scores[doc_id] = (
                    0.20 * first_norm.get(doc_id, 0.0)
                    + 0.55 * expanded_norm.get(doc_id, 0.0)
                    + 0.10 * metadata_scores.get(doc_id, 0.0)
                    + 0.15 * ml_norm.get(doc_id, 0.0)
                )
            else:
                final_scores[doc_id] = preliminary_scores[doc_id]

        ranked_doc_ids = sorted(final_scores, key=final_scores.get, reverse=True)[:top_k]
        results: list[SearchResult] = []
        for rank, doc_id in enumerate(ranked_doc_ids, start=1):
            doc = metadata[doc_id]
            results.append(
                SearchResult(
                    rank=rank,
                    doc_id=doc_id,
                    url=doc["url"],
                    title=doc["title"],
                    score=round(final_scores[doc_id], 6),
                    snippet=snippet(doc["text"], terms),
                    bm25_score=round(first_stage.get(doc_id, 0.0), 6),
                    expanded_score=round(expanded_stage.get(doc_id, 0.0), 6),
                )
            )

        return [result.as_dict() for result in results]
    finally:
        conn.close()


def load_query_file(path: str | Path) -> list[tuple[str, str]]:
    """Load assignment-style `query_id<TAB>query text` files."""
    queries: list[tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                query_id, text = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid query line {line_number}: {line!r}")
                query_id, text = parts
            queries.append((query_id.strip(), text.strip()))
    return queries


def retrieve_batch(
    query_file: str | Path,
    index: str | Path,
    *,
    top_k: int = 100,
) -> dict[str, list[dict[str, int | float | str]]]:
    """Run retrieval for every query in a batch file."""
    return {
        query_id: retrieve(text, index, top_k=top_k)
        for query_id, text in load_query_file(query_file)
    }
