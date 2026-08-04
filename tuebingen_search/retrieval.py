"""Two-stage retrieval: hand-built BM25F followed by an exploratory reranker.

The first stage is entirely implemented in this module. The second stage
combines conservative pseudo-relevance feedback, corpus-trained latent semantic
analysis, query intent agreement, link authority, and source diversification.
No retrieval-trained model or dedicated search toolkit is used.
"""

from __future__ import annotations

import math
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from .intent import (
    infer_query_intents,
    intent_label,
    intent_match_score,
    query_intent_expansion,
)
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
from .vocabulary import load_vocabulary

DEFAULT_RETRIEVAL_TOP_K = 100

try:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TruncatedSVD = None
    TfidfVectorizer = None
    cosine_similarity = None


BM25_K1 = 1.2
FIELD_WEIGHTS = {"title": 2.6, "body": 1.0, "url": 1.8}
FIELD_B = {"title": 0.35, "body": 0.75, "url": 0.20}
RM3_FEEDBACK_DOCS = 10
RM3_EXPANSION_TERMS = 12
RESULT_POOL_SIZE = 750
SEMANTIC_RERANK_POOL_SIZE = 250


def _batched(iterable: Iterable[object], size: int):
    """Python 3.10-compatible equivalent of ``itertools.batched``."""
    if size < 1:
        raise ValueError("size must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, size)):
        yield batch


def _placeholders(values: Iterable[object]) -> str:
    return ",".join("?" for _ in values)


def _collection_stats(
    conn: sqlite3.Connection,
) -> tuple[int, dict[str, float]]:
    row = conn.execute(
        """
        SELECT COUNT(*),
               COALESCE(AVG(title_token_count), 0.0),
               COALESCE(AVG(body_token_count), 0.0),
               COALESCE(AVG(url_token_count), 0.0)
        FROM documents
        """
    ).fetchone()
    return int(row[0]), {
        "title": max(float(row[1]), 1.0),
        "body": max(float(row[2]), 1.0),
        "url": max(float(row[3]), 1.0),
    }


def _document_frequencies(
    conn: sqlite3.Connection, terms: Iterable[str]
) -> dict[str, int]:
    unique_terms = sorted(set(terms))
    if not unique_terms:
        return {}
    result: dict[str, int] = {}
    for chunk in _batched(unique_terms, 900):
        rows = conn.execute(
            f"""
            SELECT term, document_frequency
            FROM terms
            WHERE term IN ({_placeholders(chunk)})
            """,
            chunk,
        )
        result.update({str(term): int(df) for term, df in rows})
    return result


def _bm25_idf(document_count: int, document_frequency: int) -> float:
    return math.log(
        1.0
        + (document_count - document_frequency + 0.5)
        / (document_frequency + 0.5)
    )


def bm25_scores(
    conn: sqlite3.Connection,
    weighted_terms: dict[str, float],
    *,
    limit: int = RESULT_POOL_SIZE,
) -> dict[int, float]:
    """Score candidates using a self-implemented field-aware BM25 formula."""
    weighted_terms = {
        term: float(weight)
        for term, weight in weighted_terms.items()
        if weight > 0
    }
    if not weighted_terms:
        return {}

    document_count, average_lengths = _collection_stats(conn)
    if document_count == 0:
        return {}
    terms = sorted(weighted_terms)
    document_frequencies = _document_frequencies(conn, terms)
    if not document_frequencies:
        return {}

    scores: Counter[int] = Counter()
    for chunk in _batched(terms, 900):
        rows = conn.execute(
            f"""
            SELECT p.term, p.doc_id,
                   p.title_frequency, p.body_frequency, p.url_frequency,
                   d.title_token_count, d.body_token_count, d.url_token_count
            FROM postings p
            JOIN documents d ON d.id = p.doc_id
            WHERE p.term IN ({_placeholders(chunk)})
            """,
            chunk,
        )
        for (
            term,
            doc_id,
            title_frequency,
            body_frequency,
            url_frequency,
            title_length,
            body_length,
            url_length,
        ) in rows:
            df = document_frequencies.get(str(term))
            if not df:
                continue
            field_values = (
                ("title", float(title_frequency), float(title_length)),
                ("body", float(body_frequency), float(body_length)),
                ("url", float(url_frequency), float(url_length)),
            )
            field_tf = 0.0
            for field, frequency, length in field_values:
                length_normalization = (
                    1.0
                    - FIELD_B[field]
                    + FIELD_B[field] * length / average_lengths[field]
                )
                field_tf += (
                    FIELD_WEIGHTS[field]
                    * frequency
                    / max(length_normalization, 0.05)
                )
            if field_tf <= 0:
                continue
            saturation = (field_tf * (BM25_K1 + 1.0)) / (field_tf + BM25_K1)
            scores[int(doc_id)] += (
                weighted_terms[str(term)]
                * _bm25_idf(document_count, df)
                * saturation
            )
    return dict(scores.most_common(max(1, limit)))


def _rm3_expansion_weights(
    conn: sqlite3.Connection,
    query_term_set: set[str],
    first_stage_scores: dict[int, float],
    *,
    feedback_docs: int = RM3_FEEDBACK_DOCS,
    expansion_terms: int = RM3_EXPANSION_TERMS,
) -> dict[str, float]:
    """Estimate conservative corpus-specific expansion terms from top results."""
    top_doc_ids = list(first_stage_scores)[:feedback_docs]
    if not top_doc_ids:
        return {}

    document_count, _average_lengths = _collection_stats(conn)
    stopwords = ENGLISH_STOPWORDS | NORMALIZED_GERMAN_STOPWORDS
    rank_weights = {
        doc_id: 1.0 / math.log(rank + 2.0)
        for rank, doc_id in enumerate(top_doc_ids, start=1)
    }
    term_scores: Counter[str] = Counter()
    for chunk in _batched(top_doc_ids, 900):
        rows = conn.execute(
            f"""
            SELECT p.term, p.doc_id, p.body_frequency,
                   d.body_token_count, t.document_frequency
            FROM postings p
            JOIN documents d ON d.id = p.doc_id
            JOIN terms t ON t.term = p.term
            WHERE p.doc_id IN ({_placeholders(chunk)})
            """,
            chunk,
        )
        for term, doc_id, frequency, length, document_frequency in rows:
            term = str(term)
            if (
                term in query_term_set
                or term in stopwords
                or len(term) < 3
                or term.isdigit()
                or int(document_frequency) / max(document_count, 1) > 0.40
            ):
                continue
            idf = _bm25_idf(document_count, int(document_frequency))
            normalized_tf = float(frequency) / max(float(length), 1.0)
            term_scores[term] += (
                normalized_tf * idf * rank_weights[int(doc_id)]
            )

    best = term_scores.most_common(expansion_terms)
    if not best:
        return {}
    maximum = max(score for _term, score in best) or 1.0
    return {term: 0.30 * score / maximum for term, score in best}


def _metadata_for_docs(
    conn: sqlite3.Connection, doc_ids: Iterable[int]
) -> dict[int, dict[str, Any]]:
    ids = list(doc_ids)
    if not ids:
        return {}
    result: dict[int, dict[str, Any]] = {}
    for chunk in _batched(ids, 900):
        rows = conn.execute(
            f"""
            SELECT id, url, canonical_url, title, description, text,
                   topic, topic_confidence, pagerank, content_fingerprint
            FROM documents
            WHERE id IN ({_placeholders(chunk)})
            """,
            chunk,
        )
        for row in rows:
            result[int(row[0])] = {
                "url": row[1] or "",
                "canonical_url": row[2] or row[1] or "",
                "title": row[3] or "",
                "description": row[4] or "",
                "text": row[5] or "",
                "topic": row[6] or "general",
                "topic_confidence": float(row[7] or 0.0),
                "pagerank": float(row[8] or 0.0),
                "content_fingerprint": row[9] or "",
            }
    return result


def _metadata_signal(
    query: str, terms: list[str], metadata: dict[str, Any]
) -> float:
    if not terms:
        return 0.0
    title_tokens = set(tokenize(str(metadata["title"])))
    url_tokens = set(tokenize(str(metadata["canonical_url"])))
    body_tokens = set(tokenize(str(metadata["text"])[:15_000]))
    term_set = set(terms)
    coverage = len(term_set & (title_tokens | url_tokens | body_tokens)) / len(
        term_set
    )
    title_overlap = len(term_set & title_tokens) / len(term_set)
    url_overlap = len(term_set & url_tokens) / len(term_set)
    query_normalized = normalize_for_matching(query)
    title_phrase = float(
        bool(query_normalized)
        and query_normalized in normalize_for_matching(str(metadata["title"]))
    )
    body_phrase = float(
        bool(query_normalized)
        and query_normalized
        in normalize_for_matching(str(metadata["text"])[:20_000])
    )
    return min(
        1.0,
        0.35 * coverage
        + 0.30 * title_overlap
        + 0.10 * url_overlap
        + 0.15 * title_phrase
        + 0.10 * body_phrase,
    )


def _normalize_score_map(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    maximum = max(scores.values())
    minimum = min(scores.values())
    if math.isclose(maximum, minimum):
        return {doc_id: 1.0 if maximum > 0 else 0.0 for doc_id in scores}
    return {
        doc_id: (score - minimum) / (maximum - minimum)
        for doc_id, score in scores.items()
    }


def _latent_semantic_scores(
    query: str,
    candidate_ids: list[int],
    metadata: dict[int, dict[str, Any]],
) -> dict[int, float]:
    """Learn an LSA space from this index's candidates and compare the query."""
    if (
        TfidfVectorizer is None
        or TruncatedSVD is None
        or cosine_similarity is None
        or len(candidate_ids) < 3
    ):
        return {}

    valid_ids = [doc_id for doc_id in candidate_ids if doc_id in metadata]
    documents = [
        " ".join(
            [
                str(metadata[doc_id]["title"]),
                str(metadata[doc_id]["description"]),
                str(metadata[doc_id]["text"])[:8_000],
            ]
        )
        for doc_id in valid_ids
    ]
    try:
        vectorizer = TfidfVectorizer(
            preprocessor=normalize_for_matching,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=12_000,
        )
        matrix = vectorizer.fit_transform([query, *documents])
        components = min(64, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if components < 2:
            return {}
        projection = TruncatedSVD(
            n_components=components, n_iter=7, random_state=4271
        ).fit_transform(matrix)
        similarities = cosine_similarity(
            projection[0:1], projection[1:]
        ).ravel()
    except (ValueError, RuntimeError):
        return {}
    return {
        doc_id: max(0.0, float(score))
        for doc_id, score in zip(valid_ids, similarities)
    }


def _diversified_order(
    scores: dict[int, float],
    metadata: dict[int, dict[str, Any]],
    *,
    top_k: int,
) -> list[tuple[int, float, float]]:
    """Apply a light xQuAD-style source/topic novelty adjustment."""
    remaining = set(scores)
    selected: list[tuple[int, float, float]] = []
    domains: Counter[str] = Counter()
    topics: Counter[str] = Counter()
    fingerprints: set[str] = set()

    while remaining and len(selected) < top_k:
        best_id: int | None = None
        best_adjusted = -math.inf
        best_penalty = 0.0
        for doc_id in remaining:
            doc = metadata[doc_id]
            fingerprint = str(doc["content_fingerprint"])
            if fingerprint and fingerprint in fingerprints:
                continue
            domain = urlparse(str(doc["canonical_url"])).netloc.lower()
            topic = str(doc["topic"])
            penalty = min(0.12, 0.028 * domains[domain])
            novelty = (0.012 if domains[domain] == 0 else 0.0) + (
                0.008 if topics[topic] == 0 else 0.0
            )
            adjusted = scores[doc_id] - penalty + novelty
            if adjusted > best_adjusted:
                best_id = doc_id
                best_adjusted = adjusted
                best_penalty = penalty
        if best_id is None:
            break
        remaining.remove(best_id)
        doc = metadata[best_id]
        domain = urlparse(str(doc["canonical_url"])).netloc.lower()
        topic = str(doc["topic"])
        domains[domain] += 1
        topics[topic] += 1
        fingerprint = str(doc["content_fingerprint"])
        if fingerprint:
            fingerprints.add(fingerprint)
        selected.append((best_id, best_adjusted, best_penalty))
    return selected


def _why_text(
    terms: list[str],
    matched_terms: list[str],
    topic: str,
    components: dict[str, float],
) -> str:
    reasons: list[str] = []
    if matched_terms:
        reasons.append("matches " + ", ".join(matched_terms[:4]))
    if components.get("semantic", 0.0) >= 0.45:
        reasons.append("strong semantic context")
    if components.get("intent", 0.0) >= 0.55:
        reasons.append(f"fits {intent_label(topic).lower()}")
    if components.get("authority", 0.0) >= 0.60:
        reasons.append("well-linked source")
    if not reasons and terms:
        reasons.append("lexically relevant to the query")
    return "; ".join(reasons).capitalize() + "."


def retrieve(
    query: str,
    index: str | Path,
    *,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> list[dict[str, Any]]:
    """Return up to 100 ranked documents for one textual query."""
    terms = query_terms(query)
    if not terms:
        return []
    top_k = max(1, min(int(top_k), 100))

    conn = connect(index)
    try:
        original_weights = {
            term: float(frequency) for term, frequency in Counter(terms).items()
        }
        first_stage = bm25_scores(
            conn, original_weights, limit=RESULT_POOL_SIZE
        )
        if not first_stage:
            return []

        feedback_expansion = _rm3_expansion_weights(
            conn, set(terms), first_stage
        )
        intent_expansion = query_intent_expansion(query)
        expansion_weights = dict(feedback_expansion)
        for term, weight in intent_expansion.items():
            expansion_weights[term] = max(
                expansion_weights.get(term, 0.0), weight
            )

        expanded_weights = dict(original_weights)
        for term, weight in expansion_weights.items():
            expanded_weights[term] = expanded_weights.get(term, 0.0) + weight
        expanded_stage = bm25_scores(
            conn, expanded_weights, limit=RESULT_POOL_SIZE
        )

        first_norm = _normalize_score_map(first_stage)
        expanded_norm = _normalize_score_map(expanded_stage)
        candidate_ids = set(first_norm) | set(expanded_norm)
        metadata = _metadata_for_docs(conn, candidate_ids)

        metadata_scores = {
            doc_id: _metadata_signal(query, terms, metadata[doc_id])
            for doc_id in candidate_ids
        }
        intent_scores = {
            doc_id: intent_match_score(
                query,
                str(metadata[doc_id]["canonical_url"]),
                str(metadata[doc_id]["title"]),
                str(metadata[doc_id]["text"]),
            )
            for doc_id in candidate_ids
        }
        authority_norm = _normalize_score_map(
            {
                doc_id: float(metadata[doc_id]["pagerank"])
                for doc_id in candidate_ids
            }
        )

        preliminary = {
            doc_id: (
                0.25 * first_norm.get(doc_id, 0.0)
                + 0.60 * expanded_norm.get(doc_id, 0.0)
                + 0.10 * metadata_scores[doc_id]
                + 0.05 * intent_scores[doc_id]
            )
            for doc_id in candidate_ids
        }
        semantic_pool = sorted(
            preliminary, key=preliminary.get, reverse=True
        )[:SEMANTIC_RERANK_POOL_SIZE]
        semantic_norm = _normalize_score_map(
            _latent_semantic_scores(query, semantic_pool, metadata)
        )

        component_maps: dict[int, dict[str, float]] = {}
        final_scores: dict[int, float] = {}
        for doc_id in candidate_ids:
            components = {
                "bm25f": first_norm.get(doc_id, 0.0),
                "feedback": expanded_norm.get(doc_id, 0.0),
                "semantic": semantic_norm.get(doc_id, 0.0),
                "intent": intent_scores.get(doc_id, 0.0),
                "metadata": metadata_scores.get(doc_id, 0.0),
                "authority": authority_norm.get(doc_id, 0.0),
            }
            component_maps[doc_id] = components
            final_scores[doc_id] = (
                0.18 * components["bm25f"]
                + 0.54 * components["feedback"]
                + 0.12 * components["semantic"]
                + 0.08 * components["intent"]
                + 0.06 * components["metadata"]
                + 0.02 * components["authority"]
            )

        selected = _diversified_order(
            final_scores, metadata, top_k=top_k
        )
        selected_scores = _normalize_score_map(
            {doc_id: adjusted for doc_id, adjusted, _penalty in selected}
        )
        expansion_terms = tuple(
            term
            for term, _weight in sorted(
                expansion_weights.items(),
                key=lambda item: (-item[1], item[0]),
            )[:RM3_EXPANSION_TERMS]
        )

        results: list[SearchResult] = []
        for rank, (doc_id, _adjusted, diversity_penalty) in enumerate(
            selected, start=1
        ):
            doc = metadata[doc_id]
            visible_tokens = set(
                tokenize(
                    " ".join(
                        [
                            str(doc["title"]),
                            str(doc["description"]),
                            str(doc["text"])[:15_000],
                            str(doc["canonical_url"]),
                        ]
                    )
                )
            )
            matched_terms = [term for term in terms if term in visible_tokens]
            components = dict(component_maps[doc_id])
            components["diversity_penalty"] = diversity_penalty
            topic = str(doc["topic"])
            canonical_url = str(doc["canonical_url"] or doc["url"])
            results.append(
                SearchResult(
                    rank=rank,
                    doc_id=doc_id,
                    url=canonical_url,
                    title=str(doc["title"]),
                    score=round(selected_scores.get(doc_id, 0.0), 6),
                    snippet=snippet(
                        str(doc["description"] or doc["text"]), terms
                    ),
                    reading_time_mins=max(1, len(str(doc["text"]).split()) // 250),
                    bm25_score=round(first_stage.get(doc_id, 0.0), 6),
                    expanded_score=round(
                        expanded_stage.get(doc_id, 0.0), 6
                    ),
                    semantic_score=round(
                        semantic_norm.get(doc_id, 0.0), 6
                    ),
                    authority_score=round(
                        authority_norm.get(doc_id, 0.0), 6
                    ),
                    intent=topic,
                    intent_label=intent_label(topic),
                    source_domain=urlparse(canonical_url).netloc.lower(),
                    matched_terms=tuple(dict.fromkeys(matched_terms)),
                    expansion_terms=expansion_terms,
                    score_components={
                        key: round(value, 4)
                        for key, value in components.items()
                    },
                    why=_why_text(terms, matched_terms, topic, components),
                )
            )
        return [result.as_dict() for result in results]
    finally:
        conn.close()
def suggest_correction(query: str, index: str | Path) -> str | None:
    """Return a spelling correction suggestion if one is found, else None."""
    sym_spell = load_vocabulary(index)
    if not sym_spell:
        return None
    query_lower = query.lower()
    suggestions = sym_spell.lookup_compound(query_lower, max_edit_distance=2)
    if suggestions:
        best = suggestions[0].term
        if best != query_lower:
            return best
    return None


def load_query_file(path: str | Path) -> list[tuple[str, str]]:
    """Load assignment-style ``query_id<TAB>query text`` files."""
    queries: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                query_id, text = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid query line {line_number}: {line!r}"
                    )
                query_id, text = parts
            query_id = query_id.strip()
            text = text.strip()
            if not query_id or not text:
                raise ValueError(
                    f"Query id and text are required on line {line_number}"
                )
            if query_id in seen_ids:
                raise ValueError(
                    f"Duplicate query id {query_id!r} on line {line_number}"
                )
            seen_ids.add(query_id)
            queries.append((query_id, text))
    return queries


def retrieve_batch_list(
    queries: list[tuple[str, str]],
    index: str | Path,
    *,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> dict[str, list[dict[str, Any]]]:
    """Run queries concurrently while preserving input query order."""
    if not queries:
        return {}
    completed: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=min(4, len(queries))) as executor:
        futures = {
            executor.submit(retrieve, text, index, top_k=top_k): query_id
            for query_id, text in queries
        }
        for future in as_completed(futures):
            completed[futures[future]] = future.result()
    return {query_id: completed[query_id] for query_id, _text in queries}


def retrieve_batch(
    query_file: str | Path,
    index: str | Path,
    *,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> dict[str, list[dict[str, Any]]]:
    """Run retrieval for every query in a batch file."""
    return retrieve_batch_list(
        load_query_file(query_file), index, top_k=top_k
    )


def query_analysis(query: str) -> dict[str, Any]:
    """Expose the transparent query interpretation used by the interface."""
    return {
        "terms": query_terms(query),
        "intents": infer_query_intents(query),
    }
