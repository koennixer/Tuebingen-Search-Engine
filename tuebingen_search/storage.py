"""SQLite persistence for documents, crawl state, links, and fielded postings."""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from .intent import classify_document
from .models import Document

DEFAULT_MAX_CRAWL_RETRY_ATTEMPTS = 3
DEFAULT_INDEX_PATH = ".index/tuebingen_index.sqlite3"

from .text import normalize_for_matching, tokenize


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(index_path: str | Path) -> sqlite3.Connection:
    """Open an index database and apply backwards-compatible migrations."""
    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    initialize_schema(conn)
    return conn


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _add_missing_columns(
    conn: sqlite3.Connection, table: str, definitions: dict[str, str]
) -> None:
    existing = _columns(conn, table)
    for column, definition in definitions.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Create the document store, postings, link graph, and durable frontier."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL UNIQUE,
            canonical_url TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            text TEXT NOT NULL,
            html_gz BLOB NOT NULL,
            sha256 TEXT NOT NULL,
            content_fingerprint TEXT NOT NULL DEFAULT '',
            fetched_at TEXT NOT NULL,
            content_type TEXT,
            status_code INTEGER,
            language TEXT NOT NULL DEFAULT '',
            topic TEXT NOT NULL DEFAULT 'general',
            topic_confidence REAL NOT NULL DEFAULT 0.0,
            length INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            title_token_count INTEGER NOT NULL DEFAULT 0,
            body_token_count INTEGER NOT NULL DEFAULT 0,
            url_token_count INTEGER NOT NULL DEFAULT 0,
            pagerank REAL NOT NULL DEFAULT 0.0
        );

        CREATE TABLE IF NOT EXISTS terms (
            term TEXT PRIMARY KEY,
            document_frequency INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS postings (
            term TEXT NOT NULL,
            doc_id INTEGER NOT NULL,
            term_frequency INTEGER NOT NULL,
            title_frequency INTEGER NOT NULL DEFAULT 0,
            body_frequency INTEGER NOT NULL DEFAULT 0,
            url_frequency INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (term, doc_id),
            FOREIGN KEY (term) REFERENCES terms(term) ON DELETE CASCADE,
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS links (
            source_url TEXT NOT NULL,
            target_url TEXT NOT NULL,
            anchor_text TEXT NOT NULL DEFAULT '',
            discovered_at TEXT NOT NULL,
            PRIMARY KEY (source_url, target_url)
        );

        CREATE TABLE IF NOT EXISTS frontier (
            url TEXT PRIMARY KEY,
            status TEXT NOT NULL
                CHECK(status IN ('queued', 'visited', 'error', 'skipped')),
            discovered_at TEXT NOT NULL,
            last_attempt_at TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,
            priority REAL NOT NULL DEFAULT 0.0,
            error TEXT,
            depth INTEGER NOT NULL DEFAULT 0,
            bridge_steps INTEGER NOT NULL DEFAULT 0,
            context_relevant INTEGER NOT NULL DEFAULT 0,
            source_host TEXT,
            retryable INTEGER NOT NULL DEFAULT 0,
            http_status INTEGER
        );

        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_frontier_status_priority
            ON frontier(status, priority DESC, discovered_at ASC);
        CREATE INDEX IF NOT EXISTS idx_postings_doc_id ON postings(doc_id);
        CREATE INDEX IF NOT EXISTS idx_links_target ON links(target_url);
        """
    )

    # Databases from earlier project iterations are upgraded in place.
    _add_missing_columns(
        conn,
        "frontier",
        {
            "depth": "INTEGER NOT NULL DEFAULT 0",
            "bridge_steps": "INTEGER NOT NULL DEFAULT 0",
            "context_relevant": "INTEGER NOT NULL DEFAULT 0",
            "source_host": "TEXT",
            "retryable": "INTEGER NOT NULL DEFAULT 0",
            "http_status": "INTEGER",
        },
    )
    _add_missing_columns(
        conn,
        "documents",
        {
            "canonical_url": "TEXT NOT NULL DEFAULT ''",
            "description": "TEXT NOT NULL DEFAULT ''",
            "content_fingerprint": "TEXT NOT NULL DEFAULT ''",
            "language": "TEXT NOT NULL DEFAULT ''",
            "topic": "TEXT NOT NULL DEFAULT 'general'",
            "topic_confidence": "REAL NOT NULL DEFAULT 0.0",
            "title_token_count": "INTEGER NOT NULL DEFAULT 0",
            "body_token_count": "INTEGER NOT NULL DEFAULT 0",
            "url_token_count": "INTEGER NOT NULL DEFAULT 0",
            "pagerank": "REAL NOT NULL DEFAULT 0.0",
        },
    )
    _add_missing_columns(
        conn,
        "postings",
        {
            "title_frequency": "INTEGER NOT NULL DEFAULT 0",
            "body_frequency": "INTEGER NOT NULL DEFAULT 0",
            "url_frequency": "INTEGER NOT NULL DEFAULT 0",
        },
    )
    conn.execute(
        """
        UPDATE postings
        SET body_frequency = term_frequency
        WHERE title_frequency = 0 AND body_frequency = 0 AND url_frequency = 0
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_documents_fingerprint
        ON documents(content_fingerprint)
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_topic ON documents(topic)"
    )
    conn.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', '3')"
    )
    conn.commit()


def add_document(doc: Document | dict, index_path: str | Path) -> int:
    """Add or replace one document in the local fielded inverted index."""
    conn = connect(index_path)
    try:
        return add_document_to_connection(conn, doc)
    finally:
        conn.close()


def add_document_to_connection(conn: sqlite3.Connection, doc: Document | dict) -> int:
    """Add or replace one document using an existing SQLite connection."""
    if isinstance(doc, dict):
        doc = Document(
            url=doc["url"],
            title=doc.get("title", ""),
            text=doc.get("text", ""),
            html=doc.get("html", ""),
            fetched_at=doc.get("fetched_at", utc_now()),
            content_type=doc.get("content_type", ""),
            status_code=int(doc.get("status_code", 0)),
            canonical_url=doc.get("canonical_url", ""),
            description=doc.get("description", ""),
            language=doc.get("language", ""),
        )

    canonical_url = doc.canonical_url or doc.url
    title_tokens = tokenize(doc.title)
    body_tokens = tokenize(doc.text)
    url_tokens = tokenize(canonical_url)
    title_counts = Counter(title_tokens)
    body_counts = Counter(body_tokens)
    url_counts = Counter(url_tokens)
    all_terms = set(title_counts) | set(body_counts) | set(url_counts)
    topic, topic_confidence = classify_document(doc.url, doc.title, doc.text)
    normalized_content = normalize_for_matching(f"{doc.title}\n{doc.text}")
    content_fingerprint = hashlib.sha256(
        normalized_content.encode("utf-8", errors="ignore")
    ).hexdigest()
    sha256 = hashlib.sha256(doc.html.encode("utf-8", errors="ignore")).hexdigest()
    html_gz = gzip.compress(doc.html.encode("utf-8", errors="ignore"))

    with conn:
        old = conn.execute(
            "SELECT id FROM documents WHERE url = ?", (doc.url,)
        ).fetchone()
        if old:
            old_doc_id = int(old[0])
            old_terms = conn.execute(
                "SELECT term FROM postings WHERE doc_id = ?", (old_doc_id,)
            ).fetchall()
            conn.execute("DELETE FROM postings WHERE doc_id = ?", (old_doc_id,))
            conn.executemany(
                """
                UPDATE terms
                SET document_frequency = document_frequency - 1
                WHERE term = ?
                """,
                old_terms,
            )
            conn.execute("DELETE FROM terms WHERE document_frequency <= 0")
            conn.execute("DELETE FROM documents WHERE id = ?", (old_doc_id,))

        cursor = conn.execute(
            """
            INSERT INTO documents(
                url, canonical_url, title, description, text, html_gz, sha256,
                content_fingerprint, fetched_at, content_type, status_code,
                language, topic, topic_confidence, length, token_count,
                title_token_count, body_token_count, url_token_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc.url,
                canonical_url,
                doc.title,
                doc.description,
                doc.text,
                html_gz,
                sha256,
                content_fingerprint,
                doc.fetched_at,
                doc.content_type,
                doc.status_code,
                doc.language,
                topic,
                topic_confidence,
                len(doc.text),
                len(title_tokens) + len(body_tokens) + len(url_tokens),
                len(title_tokens),
                len(body_tokens),
                len(url_tokens),
            ),
        )
        doc_id = int(cursor.lastrowid)

        conn.executemany(
            """
            INSERT INTO terms(term, document_frequency)
            VALUES (?, 1)
            ON CONFLICT(term) DO UPDATE
            SET document_frequency = document_frequency + 1
            """,
            [(term,) for term in all_terms],
        )
        conn.executemany(
            """
            INSERT INTO postings(
                term, doc_id, term_frequency, title_frequency,
                body_frequency, url_frequency
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    term,
                    doc_id,
                    title_counts[term] + body_counts[term] + url_counts[term],
                    title_counts[term],
                    body_counts[term],
                    url_counts[term],
                )
                for term in all_terms
            ],
        )
    return doc_id


def add_to_frontier(
    conn: sqlite3.Connection,
    urls: Iterable[str],
    priority_fn,
    *,
    depth: int = 0,
    bridge_steps: int = 0,
    context_relevant: bool = False,
    source_host: str | None = None,
) -> int:
    """Insert newly discovered URLs into the restartable crawl frontier."""
    added = 0
    now = utc_now()
    for url in urls:
        if not url:
            continue
        try:
            cursor = conn.execute(
                """
                INSERT INTO frontier(
                    url, status, discovered_at, priority, depth, bridge_steps,
                    context_relevant, source_host
                )
                VALUES (?, 'queued', ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO NOTHING
                """,
                (
                    url,
                    now,
                    priority_fn(url),
                    max(0, depth),
                    max(0, bridge_steps),
                    int(context_relevant),
                    source_host,
                ),
            )
            added += cursor.rowcount == 1
        except sqlite3.Error:
            continue
    conn.commit()
    return added


def next_frontier_url(
    conn: sqlite3.Connection, *, excluded_hosts: set[str] | None = None
) -> str | None:
    """Return the highest-priority queued URL, optionally avoiding hosts."""
    item = next_frontier_item(conn, excluded_hosts=excluded_hosts)
    return str(item["url"]) if item else None


def next_frontier_item(
    conn: sqlite3.Connection, *, excluded_hosts: set[str] | None = None
) -> dict[str, object] | None:
    """Return the next queued URL together with traversal metadata."""
    rows = conn.execute(
        """
        SELECT url, depth, bridge_steps, context_relevant, source_host, attempts
        FROM frontier
        WHERE status = 'queued'
        ORDER BY priority DESC, discovered_at ASC
        """
    )
    for url, depth, bridge_steps, context_relevant, source_host, attempts in rows:
        if excluded_hosts and urlparse(url).netloc.lower() in excluded_hosts:
            continue
        return {
            "url": str(url),
            "depth": int(depth),
            "bridge_steps": int(bridge_steps),
            "context_relevant": bool(context_relevant),
            "source_host": source_host,
            "attempts": int(attempts),
        }
    return None


def mark_frontier(
    conn: sqlite3.Connection,
    url: str,
    status: str,
    error: str | None = None,
    *,
    retryable: bool = False,
    http_status: int | None = None,
    priority_penalty: float = 0.0,
) -> None:
    """Update crawl status and diagnostics for one URL."""
    conn.execute(
        """
        UPDATE frontier
        SET status = ?,
            last_attempt_at = ?,
            attempts = attempts + 1,
            error = ?,
            retryable = ?,
            http_status = ?,
            priority = priority - ?
        WHERE url = ?
        """,
        (
            status,
            utc_now(),
            error,
            int(retryable),
            http_status,
            max(0.0, float(priority_penalty)),
            url,
        ),
    )
    conn.commit()


def requeue_retryable_errors(conn: sqlite3.Connection, *, max_attempts: int = DEFAULT_MAX_CRAWL_RETRY_ATTEMPTS) -> int:
    """Move transient and now-fixed crawler failures back to the queue.

    Earlier crawler versions stored streaming timeouts and two extraction
    failures as permanent errors. Recognizing those messages here lets an
    existing frontier recover after upgrading instead of requiring users to
    delete their index.
    """
    cursor = conn.execute(
        """
        UPDATE frontier
        SET status = 'queued', error = NULL
        WHERE status = 'error'
          AND attempts < ?
          AND (
              retryable = 1
              OR error LIKE '%timed out%'
              OR error LIKE '%unexpected keyword argument%canonical_url%'
              OR error LIKE '%NoneType%object has no attribute%get%'
          )
        """,
        (max(1, max_attempts),),
    )
    conn.commit()
    return int(cursor.rowcount)


def frontier_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Count queued, visited, skipped, and errored URLs."""
    counts = {"queued": 0, "visited": 0, "skipped": 0, "error": 0}
    rows = conn.execute(
        "SELECT status, COUNT(*) FROM frontier GROUP BY status"
    ).fetchall()
    for status, count in rows:
        counts[str(status)] = int(count)
    return counts


def record_links(
    conn: sqlite3.Connection,
    source_url: str,
    links: Iterable[tuple[str, str]],
) -> int:
    """Persist canonical outgoing links for authority and audit signals."""
    rows = [
        (source_url, target_url, anchor[:500], utc_now())
        for target_url, anchor in links
        if target_url and target_url != source_url
    ]
    if not rows:
        return 0
    before = conn.total_changes
    conn.executemany(
        """
        INSERT INTO links(source_url, target_url, anchor_text, discovered_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(source_url, target_url) DO UPDATE
        SET anchor_text = excluded.anchor_text,
            discovered_at = excluded.discovered_at
        """,
        rows,
    )
    conn.commit()
    return conn.total_changes - before


def recompute_pagerank(
    conn: sqlite3.Connection,
    *,
    damping: float = 0.85,
    iterations: int = 25,
) -> int:
    """Compute PageRank over links whose source and target were both indexed."""
    documents = conn.execute(
        "SELECT id, url, canonical_url FROM documents"
    ).fetchall()
    if not documents:
        return 0

    url_to_id: dict[str, int] = {}
    for doc_id, url, canonical_url in documents:
        url_to_id[str(url)] = int(doc_id)
        if canonical_url:
            url_to_id[str(canonical_url)] = int(doc_id)

    adjacency: dict[int, set[int]] = {int(row[0]): set() for row in documents}
    for source_url, target_url in conn.execute(
        "SELECT source_url, target_url FROM links"
    ):
        source_id = url_to_id.get(str(source_url))
        target_id = url_to_id.get(str(target_url))
        if source_id is not None and target_id is not None and source_id != target_id:
            adjacency[source_id].add(target_id)

    node_ids = list(adjacency)
    count = len(node_ids)
    rank = {doc_id: 1.0 / count for doc_id in node_ids}
    base = (1.0 - damping) / count
    for _ in range(max(1, iterations)):
        dangling = sum(rank[node] for node in node_ids if not adjacency[node])
        updated = {
            node: base + damping * dangling / count
            for node in node_ids
        }
        for source, targets in adjacency.items():
            if not targets:
                continue
            contribution = damping * rank[source] / len(targets)
            for target in targets:
                updated[target] += contribution
        rank = updated

    with conn:
        conn.executemany(
            "UPDATE documents SET pagerank = ? WHERE id = ?",
            [(float(rank[doc_id]), doc_id) for doc_id in node_ids],
        )
        conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            ("pagerank_updated_at", utc_now()),
        )
    return count


def index_statistics(index_path: str | Path) -> dict[str, float]:
    """Return high-level statistics for the current index."""
    conn = connect(index_path)
    try:
        docs = conn.execute(
            "SELECT COUNT(*), COALESCE(AVG(token_count), 0) FROM documents"
        ).fetchone()
        terms = conn.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
        queued = conn.execute(
            "SELECT COUNT(*) FROM frontier WHERE status = 'queued'"
        ).fetchone()[0]
        links = conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]
        domains = conn.execute(
            """
            SELECT COUNT(DISTINCT
                CASE
                    WHEN instr(substr(url, instr(url, '://') + 3), '/') > 0
                    THEN substr(
                        substr(url, instr(url, '://') + 3),
                        1,
                        instr(substr(url, instr(url, '://') + 3), '/') - 1
                    )
                    ELSE substr(url, instr(url, '://') + 3)
                END
            )
            FROM documents
            """
        ).fetchone()[0]
        return {
            "documents": int(docs[0]),
            "avg_document_length": float(docs[1]),
            "unique_terms": int(terms),
            "queued_urls": int(queued),
            "link_edges": int(links),
            "domains": int(domains),
        }
    finally:
        conn.close()


def export_documents_jsonl(
    index_path: str | Path, output_path: str | Path
) -> None:
    """Export indexed documents and retrieval metadata for inspection."""
    conn = connect(index_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("w", encoding="utf-8") as file:
            rows = conn.execute(
                """
                SELECT id, url, canonical_url, title, description, text,
                       fetched_at, content_type, status_code, language,
                       topic, pagerank
                FROM documents
                ORDER BY id
                """
            )
            for row in rows:
                file.write(
                    json.dumps(
                        {
                            "id": row[0],
                            "url": row[1],
                            "canonical_url": row[2],
                            "title": row[3],
                            "description": row[4],
                            "text": row[5],
                            "fetched_at": row[6],
                            "content_type": row[7],
                            "status_code": row[8],
                            "language": row[9],
                            "topic": row[10],
                            "pagerank": row[11],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        conn.close()
