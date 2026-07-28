"""SQLite persistence for documents, frontier state, and the inverted index."""

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

from .models import Document
from .text import tokenize


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(index_path: str | Path) -> sqlite3.Connection:
    """Open an index database and ensure all required tables exist."""
    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    initialize_schema(conn)
    return conn


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Create the document store, postings lists, and restartable frontier."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            text TEXT NOT NULL,
            html_gz BLOB NOT NULL,
            sha256 TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            content_type TEXT,
            status_code INTEGER,
            length INTEGER NOT NULL,
            token_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS terms (
            term TEXT PRIMARY KEY,
            document_frequency INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS postings (
            term TEXT NOT NULL,
            doc_id INTEGER NOT NULL,
            term_frequency INTEGER NOT NULL,
            PRIMARY KEY (term, doc_id),
            FOREIGN KEY (term) REFERENCES terms(term) ON DELETE CASCADE,
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS frontier (
            url TEXT PRIMARY KEY,
            status TEXT NOT NULL CHECK(status IN ('queued', 'visited', 'error', 'skipped')),
            discovered_at TEXT NOT NULL,
            last_attempt_at TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,
            priority REAL NOT NULL DEFAULT 0.0,
            error TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_frontier_status_priority
            ON frontier(status, priority DESC, discovered_at ASC);
        CREATE INDEX IF NOT EXISTS idx_postings_doc_id ON postings(doc_id);
        """
    )
    conn.commit()


def add_document(doc: Document | dict, index_path: str | Path) -> int:
    """Add or replace one document in the local inverted index."""
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
        )

    tokens = tokenize(f"{doc.title} {doc.text}")
    term_counts = Counter(tokens)
    sha256 = hashlib.sha256(doc.html.encode("utf-8", errors="ignore")).hexdigest()
    html_gz = gzip.compress(doc.html.encode("utf-8", errors="ignore"))

    with conn:
        old = conn.execute("SELECT id FROM documents WHERE url = ?", (doc.url,)).fetchone()
        if old:
            old_doc_id = old[0]
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

        cur = conn.execute(
            """
            INSERT INTO documents(
                url, title, text, html_gz, sha256, fetched_at,
                content_type, status_code, length, token_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc.url,
                doc.title,
                doc.text,
                html_gz,
                sha256,
                doc.fetched_at,
                doc.content_type,
                doc.status_code,
                len(doc.text),
                len(tokens),
            ),
        )
        doc_id = int(cur.lastrowid)

        conn.executemany(
            """
            INSERT INTO terms(term, document_frequency)
            VALUES (?, 1)
            ON CONFLICT(term) DO UPDATE
            SET document_frequency = document_frequency + 1
            """,
            [(term,) for term in term_counts.keys()],
        )
        conn.executemany(
            """
            INSERT INTO postings(term, doc_id, term_frequency)
            VALUES (?, ?, ?)
            """,
            [(term, doc_id, tf) for term, tf in term_counts.items()],
        )
    return doc_id


def add_to_frontier(conn: sqlite3.Connection, urls: Iterable[str], priority_fn) -> int:
    """Insert newly discovered URLs into the restartable crawl frontier."""
    added = 0
    now = utc_now()
    for url in urls:
        if not url:
            continue
        try:
            cur = conn.execute(
                """
                INSERT INTO frontier(url, status, discovered_at, priority)
                VALUES (?, 'queued', ?, ?)
                ON CONFLICT(url) DO NOTHING
                """,
                (url, now, priority_fn(url)),
            )
            added += cur.rowcount == 1
        except sqlite3.Error:
            continue
    conn.commit()
    return added


def next_frontier_url(
    conn: sqlite3.Connection,
    *,
    excluded_hosts: set[str] | None = None,
) -> str | None:
    """Return the highest-priority queued URL, optionally avoiding hosts."""
    cur = conn.execute(
        """
        SELECT url
        FROM frontier
        WHERE status = 'queued'
        ORDER BY priority DESC, discovered_at ASC
        """
    )
    if not excluded_hosts:
        row = cur.fetchone()
        return row[0] if row else None

    for (url,) in cur:
        if urlparse(url).netloc.lower() not in excluded_hosts:
            return url
    return None


def mark_frontier(conn: sqlite3.Connection, url: str, status: str, error: str | None = None) -> None:
    """Update the crawl status for one URL."""
    conn.execute(
        """
        UPDATE frontier
        SET status = ?,
            last_attempt_at = ?,
            attempts = attempts + 1,
            error = ?
        WHERE url = ?
        """,
        (status, utc_now(), error, url),
    )
    conn.commit()


def frontier_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Count queued, visited, skipped, and errored URLs."""
    counts = {"queued": 0, "visited": 0, "skipped": 0, "error": 0}
    rows = conn.execute("SELECT status, COUNT(*) FROM frontier GROUP BY status").fetchall()
    for status, count in rows:
        counts[str(status)] = int(count)
    return counts


def index_statistics(index_path: str | Path) -> dict[str, float]:
    """Return high-level statistics for the current index."""
    conn = connect(index_path)
    try:
        docs = conn.execute("SELECT COUNT(*), COALESCE(AVG(token_count), 0) FROM documents").fetchone()
        terms = conn.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
        queued = conn.execute("SELECT COUNT(*) FROM frontier WHERE status = 'queued'").fetchone()[0]
        return {
            "documents": int(docs[0]),
            "avg_document_length": float(docs[1]),
            "unique_terms": int(terms),
            "queued_urls": int(queued),
        }
    finally:
        conn.close()


def export_documents_jsonl(index_path: str | Path, output_path: str | Path) -> None:
    """Export indexed documents for inspection or backup."""
    conn = connect(index_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("w", encoding="utf-8") as f:
            rows = conn.execute(
                "SELECT id, url, title, text, fetched_at, content_type, status_code FROM documents ORDER BY id"
            )
            for row in rows:
                f.write(
                    json.dumps(
                        {
                            "id": row[0],
                            "url": row[1],
                            "title": row[2],
                            "text": row[3],
                            "fetched_at": row[4],
                            "content_type": row[5],
                            "status_code": row[6],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        conn.close()
