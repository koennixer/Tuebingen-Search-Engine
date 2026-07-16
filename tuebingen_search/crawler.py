"""Polite, restartable crawler for English pages related to Tübingen."""

from __future__ import annotations

import re
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

from .models import Document
from .storage import (
    add_document_to_connection,
    add_to_frontier,
    connect,
    frontier_counts,
    mark_frontier,
    next_frontier_url,
    utc_now,
)
from .text import is_probably_english, is_tuebingen_related, normalize_for_matching

try:
    import requests
except ImportError:
    requests = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


USER_AGENT = "INFO4271-TuebingenSearchBot/1.0 (+student project)"
REQUEST_TIMEOUT_SECONDS = 8
MAX_HTML_BYTES = 1_500_000
DEFAULT_CRAWL_DELAY_SECONDS = 0.5
DEFAULT_MAX_LINKS_PER_PAGE = 80


def canonicalize_url(url: str, base_url: str | None = None) -> str | None:
    """Normalize URLs so duplicates differ less often by fragments or tracking."""
    if base_url:
        url = urljoin(base_url, url)

    url, _fragment = urldefrag(url)
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.netloc:
        return None

    netloc = parsed.netloc.lower()
    if (parsed.scheme == "http" and netloc.endswith(":80")) or (
        parsed.scheme == "https" and netloc.endswith(":443")
    ):
        netloc = netloc.rsplit(":", 1)[0]

    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    ignored_query_prefixes = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid")
    query_parts = []
    for part in parsed.query.split("&"):
        if not part:
            continue
        key = part.split("=", 1)[0].lower()
        if not any(key.startswith(prefix) for prefix in ignored_query_prefixes):
            query_parts.append(part)

    return urlunparse((parsed.scheme, netloc, path, "", "&".join(query_parts), ""))


def score_url_priority(url: str) -> float:
    """Prioritize URLs that look likely to contain useful English Tübingen pages."""
    parsed = urlparse(url)
    normalized = normalize_for_matching(url)
    score = 0.0
    if "tubingen" in normalized:
        score += 5.0
    if "/en" in normalized or "lang=en" in normalized:
        score += 2.0
    if parsed.netloc.endswith(".de"):
        score += 0.5
    if any(url.lower().endswith(ext) for ext in (".pdf", ".jpg", ".png", ".zip", ".mp4", ".docx")):
        score -= 10.0
    return score

def extract_main_text(soup: BeautifulSoup, max_chars: int = 30_000) -> str:
    # Candidate containers (prefer content)
    candidates = []
    for sel in ["main", "article", '[role="main"]', "body"]:
        el = soup.select_one(sel)
        if el is not None:
            candidates.append(el)

    # If main/article includes a lot of boilerplate, pick the largest container
    if candidates:
        main = max(candidates, key=lambda el: len(el.get_text(" ", strip=True)))
    else:
        main = soup

    # Remove boilerplate inside the selected container
    for el in main.select(
        'nav, header, footer, aside, [role="navigation"], [role="contentinfo"], [aria-hidden="true"][aria-hidden="true"]'
    ):
        el.decompose()

    # Cookie/consent banners (very common English/German noise)
    for el in main.select('[id], [class]'):
        if el is not None:
            attrs = (el.get("id","") + " " + " ".join(el.get("class", []))).lower()
            if any(k in attrs for k in ["cookie", "consent", "gdpr", "privacy"]):
                el.decompose()

    text = main.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars]

def is_bad_href(href: str) -> bool:
    href = (href or "").strip().lower()
    return href.startswith(("javascript:", "mailto:", "tel:"))

def extract_document(url: str, html: str, content_type: str, status_code: int) -> tuple[Document, list[str], str | None]:
    """Extract title, body text, language hint, and outgoing links from HTML."""
    if BeautifulSoup is None:
        raise RuntimeError("Install beautifulsoup4 to parse crawled HTML: pip install beautifulsoup4")

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()

    title = ""
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        title = og["content"].strip()
    else:
        title_tag = soup.find("title")
        title = title_tag.get_text(" ", strip=True) if title_tag else ""

    text = extract_main_text(soup)

    html_tag = soup.find("html")
    html_lang = html_tag.get("lang") if html_tag else None

    links_set: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if is_bad_href(href):
            continue

        target = canonicalize_url(href, base_url=url)
        if not target:
            continue

        # canonicalize_url should ideally remove fragments and normalize queries
        # still, this protects against fragment-only links:
        parsed = urlparse(target)
        if parsed.fragment:
            continue

        links_set.add(target)

        if len(links_set) >= 50:   # hard cap per page
            break
    links = list(links_set)
    #for anchor in soup.find_all("a", href=True):
    #    target = canonicalize_url(anchor["href"], base_url=url)
    #    if target:
    #        links.append(target)

    return (
        Document(
            url=url,
            title=title,
            text=text,
            html=html,
            fetched_at=utc_now(),
            content_type=content_type,
            status_code=status_code,
        ),
        links,
        html_lang,
    )


class Politeness:
    """Apply robots.txt checks and a per-host crawl delay."""

    def __init__(self, delay_seconds: float = DEFAULT_CRAWL_DELAY_SECONDS) -> None:
        self.delay_seconds = delay_seconds
        self.last_fetch_by_host: dict[str, float] = {}
        self.robots_by_host: dict[str, RobotFileParser] = {}

    def wait(self, url: str) -> None:
        host = urlparse(url).netloc
        elapsed = time.monotonic() - self.last_fetch_by_host.get(host, 0.0)
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)

    def mark_fetched(self, url: str) -> None:
        self.last_fetch_by_host[urlparse(url).netloc] = time.monotonic()

    def allowed(self, session, url: str) -> bool:
        parsed = urlparse(url)
        host = parsed.netloc
        if host not in self.robots_by_host:
            robots_url = f"{parsed.scheme}://{host}/robots.txt"
            parser = RobotFileParser()
            parser.set_url(robots_url)
            try:
                response = session.get(robots_url, timeout=REQUEST_TIMEOUT_SECONDS)
                if response.status_code < 400:
                    parser.parse(response.text.splitlines())
                else:
                    parser.parse([])
            except Exception:
                parser.parse([])
            self.robots_by_host[host] = parser
        return self.robots_by_host[host].can_fetch(USER_AGENT, url)


class CrawlProgress:
    """Render a single-line crawl progress indicator.

    Normal mode shows only the requested progress signal: a bar plus x/y indexed.
    Verbose mode adds operational diagnostics that are useful for debugging long
    crawls, such as processed, queued, skipped, errors, and current URL.
    """

    def __init__(
        self,
        max_pages: int,
        *,
        enabled: bool = True,
        verbose: bool = False,
        interval_seconds: float = 0.8,
        stream=None,
    ) -> None:
        self.max_pages = max(max_pages, 1)
        self.enabled = enabled
        self.verbose = verbose
        self.interval_seconds = interval_seconds
        self.stream = stream or sys.stderr
        self.interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        self.last_render = 0.0
        self.last_width = 0

    def update(
        self,
        stats: dict[str, int],
        queued: int | None = None,
        current_url: str | None = None,
        *,
        force: bool = False,
    ) -> None:
        if not self.enabled:
            return

        now = time.monotonic()
        if not force and now - self.last_render < self.interval_seconds:
            return
        self.last_render = now

        indexed = stats.get("indexed", 0)
        processed = stats.get("visited", 0) + stats.get("skipped", 0) + stats.get("errors", 0)
        fraction = min(indexed / self.max_pages, 1.0)
        terminal_width = shutil.get_terminal_size((100, 24)).columns
        if self.verbose:
            bar_width = 22 if terminal_width >= 110 else 14
        else:
            bar_width = 30 if terminal_width >= 90 else 20
        filled = int(bar_width * fraction)
        bar = "#" * filled + "-" * (bar_width - filled)

        line = f"Crawling [{bar}] {indexed}/{self.max_pages} indexed"
        if self.verbose:
            queued_text = "?" if queued is None else str(queued)
            line += (
                f" | proc {processed}"
                f" | q {queued_text}"
                f" | skip {stats.get('skipped', 0)}"
                f" | err {stats.get('errors', 0)}"
                f" | new {stats.get('discovered', 0)}"
            )

        if self.verbose and current_url:
            parsed = urlparse(current_url)
            current = f"{parsed.netloc}{parsed.path}"
            remaining = max(0, terminal_width - len(line) - 8)
            if remaining:
                line += f" | now {current[:remaining]}"

        visible_width = max(20, terminal_width - 1)
        line = line[:visible_width]
        if self.interactive:
            rendered = line.ljust(max(self.last_width, len(line)))
            self.last_width = len(rendered)
            self.stream.write("\r" + rendered)
        else:
            self.stream.write(line + "\n")
        self.stream.flush()

    def done(self, stats: dict[str, int], queued: int | None = None) -> None:
        if self.enabled:
            self.update(stats, queued, force=True)
            if self.interactive:
                self.stream.write("\n")
            self.stream.flush()


def fetch_html(session, url: str) -> tuple[str, str, int]:
    """Download one HTML page while enforcing status, type, and size limits."""
    response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS, stream=True)
    status_code = response.status_code
    content_type = response.headers.get("content-type", "")

    if status_code >= 400:
        raise RuntimeError(f"HTTP {status_code}")
    if "text/html" not in content_type.lower():
        raise RuntimeError(f"not HTML: {content_type}")

    chunks = []
    total = 0
    for chunk in response.iter_content(chunk_size=8192, decode_unicode=False):
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_HTML_BYTES:
            raise RuntimeError("HTML document too large")
        chunks.append(chunk)

    encoding = response.encoding or response.apparent_encoding or "utf-8"
    html = b"".join(chunks).decode(encoding, errors="replace")
    return html, content_type, status_code


def crawl(
    frontier,
    index: str | Path,
    *,
    max_pages: int = 250,
    crawl_delay_seconds: float = DEFAULT_CRAWL_DELAY_SECONDS,
    show_progress: bool = True,
    progress_verbose: bool = False,
    max_processed: int | None = None,
    per_host_limit: int | None = None,
    max_links_per_page: int = DEFAULT_MAX_LINKS_PER_PAGE,
) -> dict[str, int]:
    """Crawl English Tübingen-related web pages into a restartable SQLite index."""
    if requests is None:
        raise RuntimeError("Install requests to crawl the web: pip install requests")

    conn = connect(index)
    seed_urls = [canonicalize_url(url) for url in frontier]
    add_to_frontier(conn, (url for url in seed_urls if url), score_url_priority)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    politeness = Politeness(delay_seconds=crawl_delay_seconds)
    max_processed = max_processed or max(max_pages * 12, max_pages + 25)
    per_host_limit = per_host_limit or max(8, max_pages // 4)
    indexed_by_host: dict[str, int] = {}

    stats = {
        "indexed": 0,
        "visited": 0,
        "skipped": 0,
        "errors": 0,
        "discovered": 0,
        "processed_limit_reached": 0,
        "frontier_depleted": 0,
    }
    progress = CrawlProgress(max_pages, enabled=show_progress, verbose=progress_verbose)

    try:
        while stats["indexed"] < max_pages:
            processed = stats["visited"] + stats["skipped"] + stats["errors"]
            if processed >= max_processed:
                stats["processed_limit_reached"] = 1
                break

            queued = frontier_counts(conn)["queued"] if progress_verbose else None
            progress.update(stats, queued)

            saturated_hosts = {
                host for host, count in indexed_by_host.items() if count >= per_host_limit
            }
            url = next_frontier_url(conn, excluded_hosts=saturated_hosts)
            if not url:
                stats["frontier_depleted"] = 1
                break

            progress.update(
                stats,
                queued,
                current_url=url,
                force=progress.interactive and progress_verbose,
            )
            try:
                if not politeness.allowed(session, url):
                    mark_frontier(conn, url, "skipped", "blocked by robots.txt")
                    stats["skipped"] += 1
                    continue

                politeness.wait(url)
                html, content_type, status_code = fetch_html(session, url)
                politeness.mark_fetched(url)

                doc, links, html_lang = extract_document(url, html, content_type, status_code)
                # 1) check relevance
                if not is_tuebingen_related(doc.url, doc.title, doc.text):
                    mark_frontier(conn, url, "skipped", "not Tübingen-related")
                    stats["skipped"] += 1
                    continue
                # 2) check language
                if not is_probably_english(doc.text, html_lang=html_lang):
                    mark_frontier(conn, url, "skipped", "not English")
                    stats["skipped"] += 1
                    continue
                # 3) only enqueue links that passed relevance and language check
                prioritized_links = sorted(set(links), key=score_url_priority, reverse=True)
                stats["discovered"] += add_to_frontier(
                    conn,
                    prioritized_links[:max_links_per_page],
                    score_url_priority,
                )
                # 4) index the page
                add_document_to_connection(conn, doc)
                mark_frontier(conn, url, "visited")
                host = urlparse(url).netloc.lower()
                indexed_by_host[host] = indexed_by_host.get(host, 0) + 1
                stats["indexed"] += 1
                stats["visited"] += 1

            except Exception as exc:
                print(exc)
                mark_frontier(conn, url, "error", str(exc)[:500])
                stats["errors"] += 1

    finally:
        queued = frontier_counts(conn)["queued"] if progress_verbose else None
        progress.done(stats, queued)
        conn.close()
        session.close()

    return stats
