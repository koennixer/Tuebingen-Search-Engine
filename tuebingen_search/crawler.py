"""Polite, restartable crawler for English pages related to Tübingen.

Traversal and indexing are deliberately separate decisions: a German page may
be followed as a short bridge to an English page, but only relevant English
documents are stored in the searchable index.
"""

from __future__ import annotations

import re
import shutil
import sys
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

from .models import Document
from .storage import (
    add_document_to_connection,
    add_to_frontier,
    connect,
    frontier_counts,
    mark_frontier,
    next_frontier_item,
    recompute_pagerank,
    record_links,
    requeue_retryable_errors,
    utc_now,
)
from .text import detect_language, is_tuebingen_related, normalize_for_matching

try:
    import requests
except ImportError:
    requests = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


USER_AGENT = "INFO4271-TuebingenSearchBot/2.0 (+student research project)"
REQUEST_TIMEOUT_SECONDS = 15
MAX_HTML_BYTES = 2_000_000
DEFAULT_CRAWL_DELAY_SECONDS = 1.0
DEFAULT_MAX_LINKS_PER_PAGE = 100
DEFAULT_BRIDGE_DEPTH = 2
DEFAULT_MAX_DEPTH = 6
DEFAULT_OFF_TOPIC_FOLLOW_DEPTH = 1
DEFAULT_MAX_EXTERNAL_DOMAINS = 100
DEFAULT_MIN_TEXT_CHARS = 180
DEFAULT_MAX_RETRIES = 3

SKIP_EXTENSIONS = {
    ".7z", ".avi", ".css", ".doc", ".docx", ".gif", ".gz", ".ico", ".jpeg",
    ".jpg", ".js", ".json", ".mov", ".mp3", ".mp4", ".pdf", ".png", ".ppt",
    ".pptx", ".rar", ".rss", ".svg", ".tar", ".webp", ".xls", ".xlsx", ".xml",
    ".zip",
}
TRACKING_PARAMETERS = {"fbclid", "gclid", "mc_cid", "mc_eid"}
ENGLISH_LINK_HINT = re.compile(r"(^|[/_.?&=-])(en|eng|english)([/_.?&=-]|$)", re.I)
BLOCKED_DISCOVERY_HOSTS = {
    "facebook.com", "instagram.com", "linkedin.com", "pinterest.com",
    "tiktok.com", "x.com", "twitter.com", "youtube.com",
}


@dataclass(frozen=True)
class ExtractedLink:
    url: str
    anchor: str
    language_hint: str


class FetchError(RuntimeError):
    """A fetch failure carrying retry and HTTP diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.http_status = http_status


def _build_document(
    *,
    url: str,
    title: str,
    text: str,
    html: str,
    content_type: str,
    status_code: int,
    canonical_url: str,
    description: str,
) -> Document:
    """Build a document while tolerating the original non-slotted model.

    The final model declares ``canonical_url`` and ``description``. Older
    project copies did not. Attaching those values after construction prevents
    a partially updated checkout from failing inside HTML extraction.
    """
    required = {
        "url": url,
        "title": title,
        "text": text,
        "html": html,
        "fetched_at": utc_now(),
        "content_type": content_type,
        "status_code": status_code,
    }
    metadata = {
        "canonical_url": canonical_url,
        "description": description,
    }
    declared_fields = set(getattr(Document, "__dataclass_fields__", {}))
    constructor_values = dict(required)
    constructor_values.update(
        {
            name: value
            for name, value in metadata.items()
            if not declared_fields or name in declared_fields
        }
    )
    document = Document(**constructor_values)
    for name, value in metadata.items():
        if name in declared_fields:
            continue
        try:
            object.__setattr__(document, name, value)
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                "The crawler and models.py are from different project "
                "versions. Replace tuebingen_search/models.py with the "
                "version supplied alongside this crawler."
            ) from exc
    return document


def _set_document_metadata(document: Document, **values: str) -> Document:
    """Attach crawl metadata, including to the original frozen model."""
    for name, value in values.items():
        try:
            object.__setattr__(document, name, value)
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                "The crawler and models.py are from different project "
                "versions. Replace tuebingen_search/models.py with the "
                "version supplied alongside this crawler."
            ) from exc
    return document


def _tag_attribute(tag, name: str, default=None):
    """Read a Beautiful Soup attribute even after a tag was decomposed."""
    attributes = getattr(tag, "attrs", None)
    if attributes is None or not hasattr(attributes, "get"):
        return default
    value = attributes.get(name, default)
    return default if value is None else value


class _FallbackHTMLParser(HTMLParser):
    """Small dependency-free extractor used when Beautiful Soup is absent."""

    ignored_tags = {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "nav",
        "header",
        "footer",
        "aside",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool]] = []
        self.ignored_depth = 0
        self.in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.html_lang: str | None = None
        self.description = ""
        self.canonical = ""
        self.links: list[tuple[str, str, str]] = []
        self.anchor_href = ""
        self.anchor_hreflang = ""
        self.anchor_parts: list[str] = []
        self.base_href = ""

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = {key.lower(): value or "" for key, value in attrs}
        ignored = tag in self.ignored_tags
        self.stack.append((tag, ignored))
        if ignored:
            self.ignored_depth += 1
        if tag == "html":
            self.html_lang = attributes.get("lang") or None
        elif tag == "title":
            self.in_title = True
        elif tag == "meta":
            name = attributes.get("name", "").lower()
            prop = attributes.get("property", "").lower()
            if name == "description" or prop == "og:description":
                self.description = attributes.get("content", "")[:1_000]
            elif prop == "og:title" and attributes.get("content"):
                self.title_parts = [attributes["content"]]
        elif tag == "link" and "canonical" in attributes.get("rel", "").lower():
            self.canonical = attributes.get("href", "")
        elif tag == "base" and attributes.get("href"):
            self.base_href = attributes["href"]
        elif tag == "a":
            self.anchor_href = attributes.get("href", "")
            self.anchor_hreflang = attributes.get("hreflang", "")
            self.anchor_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        elif tag == "a" and self.anchor_href:
            self.links.append(
                (
                    self.anchor_href,
                    " ".join(self.anchor_parts).strip(),
                    self.anchor_hreflang,
                )
            )
            self.anchor_href = ""
            self.anchor_hreflang = ""
            self.anchor_parts = []
        for index in range(len(self.stack) - 1, -1, -1):
            open_tag, ignored = self.stack[index]
            if open_tag == tag:
                del self.stack[index:]
                if ignored:
                    self.ignored_depth = max(0, self.ignored_depth - 1)
                break

    def handle_data(self, data: str) -> None:
        cleaned = re.sub(r"\s+", " ", data).strip()
        if not cleaned:
            return
        if self.in_title:
            self.title_parts.append(cleaned)
        if self.anchor_href:
            self.anchor_parts.append(cleaned)
        if self.ignored_depth == 0 and not self.in_title:
            self.text_parts.append(cleaned)


def _extract_document_without_bs4(
    url: str, html: str, content_type: str, status_code: int
) -> tuple[Document, list[ExtractedLink], str | None]:
    parser = _FallbackHTMLParser()
    parser.feed(html)
    links: dict[str, ExtractedLink] = {}
    page_base = parser.base_href or url
    for raw_url, anchor, hreflang in parser.links:
        target = canonicalize_url(raw_url, base_url=page_base)
        if not target:
            continue
        language_hint = hreflang.lower().split("-", 1)[0]
        links.setdefault(
            target,
            ExtractedLink(target, anchor[:250], language_hint),
        )
    canonical_url = (
        canonicalize_url(parser.canonical, base_url=page_base)
        if parser.canonical
        else page_base
    ) or page_base
    text = re.sub(r"\s+", " ", " ".join(parser.text_parts)).strip()[:40_000]
    return (
        _build_document(
            url=url,
            title=" ".join(parser.title_parts).strip(),
            text=text,
            html=html,
            content_type=content_type,
            status_code=status_code,
            canonical_url=canonical_url,
            description=parser.description,
        ),
        list(links.values()),
        parser.html_lang,
    )


def canonicalize_url(url: str, base_url: str | None = None) -> str | None:
    """Resolve and normalize an HTTP URL while removing tracking noise."""
    if base_url:
        url = urljoin(base_url, url)
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return None
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.hostname:
        return None
    host = parsed.hostname.lower().rstrip(".")
    netloc = host
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        netloc = f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in TRACKING_PARAMETERS
    ]
    suffix = Path(path).suffix.lower()
    if suffix in SKIP_EXTENSIONS:
        return None
    return urlunsplit((scheme, netloc, path, urlencode(sorted(query)), ""))


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def _is_host_or_subdomain(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def score_url_priority(url: str) -> float:
    """Prioritize likely English Tübingen pages and penalize crawl traps."""
    normalized = normalize_for_matching(url)
    score = 0.0
    if "tubingen" in normalized:
        score += 5.0
    if ENGLISH_LINK_HINT.search(url):
        score += 3.0
    if any(token in normalized for token in ("restaurant", "hotel", "museum", "visit")):
        score += 1.0
    if len(urlsplit(url).query) > 120:
        score -= 2.0
    if any(
        token in normalized
        for token in ("calendar", "login", "share=", "replytocom")
    ):
        score -= 3.0
    return score


def extract_main_text(soup: BeautifulSoup, max_chars: int = 40_000) -> str:
    """Extract useful page text while dropping navigation and consent noise."""
    root = (
        soup.select_one("main")
        or soup.select_one("article")
        or soup.select_one('[role="main"]')
        or soup.body
        or soup
    )
    for element in root.select(
        "script, style, noscript, template, svg, nav, header, footer, aside, "
        '[role="navigation"], [role="contentinfo"], [aria-hidden="true"]'
    ):
        element.decompose()
    for element in list(root.select("[id], [class]")):
        if getattr(element, "attrs", None) is None:
            continue
        classes = _tag_attribute(element, "class", [])
        if isinstance(classes, str):
            class_text = classes
        else:
            class_text = " ".join(str(value) for value in classes)
        attrs = (
            str(_tag_attribute(element, "id", "")) + " " + class_text
        ).lower()
        if any(word in attrs for word in ("cookie", "consent", "gdpr")):
            element.decompose()
    return re.sub(r"\s+", " ", root.get_text(" ", strip=True)).strip()[:max_chars]


def extract_document(
    url: str, html: str, content_type: str, status_code: int
) -> tuple[Document, list[ExtractedLink], str | None]:
    """Extract a document, language metadata, and annotated outgoing links."""
    if BeautifulSoup is None:
        return _extract_document_without_bs4(
            url, html, content_type, status_code
        )
    soup = BeautifulSoup(html, "html.parser")
    base_tag = soup.find("base", href=True)
    page_base = str(base_tag["href"]).strip() if base_tag else url
    title_tag = soup.find("meta", property="og:title")
    title = (
        str(_tag_attribute(title_tag, "content", "")).strip()
        if title_tag is not None
        else (soup.title.get_text(" ", strip=True) if soup.title else "")
    )
    description_tag = (
        soup.select_one('meta[name="description"]')
        or soup.select_one('meta[property="og:description"]')
    )
    description = (
        str(_tag_attribute(description_tag, "content", "")).strip()[:1_000]
        if description_tag is not None
        else ""
    )
    canonical_tag = soup.select_one('link[rel~="canonical"][href]')
    canonical_url = (
        canonicalize_url(
            str(_tag_attribute(canonical_tag, "href", "")), base_url=page_base
        )
        if canonical_tag is not None
        else page_base
    ) or page_base
    text = extract_main_text(soup)
    html_tag = soup.find("html")
    html_lang = (
        str(_tag_attribute(html_tag, "lang", "")) or None
        if html_tag is not None
        else None
    )

    links: dict[str, ExtractedLink] = {}
    for anchor in soup.select("a[href]"):
        raw = str(_tag_attribute(anchor, "href", "")).strip()
        if not raw or raw.lower().startswith(
            ("javascript:", "mailto:", "tel:", "data:", "#")
        ):
            continue
        target = canonicalize_url(raw, base_url=page_base)
        if not target:
            continue
        label = re.sub(r"\s+", " ", anchor.get_text(" ", strip=True))[:250]
        hreflang = (
            str(_tag_attribute(anchor, "hreflang", ""))
            .lower()
            .split("-", 1)[0]
        )
        links.setdefault(target, ExtractedLink(target, label, hreflang))

    return (
        _build_document(
            url=url,
            title=title,
            text=text,
            html=html,
            content_type=content_type,
            status_code=status_code,
            canonical_url=canonical_url,
            description=description,
        ),
        list(links.values()),
        html_lang,
    )


class CrawlScope:
    """Bound discovery without whitelisting an entire external website."""

    def __init__(
        self,
        seeds: list[str],
        *,
        allow_external: bool,
        max_external_domains: int,
    ) -> None:
        self.seed_hosts = {_host(url) for url in seeds}
        self.allow_external = allow_external
        self.max_external_domains = max_external_domains
        self.admitted_hosts: set[str] = set()

    @staticmethod
    def _matches(host: str, domains: set[str]) -> bool:
        return any(
            _is_host_or_subdomain(host, domain)
            for domain in domains
        )

    def is_seed_host(self, host: str) -> bool:
        return self._matches(host, self.seed_hosts)

    def is_admitted_host(self, host: str) -> bool:
        return self._matches(host, self.admitted_hosts)

    def contains(self, host: str) -> bool:
        return self.is_seed_host(host) or self.is_admitted_host(host)

    def admit_link(
        self,
        link: ExtractedLink,
        *,
        source_host: str,
        source_relevant: bool,
        source_english: bool,
        source_depth: int,
    ) -> bool:
        target_host = _host(link.url)
        if not target_host:
            return False

        # Seed websites retain their normal bounded traversal behavior.
        if self.is_seed_host(target_host):
            return True

        link_relevant = is_tuebingen_related(
            link.url,
            link.anchor,
            "",
        )

        english_hint = (
            link.language_hint == "en"
            or bool(ENGLISH_LINK_HINT.search(f"{link.url} {link.anchor}"))
        )

        # An admitted external host is not completely whitelisted.
        if self.is_admitted_host(target_host):
            return link_relevant or (
                source_relevant
                and not source_english
                and english_hint
            )

        # Only a relevant page on a supplied seed host may introduce
        # another external domain.
        if (
            not self.allow_external
            or not self.is_seed_host(source_host)
            or not source_relevant
            or source_depth > 2
            or len(self.admitted_hosts) >= self.max_external_domains
            or any(
                _is_host_or_subdomain(target_host, blocked)
                for blocked in BLOCKED_DISCOVERY_HOSTS
            )
        ):
            return False

        # Admit this first linked URL. Further links on the same domain
        # must satisfy the checks above.
        self.admitted_hosts.add(target_host)
        return True


class Politeness:
    """Apply robots.txt checks and a per-host request delay."""

    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = max(0.0, delay_seconds)
        self.last_fetch_by_host: dict[str, float] = {}
        self.robots_by_origin: dict[str, RobotFileParser] = {}

    def wait(self, url: str) -> None:
        host = _host(url)
        elapsed = time.monotonic() - self.last_fetch_by_host.get(host, 0.0)
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)

    def mark_fetched(self, url: str) -> None:
        self.last_fetch_by_host[_host(url)] = time.monotonic()

    def allowed(self, session, url: str) -> bool:
        parsed = urlsplit(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self.robots_by_origin:
            parser = RobotFileParser()
            parser.set_url(origin + "/robots.txt")
            try:
                response = session.get(parser.url, timeout=REQUEST_TIMEOUT_SECONDS)
                if response.status_code in {401, 403}:
                    parser.parse(["User-agent: *", "Disallow: /"])
                elif response.status_code < 400:
                    parser.parse(response.text.splitlines())
                else:
                    parser.parse([])
            except requests.RequestException:
                parser.parse([])
            self.robots_by_origin[origin] = parser
        return self.robots_by_origin[origin].can_fetch(USER_AGENT, url)


class CrawlProgress:
    def __init__(self, max_pages: int, *, enabled: bool, verbose: bool) -> None:
        self.max_pages = max(max_pages, 1)
        self.enabled = enabled
        self.verbose = verbose
        self.stream = sys.stderr
        self.interactive = self.stream.isatty()
        self.last_render = 0.0
        self.last_width = 0

    def update(
        self, stats: dict[str, int], queued: int | None = None,
        current_url: str | None = None, *, force: bool = False
    ) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self.last_render < 0.8:
            return
        self.last_render = now
        width = shutil.get_terminal_size((100, 24)).columns
        bar_width = 22
        filled = int(bar_width * min(stats["indexed"] / self.max_pages, 1.0))
        line = (
            f"Crawling [{'#' * filled}{'-' * (bar_width - filled)}] "
            f"{stats['indexed']}/{self.max_pages} indexed"
        )
        if self.verbose:
            line += (
                f" | fetched {stats['fetched']} | bridge {stats['bridges']} "
                f"| skip {stats['skipped']} | err {stats['errors']} | q {queued or 0}"
            )
            if current_url:
                line += f" | {_host(current_url)}"
        line = line[: max(20, width - 1)]
        if self.interactive:
            self.stream.write("\r" + line.ljust(max(self.last_width, len(line))))
            self.last_width = max(self.last_width, len(line))
        else:
            self.stream.write(line + "\n")
        self.stream.flush()

    def done(self, stats: dict[str, int], queued: int | None = None) -> None:
        self.update(stats, queued, force=True)
        if self.enabled and self.interactive:
            self.stream.write("\n")


def fetch_html(session, url: str) -> tuple[str, str, int, str]:
    """Download one bounded HTML response and return its canonical final URL."""
    response = None
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS, stream=True)
        if response.status_code >= 400:
            retryable = response.status_code == 429 or response.status_code >= 500
            raise FetchError(
                f"HTTP {response.status_code}",
                retryable=retryable,
                http_status=response.status_code,
            )
        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type.lower():
            raise FetchError(f"not HTML: {content_type}")
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=8192):
            total += len(chunk)
            if total > MAX_HTML_BYTES:
                raise FetchError("HTML document too large")
            chunks.append(chunk)
        encoding = response.encoding or response.apparent_encoding or "utf-8"
        final_url = canonicalize_url(response.url) or url
        return (
            b"".join(chunks).decode(encoding, errors="replace"),
            content_type,
            response.status_code,
            final_url,
        )
    except requests.RequestException as exc:
        raise FetchError(str(exc), retryable=True) from exc
    finally:
        if response is not None:
            response.close()


def _should_follow(
    link: ExtractedLink,
    *,
    page_english: bool,
    page_relevant: bool,
    context_relevant: bool,
    depth: int,
    next_bridge_steps: int,
    bridge_depth: int,
    off_topic_follow_depth: int,
) -> bool:
    english_hint = link.language_hint == "en" or bool(
        ENGLISH_LINK_HINT.search(f"{link.url} {link.anchor}")
    )
    language_ok = page_english or english_hint or next_bridge_steps <= bridge_depth
    topic_ok = (
        page_relevant
        or context_relevant
        or depth < off_topic_follow_depth
        or is_tuebingen_related(link.url, link.anchor, "")
    )
    return language_ok and topic_ok


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
    bridge_depth: int = DEFAULT_BRIDGE_DEPTH,
    max_depth: int = DEFAULT_MAX_DEPTH,
    off_topic_follow_depth: int = DEFAULT_OFF_TOPIC_FOLLOW_DEPTH,
    allow_external: bool = True,
    max_external_domains: int = DEFAULT_MAX_EXTERNAL_DOMAINS,
    retry_errors: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
    min_text_chars: int = DEFAULT_MIN_TEXT_CHARS,
) -> dict[str, int]:
    """Crawl candidate pages and index only relevant English documents."""
    if requests is None:
        raise RuntimeError("Install requests to crawl the web")
    seeds = [url for raw in frontier if (url := canonicalize_url(raw))]
    if not seeds:
        raise ValueError("At least one valid HTTP(S) seed URL is required")

    conn = connect(index)
    add_to_frontier(conn, seeds, score_url_priority, context_relevant=True)
    retried = (
        requeue_retryable_errors(conn, max_attempts=max_retries)
        if retry_errors
        else 0
    )
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    politeness = Politeness(crawl_delay_seconds)
    scope = CrawlScope(
        seeds, allow_external=allow_external,
        max_external_domains=max_external_domains,
    )
    # Preserve externally discovered hosts across interrupted runs.
    for (queued_url,) in conn.execute(
        "SELECT url FROM frontier WHERE source_host IS NOT NULL"
    ):
        host = _host(str(queued_url))
        if host and not scope.contains(host):
            scope.admitted_hosts.add(host)
    max_processed = max_processed or max(max_pages * 12, max_pages + 25)
    per_host_limit = per_host_limit or max(8, max_pages // 4)
    indexed_by_host: dict[str, int] = {}
    for (document_url,) in conn.execute("SELECT url FROM documents"):
        host = _host(document_url)
        indexed_by_host[host] = indexed_by_host.get(host, 0) + 1

    stats = {
        "indexed": 0, "visited": 0, "attempted": 0, "fetched": 0, "bridges": 0,
        "skipped": 0, "errors": 0, "discovered": 0,
        "processed_limit_reached": 0, "frontier_depleted": 0,
        "retried": retried, "authority_nodes": 0,
    }
    progress = CrawlProgress(
        max_pages, enabled=show_progress, verbose=progress_verbose
    )
    try:
        while stats["indexed"] < max_pages:
            processed = stats["attempted"]
            if processed >= max_processed:
                stats["processed_limit_reached"] = 1
                break
            saturated = {
                host for host, count in indexed_by_host.items()
                if count >= per_host_limit
            }
            item = next_frontier_item(conn, excluded_hosts=saturated)
            if not item:
                stats["frontier_depleted"] = 1
                break
            url = str(item["url"])
            queued = frontier_counts(conn)["queued"] if progress_verbose else None
            progress.update(stats, queued, url)
            try:
                if not politeness.allowed(session, url):
                    mark_frontier(conn, url, "skipped", "blocked by robots.txt")
                    stats["skipped"] += 1
                    continue
                politeness.wait(url)
                stats["attempted"] += 1
                html, content_type, status_code, final_url = fetch_html(session, url)
                politeness.mark_fetched(url)
                stats["fetched"] += 1
                document, links, html_lang = extract_document(
                    final_url, html, content_type, status_code
                )
                language, language_confidence = detect_language(
                    document.text, html_lang
                )
                page_english = language == "en" and language_confidence >= 0.70
                document = _set_document_metadata(document, language=language)
                page_relevant = is_tuebingen_related(
                    document.url, document.title, document.text
                )
                record_links(
                    conn,
                    final_url,
                    ((link.url, link.anchor) for link in links),
                )

                depth = int(item["depth"])
                if depth < max_depth:
                    next_bridge = 0 if page_english else int(item["bridge_steps"]) + 1
                    candidates = sorted(
                        links,
                        key=lambda link: score_url_priority(link.url),
                        reverse=True,
                    )[:max_links_per_page]
                    for link in candidates:
                        if not _should_follow(
                            link,
                            page_english=page_english,
                            page_relevant=page_relevant,
                            context_relevant=bool(item["context_relevant"]),
                            depth=depth,
                            next_bridge_steps=next_bridge,
                            bridge_depth=bridge_depth,
                            off_topic_follow_depth=off_topic_follow_depth,
                        ):
                            continue
                        if not scope.admit_link(
                            link,
                            source_host=_host(final_url),
                            source_relevant=page_relevant,
                            source_english=page_english,
                            source_depth=depth,
                        ):
                           continue
                        stats["discovered"] += add_to_frontier(
                            conn,
                            [link.url],
                            score_url_priority,
                            depth=depth + 1,
                            bridge_steps=next_bridge,
                            context_relevant=page_relevant,
                            source_host=_host(final_url),
                        )

                if (
                    page_english
                    and page_relevant
                    and len(document.text) >= max(0, min_text_chars)
                ):
                    add_document_to_connection(conn, document)
                    mark_frontier(conn, url, "visited")
                    host = _host(final_url)
                    indexed_by_host[host] = indexed_by_host.get(host, 0) + 1
                    stats["indexed"] += 1
                    stats["visited"] += 1
                else:
                    reason = (
                        f"traversed only: language={language} "
                        f"confidence={language_confidence:.2f} "
                        f"relevant={page_relevant} "
                        f"text_chars={len(document.text)}"
                    )
                    mark_frontier(conn, url, "skipped", reason)
                    stats["skipped"] += 1
                    if not page_english:
                        stats["bridges"] += 1
            except Exception as exc:
                retryable = bool(getattr(exc, "retryable", False))
                attempt_number = int(item["attempts"]) + 1
                retry_now = retryable and attempt_number < max(1, max_retries)
                mark_frontier(
                    conn,
                    url,
                    "queued" if retry_now else "error",
                    str(exc)[:500],
                    retryable=retryable,
                    http_status=getattr(exc, "http_status", None),
                    priority_penalty=(
                        1.5 * attempt_number if retry_now else 0.0
                    ),
                )
                if retry_now:
                    stats["retried"] += 1
                else:
                    stats["errors"] += 1
                if progress_verbose:
                    suffix = (
                        f" (retry {attempt_number + 1}/{max_retries} queued)"
                        if retry_now
                        else ""
                    )
                    print(f"\n{url}: {exc}{suffix}", file=sys.stderr)
    finally:
        stats["authority_nodes"] = recompute_pagerank(conn)
        queued = frontier_counts(conn)["queued"] if progress_verbose else None
        progress.done(stats, queued)
        conn.close()
        session.close()
    return stats
