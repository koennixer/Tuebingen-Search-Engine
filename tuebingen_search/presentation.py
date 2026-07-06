"""Terminal presentation and batch-output helpers for search results."""

from __future__ import annotations

import math
import textwrap
import webbrowser
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from .console import rule, subheading, terminal_width
from .models import SearchResult
from .retrieval import retrieve, retrieve_batch
from .text import query_terms, tokenize


ResultLike = dict[str, int | float | str] | SearchResult


def _result_value(result: ResultLike, key: str, default: object = "") -> object:
    if isinstance(result, SearchResult):
        return getattr(result, key, default)
    return result.get(key, default)


def result_domain(result: ResultLike) -> str:
    url = str(_result_value(result, "url", ""))
    return urlparse(url).netloc.lower() or "(unknown)"


def domain_facets(results: list[ResultLike]) -> list[tuple[str, int, float]]:
    """Group results by source domain for faceted browsing."""
    counts: Counter[str] = Counter()
    best_scores: dict[str, float] = {}
    for result in results:
        domain = result_domain(result)
        score = float(_result_value(result, "score", 0.0) or 0.0)
        counts[domain] += 1
        best_scores[domain] = max(best_scores.get(domain, 0.0), score)

    return sorted(
        ((domain, count, best_scores[domain]) for domain, count in counts.items()),
        key=lambda item: (-item[1], -item[2], item[0]),
    )


def matched_query_terms(query: str, result: ResultLike) -> list[str]:
    """Return query terms visible in a result title, URL, or snippet."""
    haystack = tokenize(
        " ".join(
            [
                str(_result_value(result, "title", "")),
                str(_result_value(result, "snippet", "")),
                str(_result_value(result, "url", "")),
            ]
        )
    )
    haystack_set = set(haystack)
    return [term for term in query_terms(query) if term in haystack_set]


def format_result_card(result: ResultLike, query: str = "", *, width: int | None = None) -> str:
    """Format one result as a compact terminal card."""
    width = width or terminal_width()
    rank = int(_result_value(result, "rank", 0) or 0)
    score = float(_result_value(result, "score", 0.0) or 0.0)
    title = str(_result_value(result, "title", "") or "(untitled)")
    url = str(_result_value(result, "url", ""))
    result_snippet = str(_result_value(result, "snippet", ""))
    domain = result_domain(result)
    terms = matched_query_terms(query, result) if query else []

    header = f"{rank:>3}. {title}"
    meta = f"score {score:.4f} | source {domain}"
    if terms:
        meta += " | matched " + ", ".join(terms)

    lines = [
        textwrap.shorten(header, width=width, placeholder="..."),
        "     " + textwrap.shorten(url, width=max(20, width - 5), placeholder="..."),
        "     " + textwrap.shorten(meta, width=max(20, width - 5), placeholder="..."),
    ]
    if result_snippet:
        lines.extend(
            textwrap.wrap(
                result_snippet,
                width=width,
                initial_indent="     ",
                subsequent_indent="     ",
            )
        )
    return "\n".join(lines)


def print_result_page(
    query: str,
    results: list[ResultLike],
    *,
    page: int = 1,
    page_size: int = 10,
    domain_filter: str | None = None,
) -> None:
    """Print one page of result cards, optionally filtered by domain."""
    visible = [
        result
        for result in results
        if domain_filter is None or result_domain(result) == domain_filter
    ]
    total_pages = max(1, math.ceil(len(visible) / page_size))
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = start + page_size
    width = terminal_width()

    print(subheading(f"Results for: {query}"))
    if domain_filter:
        print(f"Filter: {domain_filter}")
    print(f"Showing {start + 1 if visible else 0}-{min(end, len(visible))} of {len(visible)} results")
    print(rule("-"))
    for result in visible[start:end]:
        print(format_result_card(result, query, width=width))
        print(rule("-"))

    if not visible:
        print("No results for this query/filter.")
    print(f"Page {page}/{total_pages}")


def print_domain_facets(results: list[ResultLike], *, limit: int = 12) -> None:
    """Print the most frequent result domains."""
    facets = domain_facets(results)[:limit]
    if not facets:
        print("No domain facets available.")
        return

    print(subheading("Source Facets"))
    for domain, count, best_score in facets:
        print(f"  {domain:<38} {count:>3} results   best {best_score:.4f}")


def explain_result(query: str, result: ResultLike) -> str:
    """Explain the score components shown for one result."""
    rank = int(_result_value(result, "rank", 0) or 0)
    bm25_score = float(_result_value(result, "bm25_score", 0.0) or 0.0)
    expanded_score = float(_result_value(result, "expanded_score", 0.0) or 0.0)
    final_score = float(_result_value(result, "score", 0.0) or 0.0)
    matches = matched_query_terms(query, result)
    return "\n".join(
        [
            f"Explanation for rank {rank}",
            f"  final score:    {final_score:.6f}",
            f"  BM25 score:     {bm25_score:.6f}",
            f"  expanded score: {expanded_score:.6f}",
            f"  matched terms:  {', '.join(matches) if matches else '(none shown in title/snippet/url)'}",
            f"  URL:            {_result_value(result, 'url', '')}",
        ]
    )


def interactive_search(index: str | Path, *, page_size: int = 10, top_k: int = 100) -> None:
    """Open the paged interactive search interface."""
    print(subheading("Tübingen Search"))
    print("Type a query, or 'quit' to exit.")

    while True:
        query = input("\nsearch> ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            return
        if not query:
            continue

        results = retrieve(query, index, top_k=top_k)
        page = 1
        domain_filter: str | None = None
        print_domain_facets(results, limit=6)
        print_result_page(query, results, page=page, page_size=page_size)

        by_rank = {int(_result_value(result, "rank", 0) or 0): result for result in results}

        while True:
            command = input("command> ").strip()
            lowered = command.lower()

            if lowered in {"quit", "exit", "q"}:
                return
            if lowered in {"new", "search", "/"}:
                break
            if lowered in {"", "n", "next"}:
                page += 1
                print_result_page(query, results, page=page, page_size=page_size, domain_filter=domain_filter)
                continue
            if lowered in {"p", "prev", "previous"}:
                page -= 1
                print_result_page(query, results, page=page, page_size=page_size, domain_filter=domain_filter)
                continue
            if lowered == "facets":
                print_domain_facets(results)
                continue
            if lowered == "all":
                domain_filter = None
                page = 1
                print_result_page(query, results, page=page, page_size=page_size)
                continue
            if lowered.startswith("domain "):
                domain_filter = command.split(maxsplit=1)[1].strip().lower()
                page = 1
                print_result_page(query, results, page=page, page_size=page_size, domain_filter=domain_filter)
                continue
            if lowered.startswith("explain "):
                try:
                    rank = int(command.split(maxsplit=1)[1])
                    print(explain_result(query, by_rank[rank]))
                except (ValueError, KeyError):
                    print("Use: explain <rank>, for example: explain 3")
                continue
            if lowered.startswith("open "):
                try:
                    rank = int(command.split(maxsplit=1)[1])
                    webbrowser.open(str(_result_value(by_rank[rank], "url", "")))
                except (ValueError, KeyError):
                    print("Use: open <rank>, for example: open 3")
                continue

            print("Commands: n, p, facets, domain <host>, all, explain <rank>, open <rank>, new, quit")


def batch(
    results: dict[str, list[ResultLike]] | list[tuple[str, list[ResultLike]]],
    output_path: str | Path = "search_results.tsv",
) -> Path:
    """Write results as `query_id<TAB>rank<TAB>url<TAB>score`."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    items = results.items() if isinstance(results, dict) else results
    with output.open("w", encoding="utf-8", newline="\n") as f:
        for query_id, query_results in items:
            for fallback_rank, result in enumerate(query_results[:100], start=1):
                rank = int(_result_value(result, "rank", fallback_rank) or fallback_rank)
                url = str(_result_value(result, "url", ""))
                score = float(_result_value(result, "score", 0.0) or 0.0)
                f.write(f"{query_id}\t{rank}\t{url}\t{score:.6f}\n")

    return output


def run_batch_file(
    query_file: str | Path,
    index: str | Path,
    output_path: str | Path,
    *,
    top_k: int = 100,
) -> Path:
    """Retrieve a batch query file and write assignment-format output."""
    results = retrieve_batch(query_file, index, top_k=top_k)
    return batch(results, output_path)
