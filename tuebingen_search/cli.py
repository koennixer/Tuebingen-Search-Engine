"""Command-line entry points for crawling, querying, and batch output."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from .console import heading, success, table
from .crawler import (
    DEFAULT_CRAWL_DELAY_SECONDS,
    DEFAULT_MAX_LINKS_PER_PAGE,
    crawl as run_crawl,
)
from .presentation import (
    interactive_search,
    print_domain_facets,
    print_result_page,
    run_batch_file,
)
from .web import start_web_interface
from .retrieval import retrieve
from .storage import export_documents_jsonl, index_statistics


DEFAULT_INDEX = "tuebingen_index.sqlite3"
DEFAULT_MAX_PAGES = 50
DEFAULT_SEEDS = [
    "https://www.tuebingen.de/en/",
    #"https://www.tuebingen-info.de/en/",
    #"https://www.tuebingen-info.de/en/attractions",
    #"https://www.tuebingen-info.de/en/restaurants",
    #"https://www.tuebingen-info.de/en/events",
    #"https://www.tuebingen.de/en/3521.html",
    #"https://www.tuebingen.de/en/3773.html",
    #"https://www.tuebingen.de/en/4456.html",
    "https://www.unimuseum.uni-tuebingen.de/en/museum-at-hohentuebingen-castle",
    "https://www.komoot.com/guide/355570/castles-in-tuebingen-district",
    "https://www.outdooractive.com/en/routes/tuebingen/routes-in-tuebingen/1442519/",
    "https://www.tripadvisor.com/Tourism-g198539-Tubingen_Baden_Wurttemberg-Vacations.html",
    #"https://uni-tuebingen.de/en/",
    "https://www.germany.travel/en/",
    "https://uni-tuebingen.de/en/international/study-in-tuebingen/erasmus-and-exchange-to-tuebingen/",
    "https://www.visit-bw.com/en/",
    "https://www.mygermanyvacation.com/best-things-to-do-and-see-in-tubingen-germany/",
]


class SearchShell:
    """Small persistent shell that avoids repeatedly typing long commands."""

    def __init__(self, index: str | Path = DEFAULT_INDEX) -> None:
        self.index = str(index)

    def run(self) -> None:
        print(heading("Tübingen Search CLI"))
        print(table([("Current index", self.index)]))
        print("Type 'help' for commands. Type a normal search query to search.")

        while True:
            try:
                line = input("\nts> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return

            if not line:
                continue
            if line.lower() in {"quit", "exit", "q"}:
                return

            try:
                self.handle(line)
            except Exception as exc:
                print(f"Error: {exc}")

    def handle(self, line: str) -> None:
        parts = shlex.split(line)
        if not parts:
            return

        command = parts[0].lower()
        args = parts[1:]

        if command in {"help", "?"}:
            self.print_help()
        elif command == "index":
            self.set_index(args)
        elif command == "crawl":
            self.crawl(args)
        elif command == "stats":
            self.stats()
        elif command == "batch":
            self.batch(args)
        elif command in {"query", "search"}:
            self.query(" ".join(args))
        elif command == "ui":
            interactive_search(self.index)
        elif command == "web":
            self.web(args)
        else:
            self.query(line)

    def print_help(self) -> None:
        print(
            """
Commands
  crawl [max_pages] [seed_url ...]    crawl and index pages
  crawl 500 --verbose                 crawl with detailed progress diagnostics
  crawl 500 --no-progress             crawl without a progress display
  crawl 500 --max-processed 3000      stop after too many filtered pages
  <your query>                        search directly, e.g. tübingen attractions
  query <your query>                  same as typing the query directly
  ui                                  open the paged search interface
  web                                 start the web search interface
  batch <queries.tsv> <results.tsv>   write assignment evaluation output
  stats                               show index statistics
  index <path.sqlite3>                switch/create the active index file
  quit                                exit

Examples
  crawl 500
  food and drinks
  batch queries.tsv results.tsv
""".strip()
        )

    def set_index(self, args: list[str]) -> None:
        if not args:
            print(f"Current index: {self.index}")
            return
        self.index = args[0]
        print(f"Current index: {self.index}")

    def crawl(self, args: list[str]) -> None:
        try:
            options = parse_shell_crawl_args(args)
        except ValueError as exc:
            print(exc)
            return

        print(f"Crawling into {self.index}")
        print(f"Target: {options.max_pages} newly indexed pages")
        summary = run_crawl(
            options.seeds or DEFAULT_SEEDS,
            self.index,
            max_pages=options.max_pages,
            crawl_delay_seconds=options.delay,
            show_progress=not options.no_progress,
            progress_verbose=options.verbose,
            max_processed=options.max_processed,
            per_host_limit=options.per_host_limit,
            max_links_per_page=options.max_links_per_page,
        )
        print_crawl_summary(summary)
        print_index_statistics(self.index)

    def query(self, query_text: str) -> None:
        query_text = query_text.strip()
        if not query_text:
            print("Please enter a query.")
            return
        results = retrieve(query_text, self.index, top_k=100)
        print_domain_facets(results, limit=6)
        print_result_page(query_text, results, page_size=10)

    def batch(self, args: list[str]) -> None:
        if len(args) != 2:
            print("Use: batch <queries.tsv> <results.tsv>")
            return
        output = run_batch_file(args[0], self.index, args[1], top_k=100)
        print(success(f"Wrote {output}"))

    def stats(self) -> None:
        print_index_statistics(self.index)

    def web(self, args: list[str]) -> None:
        try:
            options = parse_shell_web_args(args)
        except ValueError as exc:
            print(exc)
            return
        start_web_interface(options.host, options.port)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the top-level command parser."""
    parser = argparse.ArgumentParser(description="Tübingen search engine project")
    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX,
        help=f"default SQLite index path for shell/search commands ({DEFAULT_INDEX})",
    )
    subparsers = parser.add_subparsers(dest="command")

    web_parser = subparsers.add_parser("web", help="start the web search interface")
    web_parser.add_argument("--host", default="127.0.0.1", help="host to bind to")
    web_parser.add_argument("--port", type=int, default=5000, help="port to bind to")

    shell_parser = subparsers.add_parser("shell", help="open the friendly command shell")
    shell_parser.add_argument("index_override", nargs="?", help="optional SQLite index path")


    crawl_parser = subparsers.add_parser("crawl", help="crawl pages and update the local index")
    crawl_parser.add_argument("--index", default=argparse.SUPPRESS, help="SQLite index path")
    crawl_parser.add_argument("seeds", nargs="*", help="seed URLs")
    crawl_parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    crawl_parser.add_argument("--delay", type=float, default=DEFAULT_CRAWL_DELAY_SECONDS)
    crawl_parser.add_argument("--max-processed", type=int)
    crawl_parser.add_argument("--per-host-limit", type=int)
    crawl_parser.add_argument("--max-links-per-page", type=int, default=DEFAULT_MAX_LINKS_PER_PAGE)
    crawl_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show queued/skipped/error counts and the current URL while crawling",
    )
    crawl_parser.add_argument("--no-progress", action="store_true")

    search_parser = subparsers.add_parser("search", help="open the paged search UI")
    search_parser.add_argument("--index", default=argparse.SUPPRESS, help="SQLite index path")
    search_parser.add_argument("--page-size", type=int, default=10)
    search_parser.add_argument("--top-k", type=int, default=100)

    query_parser = subparsers.add_parser("query", help="run one query and print result cards")
    query_parser.add_argument("--index", default=argparse.SUPPRESS, help="SQLite index path")
    query_parser.add_argument("query", nargs="+", help="query text")
    query_parser.add_argument("--top-k", type=int, default=10)

    batch_parser = subparsers.add_parser("batch", help="run batch queries and write evaluation TSV")
    batch_parser.add_argument("--index", default=argparse.SUPPRESS, help="SQLite index path")
    batch_parser.add_argument("query_file", help="tab-separated query file")
    batch_parser.add_argument("output", help="output TSV path")
    batch_parser.add_argument("--top-k", type=int, default=100)

    stats_parser = subparsers.add_parser("stats", help="print index statistics")
    stats_parser.add_argument("--index", default=argparse.SUPPRESS, help="SQLite index path")
    stats_parser.set_defaults(command="stats")

    export_parser = subparsers.add_parser("export-jsonl", help="export indexed documents as JSONL")
    export_parser.add_argument("--index", default=argparse.SUPPRESS, help="SQLite index path")
    export_parser.add_argument("output", help="output JSONL path")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        SearchShell(args.index).run()
        return 0

    if args.command == "web":
        start_web_interface(args.host, args.port)
        return 0

    if args.command == "shell":
        SearchShell(args.index_override or args.index).run()
        return 0

    if args.command == "crawl":
        max_pages, seeds = normalize_crawl_targets(args.max_pages, args.seeds)
        print(heading("Crawl"))
        print(table([("Index", args.index), ("Target", f"{max_pages} newly indexed pages")]))
        summary = run_crawl(
            seeds or DEFAULT_SEEDS,
            args.index,
            max_pages=max_pages,
            crawl_delay_seconds=args.delay,
            show_progress=not args.no_progress,
            progress_verbose=args.verbose,
            max_processed=args.max_processed,
            per_host_limit=args.per_host_limit,
            max_links_per_page=args.max_links_per_page,
        )
        print_crawl_summary(summary)
        print_index_statistics(args.index)
        return 0

    if args.command == "search":
        interactive_search(args.index, page_size=args.page_size, top_k=args.top_k)
        return 0

    if args.command == "query":
        query_text = " ".join(args.query)
        results = retrieve(query_text, args.index, top_k=args.top_k)
        print_domain_facets(results, limit=6)
        print_result_page(query_text, results, page_size=args.top_k)
        return 0

    if args.command == "batch":
        output = run_batch_file(args.query_file, args.index, args.output, top_k=args.top_k)
        print(success(f"Wrote {output}"))
        return 0

    if args.command == "stats":
        print_index_statistics(args.index)
        return 0

    if args.command == "export-jsonl":
        export_documents_jsonl(args.index, args.output)
        print(success(f"Wrote {args.output}"))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def parse_shell_web_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="web", add_help=False)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    try:
        return parser.parse_args(args)
    except SystemExit as exc:
        raise ValueError("Use: web [--host IP] [--port PORT]") from exc


def parse_shell_crawl_args(args: list[str]) -> argparse.Namespace:
    """Parse `crawl ...` inside the interactive shell.

    The shell accepts a shorthand first positional integer for max pages:
    `crawl 500 --verbose` is equivalent to `crawl --max-pages 500 --verbose`.
    """
    max_pages = DEFAULT_MAX_PAGES
    remaining = list(args)
    if remaining:
        try:
            max_pages = int(remaining[0])
            remaining = remaining[1:]
        except ValueError:
            pass

    parser = argparse.ArgumentParser(prog="crawl", add_help=False)
    parser.add_argument("seeds", nargs="*")
    parser.add_argument("--max-pages", type=int, default=max_pages)
    parser.add_argument("--delay", type=float, default=DEFAULT_CRAWL_DELAY_SECONDS)
    parser.add_argument("--max-processed", type=int)
    parser.add_argument("--per-host-limit", type=int)
    parser.add_argument("--max-links-per-page", type=int, default=DEFAULT_MAX_LINKS_PER_PAGE)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--no-progress", action="store_true")

    try:
        options = parser.parse_args(remaining)
    except SystemExit as exc:
        raise ValueError("Use: crawl [max_pages] [--verbose] [--no-progress] [seed_url ...]") from exc

    options.max_pages, options.seeds = normalize_crawl_targets(options.max_pages, options.seeds)
    return options


def normalize_crawl_targets(current_max_pages: int, seeds: list[str]) -> tuple[int, list[str]]:
    """Treat a leading numeric crawl argument as the target page count, if not explicitly set."""
    if seeds and seeds[0].isdigit():
        if current_max_pages == DEFAULT_MAX_PAGES:
            return int(seeds[0]), seeds[1:]
    return current_max_pages, seeds


def print_crawl_summary(summary: dict[str, int]) -> None:
    """Print a clear crawl summary after the progress bar finishes."""
    print(heading("Crawl Summary"))
    print(
        table(
            [
                ("Indexed", summary.get("indexed", 0)),
                ("Visited/indexed", summary.get("visited", 0)),
                ("Skipped", summary.get("skipped", 0)),
                ("Errors", summary.get("errors", 0)),
                ("Discovered URLs", summary.get("discovered", 0)),
                ("Processed cap hit", "yes" if summary.get("processed_limit_reached") else "no"),
                ("Frontier depleted", "yes" if summary.get("frontier_depleted") else "no"),
            ]
        )
    )


def print_index_statistics(index: str | Path) -> None:
    """Print current index statistics in a stable table format."""
    stats = index_statistics(index)
    print(heading("Index Statistics"))
    print(
        table(
            [
                ("Documents", stats["documents"]),
                ("Avg doc length", f"{stats['avg_document_length']:.1f} tokens"),
                ("Unique terms", stats["unique_terms"]),
                ("Queued URLs", stats["queued_urls"]),
            ]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
