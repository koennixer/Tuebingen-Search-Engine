"""
Launcher for the Tübingen Search Engine project.

Run the friendly command shell:
    python3 tuebingen_search_engine.py

Or run one command directly:
    python3 tuebingen_search_engine.py crawl --max-pages 500
    python3 tuebingen_search_engine.py query "tübingen attractions"
    python3 tuebingen_search_engine.py batch queries.tsv results.tsv
"""

from __future__ import annotations

import sys

from tuebingen_search.cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
