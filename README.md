# Tübingen Search Engine

Python implementation for the Modern Search Engines project:

- restartable web crawler for English Tübingen-related pages
- local SQLite document store and inverted index
- BM25 retrieval with pseudo-relevance-feedback expansion
- terminal result presentation with snippets, facets, explanations, and batch output

## Install

```bash
pip install -r requirements.txt
```

## Friendly CLI

```bash
python3 tuebingen_search_engine.py
```

Inside the shell:

```text
ts> crawl 500
ts> crawl 500 --verbose
ts> crawl 500 --max-processed 3000
ts> tübingen attractions
ts> food and drinks
ts> stats
ts> batch queries.tsv results.tsv
ts> quit
```

## Direct Commands

```bash
python3 tuebingen_search_engine.py crawl --max-pages 500
python3 tuebingen_search_engine.py crawl 500 --verbose
python3 tuebingen_search_engine.py crawl 500 --max-processed 3000
python3 tuebingen_search_engine.py query "tübingen attractions"
python3 tuebingen_search_engine.py search
python3 tuebingen_search_engine.py batch queries.tsv results.tsv
```

By default, crawling shows a compact progress bar:

```text
Crawling [##########--------------------] 168/500 indexed
```

Use `--verbose` for diagnostics during long crawls:

```text
Crawling [######----------] 168/500 indexed | proc 420 | q 915 | skip 231 | err 21 | new 1202 | now www.tuebingen.de/en/...
```

The crawler is bounded by default so it does not churn forever through pages
that are filtered out as non-English or not Tübingen-related. Important options:

```bash
--max-processed 3000     stop after processing this many URLs in one run
--per-host-limit 100     avoid one website dominating a crawl run
--max-links-per-page 80  cap newly queued links per fetched page
--no-progress            suppress progress output for fragile consoles
```

The default index file is `tuebingen_index.sqlite3`. To use another one:

```bash
python3 tuebingen_search_engine.py --index my_index.sqlite3
python3 tuebingen_search_engine.py crawl --index my_index.sqlite3 --max-pages 500
```

## Python API

```python
from tuebingen_search import crawl, retrieve, retrieve_batch, batch

crawl(["https://www.tuebingen.de/en/"], "tuebingen_index.sqlite3", max_pages=500)
results = retrieve("tübingen attractions", "tuebingen_index.sqlite3")
batch({"1": results}, "results.tsv")
```

## Ranking

The first stage is a self-implemented BM25 scorer. The second stage applies
pseudo-relevance feedback query expansion. If `scikit-learn` is installed, the
top candidate pool is also reranked with a TF-IDF cosine-similarity signal.
The system still works without scikit-learn; it simply falls back to the
classical BM25 plus feedback pipeline.
