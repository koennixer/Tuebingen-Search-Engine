# TüSearch

Python implementation for the Modern Search Engines project:

<!-- FIXME is this list complete? -->

- restartable web crawler for English Tübingen-related pages
- local SQLite document store and inverted index
- BM25 retrieval with pseudo-relevance-feedback expansion
- terminal result presentation with snippets, facets, explanations, and batch output
- web interface for searching and batch querying

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

### Commands: 

"crawl \<number>" 
-  crawl/index a specified number of pages

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

"index \<path.sqlite3>"
- switch/create the active index file
- The default index file is `.index/tuebingen_index.sqlite3`

"stats"
- shows index statistics

"batch \<query-file> \<result-file>"
- perform batch querying. Specify and input and output file

"ui"
- open the paged search interface

"web"
- start the web search interface

"quit"/"exit"/"q"
- quit the program

"help"/"?"
- show list of commands


"\<query> or "query"/"search" \<query>"
- anything that is not a command specified above gets processed as a query and retuns search results (if there are any that fit the specified query). Can also be have the word query of search in front of it, but not necessary


### Command-line search ui:
```text
ts> ui
search> food and drinks
...
Page 1/10
command> 
```
After searching, there are several commands you can use:

"quit"/"exit"/"q"
- quit the ui, go back to the base ts command line interface

"new"/"search"/"/"
- start a new search

""/"n"/"next"
- go to the next results page

"p"/"prev"/"previous"
- go to the previous results page

"facets"
- show more info about the results

"all"
- go back to result page 1

"domain \<domain>"
- show only search results of a specific domain (e.g. uni-tuebingen.de)

"explain \<rank>"
- explain/show details of a specific result by rank (e.g. explain 12)

"open \<rank>"
- open a specific result page by rank (e.g. open 42)


### Web-based ui:
```text
ts> web
```
This opens a html-based gui in your browser where you can search for things with the search bar or run batch queries by uploading a barch query file with the button on the right side of the search bar.
Results are shown as a srollable list with the read time shown to the right. With the button Show/Hide scores, the ranking scores of the individual results can be displayed. When running batch queries, the results can be viewed in the browser and downloaded as a tsv file

## Direct Commands

```bash
python3 tuebingen_search_engine.py crawl --max-pages 500
python3 tuebingen_search_engine.py crawl 500 --verbose
python3 tuebingen_search_engine.py crawl 500 --max-processed 3000
python3 tuebingen_search_engine.py query "tübingen attractions"
python3 tuebingen_search_engine.py search
python3 tuebingen_search_engine.py batch queries.tsv results.tsv
```

<!-- FIXME do we need this extra explanation, or is the one for index above sufficient and we can just append this as an additional command -->
The default index file is `.index/tuebingen_index.sqlite3`. To use another one:

```bash
python3 tuebingen_search_engine.py --index my_index.sqlite3
python3 tuebingen_search_engine.py crawl --index my_index.sqlite3 --max-pages 500
```

## Python API

```python
from tuebingen_search import crawl, retrieve, retrieve_batch, batch

crawl(["https://www.tuebingen.de/en/"], ".index/tuebingen_index.sqlite3", max_pages=500)
results = retrieve("tübingen attractions", ".index/tuebingen_index.sqlite3")
batch({"1": results}, "results.tsv")
```

## Ranking
<!-- FIXME is this still the case? Should there be more info on ranking here? -->

The first stage is a self-implemented BM25 scorer. The second stage applies
pseudo-relevance feedback query expansion. If `scikit-learn` is installed, the
top candidate pool is also reranked with a TF-IDF cosine-similarity signal.
The system still works without scikit-learn; it simply falls back to the
classical BM25 plus feedback pipeline.
