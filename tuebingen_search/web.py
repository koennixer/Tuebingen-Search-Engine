"""Accessible browser interface and JSON API for the local search engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .presentation import domain_facets, intent_facets
from .retrieval import query_analysis, retrieve, retrieve_batch_list
from .storage import index_statistics

try:
    from flask import (
        Flask,
        Response,
        jsonify,
        render_template,
        request,
    )
except ImportError:
    Flask = None
    Response = None
    jsonify = None
    render_template = None
    request = None


MAX_BATCH_BYTES = 1_000_000
MAX_QUERY_CHARS = 300


def _parse_uploaded_queries(stream) -> list[tuple[str, str]]:
    """Parse and validate an assignment-style query stream."""
    queries: list[tuple[str, str]] = []
    seen: set[str] = set()
    total = 0
    for line_number, raw_line in enumerate(stream, start=1):
        total += len(raw_line)
        if total > MAX_BATCH_BYTES:
            raise ValueError("Batch file is larger than 1 MB")
        line = raw_line.decode("utf-8-sig", errors="strict").strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            query_id, text = line.split("\t", 1)
        else:
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"Line {line_number} must contain an id and query text"
                )
            query_id, text = parts
        query_id, text = query_id.strip(), text.strip()
        if not query_id or not text:
            raise ValueError(f"Line {line_number} has an empty id or query")
        if query_id in seen:
            raise ValueError(f"Duplicate query id {query_id!r}")
        if len(text) > MAX_QUERY_CHARS:
            raise ValueError(f"Query on line {line_number} is too long")
        seen.add(query_id)
        queries.append((query_id, text))
    if not queries:
        raise ValueError("The batch file contains no queries")
    return queries


def _generate_tsv_output(
    queries: list[tuple[str, str]], batch_results: dict[str, list[dict]]
) -> str:
    """Generate stable, assignment-format TSV in original query order."""
    lines: list[str] = []
    for query_id, _query_text in queries:
        for result in batch_results.get(query_id, [])[:100]:
            lines.append(
                f"{query_id}\t{int(result['rank'])}\t{result['url']}\t"
                f"{float(result['score']):.6f}"
            )
    return "\n".join(lines) + ("\n" if lines else "")


def _empty_context() -> dict[str, Any]:
    return {
        "query": "",
        "queries": [],
        "batch_results": {},
        "results": [],
        "domain_facets": [],
        "intent_facets": [],
        "analysis": {"terms": [], "intents": []},
        "tsv_output": "",
        "batch_summary": [],
        "show_scores": False,
        "error": "",
        "stats": {},
    }


def create_app(index_path: str | Path = "tuebingen_index.sqlite3"):
    """Create an isolated Flask application for tests and local use."""
    if Flask is None:
        raise RuntimeError(
            "The browser interface requires Flask. "
            "Install dependencies with: python -m pip install -r requirement.txt"
        )

    application = Flask(__name__)
    application.config.update(
        SEARCH_INDEX=str(index_path),
        MAX_CONTENT_LENGTH=MAX_BATCH_BYTES + 50_000,
    )

    @application.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store"
        return response

    @application.route("/", methods=["GET", "POST"])
    def home():
        context = _empty_context()
        try:
            context["stats"] = index_statistics(
                application.config["SEARCH_INDEX"]
            )
        except Exception:
            context["stats"] = {}

        if request.method == "GET":
            query = request.args.get("q", "").strip()
            show_scores = request.args.get("scores") == "1"
        else:
            query = request.form.get("query", "").strip()
            show_scores = request.form.get("show_scores") == "1"
        context["query"] = query
        context["show_scores"] = show_scores

        try:
            batch_file = (
                request.files.get("batch_file")
                if request.method == "POST"
                else None
            )
            if batch_file and batch_file.filename:
                queries = _parse_uploaded_queries(batch_file.stream)
                batch_results = retrieve_batch_list(
                    queries,
                    application.config["SEARCH_INDEX"],
                    top_k=100,
                )
                context["tsv_output"] = _generate_tsv_output(
                    queries, batch_results
                )
                context["queries"] = queries
                context["batch_results"] = batch_results
                context["batch_summary"] = [
                    {
                        "id": query_id,
                        "query": query_text,
                        "count": len(batch_results.get(query_id, [])),
                    }
                    for query_id, query_text in queries
                ]
            elif query:
                if len(query) > MAX_QUERY_CHARS:
                    raise ValueError("Query is too long")
                results = retrieve(
                    query,
                    application.config["SEARCH_INDEX"],
                    top_k=100,
                )
                context["results"] = results
                context["domain_facets"] = domain_facets(results)
                context["intent_facets"] = intent_facets(results)
                context["analysis"] = query_analysis(query)
        except (ValueError, UnicodeDecodeError) as exc:
            context["error"] = str(exc)
        return render_template("index.html", **context)

    @application.get("/api/search")
    def api_search():
        query = request.args.get("q", "").strip()
        if not query:
            return jsonify({"error": "Missing q parameter"}), 400
        if len(query) > MAX_QUERY_CHARS:
            return jsonify({"error": "Query is too long"}), 400
        try:
            top_k = max(1, min(int(request.args.get("top_k", "100")), 100))
        except ValueError:
            return jsonify({"error": "top_k must be an integer"}), 400
        results = retrieve(
            query, application.config["SEARCH_INDEX"], top_k=top_k
        )
        return jsonify(
            {
                "query": query,
                "analysis": query_analysis(query),
                "count": len(results),
                "results": results,
            }
        )

    @application.post("/download")
    def download():
        tsv_data = request.form.get("tsv_data", "")
        if len(tsv_data.encode("utf-8")) > 5_000_000:
            return Response("Result data is too large", status=413)
        return Response(
            tsv_data,
            mimetype="text/tab-separated-values",
            headers={
                "Content-Disposition": (
                    'attachment; filename="tuebingen-search-results.tsv"'
                )
            },
        )

    return application


app = create_app() if Flask is not None else None


def start_web_interface(
    host: str, port: int, index_path: str | Path = "tuebingen_index.sqlite3"
) -> None:
    application = create_app(index_path)
    print(f"Starting Tübingen Search on http://{host}:{port}")
    application.run(
        host=host, port=port, debug=False, use_reloader=False, threaded=True
    )
