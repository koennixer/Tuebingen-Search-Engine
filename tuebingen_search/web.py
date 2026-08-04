"""Accessible browser interface and JSON API for the local search engine."""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any
import webbrowser

from .presentation import domain_facets, intent_facets
from .retrieval import query_analysis, retrieve, retrieve_batch_list, suggest_correction
from .storage import DEFAULT_INDEX_PATH, index_statistics
from .text import query_terms

try:
    from flask import (
        Flask,
        Response,
        jsonify,
        render_template,
        request,
        redirect,
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
        "batch_typos": {},
        "results": [],
        "domain_facets": [],
        "intent_facets": [],
        "analysis": {"terms": [], "intents": []},
        "tsv_output": "",
        "batch_summary": [],
        "show_scores": False,
        "error": "",
        "stats": {},
        "suggestion": None,
    }


def _handle_batch_review_submission(request, context: dict[str, Any], index_path: str | Path):
    query_ids = request.form.getlist("batch_query_ids")
    query_texts = request.form.getlist("batch_query_texts")
    queries = list(zip(query_ids, query_texts))
    
    keep_all_original = request.form.get("keep_all_original") == "1"
    
    updated_queries = []
    for query_id, query_text in queries:
        if keep_all_original or request.form.get(f"keep_original_{query_id}") == "1":
            updated_queries.append((query_id, query_text))
        else:
            corrected_text = request.form.get(f"correction_{query_id}")
            if corrected_text and corrected_text.strip():
                updated_queries.append((query_id, corrected_text.strip()))
            else:
                updated_queries.append((query_id, query_text))
    
    batch_results = retrieve_batch_list(updated_queries, index_path, top_k=100)
    context["tsv_output"] = _generate_tsv_output(updated_queries, batch_results)
    context["queries"] = updated_queries
    context["batch_results"] = batch_results
    context["batch_summary"] = [
        {"id": qid, "query": qtext, "count": len(batch_results.get(qid, []))}
        for qid, qtext in updated_queries
    ]


def _handle_batch_upload(batch_file, context: dict[str, Any], index_path: str | Path):
    queries = _parse_uploaded_queries(batch_file.stream)
    batch_results = retrieve_batch_list(queries, index_path, top_k=100)
    
    batch_typos = {}
    for query_id, query_text in queries:
        suggestion = suggest_correction(query_text, index_path)
        if suggestion:
            batch_typos[query_id] = {"original": query_text, "suggestion": suggestion}
    
    if batch_typos:
        context["batch_typos"] = batch_typos
        context["queries"] = queries
    else:
        context["tsv_output"] = _generate_tsv_output(queries, batch_results)
        context["queries"] = queries
        context["batch_results"] = batch_results
        context["batch_summary"] = [
            {"id": qid, "query": qtext, "count": len(batch_results.get(qid, []))}
            for qid, qtext in queries
        ]


def _handle_single_query(query: str, button: str | None, context: dict[str, Any], index_path: str | Path) -> str | None:
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError("Query is too long")
    results = retrieve(query, index_path, top_k=100)
    context["results"] = results
    context["domain_facets"] = domain_facets(results)
    context["intent_facets"] = intent_facets(results)
    context["analysis"] = query_analysis(query)
    context["suggestion"] = suggest_correction(query, index_path)
    context["highlight_terms"] = query_terms(query)

    if button == "lucky" and results:
        return results[0]["url"]
    return None


def create_app(index_path: str | Path = DEFAULT_INDEX_PATH):
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
            button = "search"
        else:
            query = request.form.get("query", "").strip()
            show_scores = request.form.get("show_scores") == "1"
            button = request.form.get("button")
        context["query"] = query
        context["show_scores"] = show_scores
        context["button"] = button

        try:
            batch_file = (
                request.files.get("batch_file")
                if request.method == "POST"
                else None
            )
            batch_review_submitted = request.form.get("batch_review_submitted")

            if batch_review_submitted:
                _handle_batch_review_submission(request, context, application.config["SEARCH_INDEX"])
                
            elif batch_file and batch_file.filename:
                _handle_batch_upload(batch_file, context, application.config["SEARCH_INDEX"])
            elif query:
                redirect_url = _handle_single_query(query, button, context, application.config["SEARCH_INDEX"])
                if redirect_url:
                    return redirect(redirect_url)
            elif not query and button == "lucky":
                return redirect("https://www.youtube.com/watch?v=5xBSrqpiiCk&start_radio=1")
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
    host: str, port: int, index_path: str | Path = DEFAULT_INDEX_PATH
) -> None:
    application = create_app(index_path)
    url = f"http://{host}:{port}"
    print(f"Starting Tübingen Search on {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    application.run(
        host=host, port=port, debug=False, use_reloader=False, threaded=True
    )
