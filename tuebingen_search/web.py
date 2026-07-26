"""Web interface for the search engine."""

from __future__ import annotations

from flask import Flask, render_template, request, Response

from .retrieval import retrieve, retrieve_batch_list

app = Flask(__name__)

def _parse_uploaded_queries(stream) -> list[tuple[str, str]]:
    """Parse a tab-separated query file stream from the web upload."""
    queries = []
    for line_num, raw_line in enumerate(stream, start=1):
        line = raw_line.decode('utf-8').strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            query_id, text = line.split("\t", 1)
        else:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                query_id, text = parts
            else:
                continue  # Skip invalid lines gracefully in web UI
        queries.append((query_id.strip(), text.strip()))
    return queries


def _generate_tsv_output(queries: list[tuple[str, str]], batch_results: dict) -> str:
    """Generate a TSV string of the batch results for download."""
    tsv_lines = []
    for q_id, _ in queries:
        for res in batch_results.get(q_id, []):
            tsv_lines.append(f"{q_id}\t{res['rank']}\t{res['url']}\t{res['score']}")
    return "\n".join(tsv_lines)


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    batch_results = None
    queries = []
    tsv_output = ""
    query = ""

    if request.method == "POST":
        query = request.form.get("query", "")
        show = request.form.get("show_scores") == "True"
        
        batch_file = request.files.get("batch_file")
        if batch_file and batch_file.filename:
            queries = _parse_uploaded_queries(batch_file.stream)
            if queries:
                batch_results = retrieve_batch_list(queries, "tuebingen_index.sqlite3", top_k=100)
                tsv_output = _generate_tsv_output(queries, batch_results)
        elif query.strip():
            results = retrieve(query, "tuebingen_index.sqlite3", top_k=100)

        return render_template(
            "index.html",
            query=query,
            results=results,
            batch_results=batch_results,
            queries=queries,
            tsv_output=tsv_output,
            show=show,
        )

    return render_template("index.html", query=query, results=results, batch_results=batch_results, queries=queries, tsv_output=tsv_output, show=False)

@app.route("/download", methods=["POST"])
def download():
    tsv_data = request.form.get("tsv_data", "")
    return Response(
        tsv_data,
        mimetype="text/tab-separated-values",
        headers={"Content-disposition": "attachment; filename=results.tsv"}
    )


def start_web_interface(host, port):
    print(f"starting web interface on {host}:{port}")
    app.run(host=host, port=port, debug=True, use_reloader=False)
