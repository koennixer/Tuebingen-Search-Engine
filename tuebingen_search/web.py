"""Web interface for the search engine."""

from __future__ import annotations

from flask import Flask, render_template, request

from .retrieval import retrieve

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query = ""
    #show = False

    if request.method == "POST":
        query = request.form.get("query", "")
        show = request.form.get("show_scores") == "True"

        if query.strip():
            results = retrieve(query, "tuebingen_index.sqlite3", top_k=100)

        return render_template(
            "index.html",
            query=query,
            results=results,
            show=show,
        )

    return render_template("index.html", query=query, results=results, show=False)


def start_web_interface(host, port):
    print(f"starting web interface on {host}:{port}")
    app.run(host=host, port=port, debug=True, use_reloader=False)
