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
        if request.form["type"] == "query":
            query = request.form["query"]
            show = request.form["show_scores"]

            if query.strip():
                results = retrieve(query, "tuebingen_index.sqlite3", top_k=100)

            return render_template(
                "index.html",
                query=query, 
                results=results,
                show=show
            )
        if request.form["type"] == "scores":
            show = request.form["show_scores"] == "True"
            query = request.form["query"]
            results = request.form["results"]
            return render_template(
                "index.html",
                query=query, 
                results=results,
                show = show
            )


def start_web_interface(host, port):
    print(f"starting web interface on {host}:{port}")
    app.run(host=host, port=port, debug=True, use_reloader=False)
