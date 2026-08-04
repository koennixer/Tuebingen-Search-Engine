"""Small, dependency-free nDCG evaluator for local engineering judgments."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path


def dcg(relevances: list[float], *, k: int = 10) -> float:
    """Compute graded discounted cumulative gain at ``k``."""
    return sum(
        (2.0**relevance - 1.0) / math.log2(rank + 1.0)
        for rank, relevance in enumerate(relevances[:k], start=1)
    )


def ndcg(relevances: list[float], *, k: int = 10) -> float:
    """Compute nDCG at ``k`` for one ranked relevance sequence."""
    ideal = dcg(sorted(relevances, reverse=True), k=k)
    return dcg(relevances, k=k) / ideal if ideal > 0 else 0.0


def load_qrels(path: str | Path) -> dict[str, dict[str, float]]:
    """Load ``query_id<TAB>url<TAB>graded_relevance`` judgments."""
    judgments: dict[str, dict[str, float]] = defaultdict(dict)
    with Path(path).open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"Qrels line {line_number} must have three TSV columns"
                )
            query_id, url, relevance = parts
            judgments[query_id][url] = float(relevance)
    return dict(judgments)


def load_run(path: str | Path) -> dict[str, list[str]]:
    """Load the assignment's four-column result format."""
    ranked: dict[str, list[tuple[int, str]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    with Path(path).open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(
                    f"Run line {line_number} must have four TSV columns"
                )
            query_id, rank_text, url, _score = parts
            key = (query_id, url)
            if key in seen:
                raise ValueError(
                    f"Duplicate URL for query {query_id} on line {line_number}"
                )
            seen.add(key)
            ranked[query_id].append((int(rank_text), url))
    return {
        query_id: [url for _rank, url in sorted(rows)]
        for query_id, rows in ranked.items()
    }


def evaluate_run(
    run_path: str | Path,
    qrels_path: str | Path,
    *,
    cutoffs: tuple[int, ...] = (5, 10, 20),
) -> dict[str, object]:
    """Evaluate one assignment-format run against local qrels."""
    run = load_run(run_path)
    qrels = load_qrels(qrels_path)
    per_query: dict[str, dict[str, float]] = {}
    for query_id, judgments in qrels.items():
        retrieved = run.get(query_id, [])
        relevances = [judgments.get(url, 0.0) for url in retrieved]
        # Include judged but unretrieved documents when constructing the ideal.
        ideal_relevances = sorted(judgments.values(), reverse=True)
        metrics: dict[str, float] = {}
        for cutoff in cutoffs:
            denominator = dcg(ideal_relevances, k=cutoff)
            metrics[f"nDCG@{cutoff}"] = (
                dcg(relevances, k=cutoff) / denominator
                if denominator > 0
                else 0.0
            )
        per_query[query_id] = metrics

    aggregate = {
        metric: (
            sum(values[metric] for values in per_query.values())
            / len(per_query)
            if per_query
            else 0.0
        )
        for metric in (f"nDCG@{cutoff}" for cutoff in cutoffs)
    }
    return {"aggregate": aggregate, "per_query": per_query}
