"""Public API for the Tübingen Search Engine project."""

from .crawler import crawl
from .models import Document, SearchResult
from .presentation import batch, interactive_search, run_batch_file
from .retrieval import load_query_file, retrieve, retrieve_batch
from .storage import add_document as index
from .storage import export_documents_jsonl, index_statistics

__all__ = [
    "Document",
    "SearchResult",
    "batch",
    "crawl",
    "export_documents_jsonl",
    "index",
    "index_statistics",
    "interactive_search",
    "load_query_file",
    "retrieve",
    "retrieve_batch",
    "run_batch_file",
]
