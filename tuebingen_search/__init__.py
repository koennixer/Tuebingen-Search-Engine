"""Public API for the Tübingen Search Engine project."""

from .crawler import crawl
from .evaluation import evaluate_run
from .models import Document, SearchResult
from .presentation import batch, interactive_search, run_batch_file
from .retrieval import load_query_file, query_analysis, retrieve, retrieve_batch
from .storage import DEFAULT_INDEX_PATH
from .storage import add_document as index
from .storage import export_documents_jsonl, index_statistics
from .vocabulary import DEFAULT_MIN_FREQUENCY, build_vocabulary, get_vocabulary_info, load_vocabulary

__all__ = [
    "DEFAULT_INDEX_PATH",
    "DEFAULT_MIN_FREQUENCY",
    "Document",
    "SearchResult",
    "batch",
    "crawl",
    "evaluate_run",
    "export_documents_jsonl",
    "index",
    "index_statistics",
    "interactive_search",
    "load_query_file",
    "query_analysis",
    "retrieve",
    "retrieve_batch",
    "run_batch_file",
    "build_vocabulary",
    "get_vocabulary_info",
    "load_vocabulary",
]
