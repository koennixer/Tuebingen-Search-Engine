"""Module for managing the custom SymSpell vocabulary from the crawled corpus."""

from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

from symspellpy import SymSpell

from .storage import connect


VOCAB_PKL = "tuebingen_vocab.pkl"
VOCAB_META = "tuebingen_vocab_meta.json"
VOCAB_COUNTS_PKL = "tuebingen_vocab_counts.pkl"
SURFACE_TOKEN_RE = re.compile(r"\w+")

DEFAULT_MIN_FREQUENCY = 3


class VocabProgress:
    def __init__(self, total: int, prefix: str = "Processing") -> None:
        self.total = max(total, 1)
        self.prefix = prefix
        self.stream = sys.stdout
        self.interactive = self.stream.isatty()
        self.last_render = 0.0
        self.last_width = 0
        self.term_width = shutil.get_terminal_size((100, 24)).columns

    def update(self, current: int, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_render < 0.2:
            return
        self.last_render = now
        bar_width = 22
        filled = int(bar_width * min(current / self.total, 1.0))
        line = (
            f"{self.prefix} [{'#' * filled}{'-' * (bar_width - filled)}] "
            f"{current}/{self.total}"
        )
        line = line[: max(20, self.term_width - 1)]
        if self.interactive:
            self.stream.write("\r" + line.ljust(max(self.last_width, len(line))))
            self.last_width = max(self.last_width, len(line))
        else:
            if force:
                self.stream.write(line + "\n")
        self.stream.flush()

    def done(self) -> None:
        self.update(self.total, force=True)
        if self.interactive:
            self.stream.write("\n")


def build_vocabulary(index_path: str | Path, min_frequency: int = DEFAULT_MIN_FREQUENCY, refresh_counts: bool = False) -> None:
    """Extract surface forms from the corpus, build a SymSpell dictionary, and serialize it."""
    index_path = Path(index_path)
    vocab_dir = index_path.parent / ".vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = vocab_dir / VOCAB_PKL
    meta_path = vocab_dir / VOCAB_META
    counts_path = vocab_dir / VOCAB_COUNTS_PKL

    if refresh_counts or not counts_path.exists():
        print()
        conn = connect(index_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, description, text FROM documents")
        rows = cursor.fetchall()
        total_docs = len(rows)
        
        # Measure total raw text bytes using file size
        raw_text_bytes = index_path.stat().st_size

        word_counts: Counter[str] = Counter()
        
        progress = VocabProgress(total_docs, "1/2 Extracting vocabulary")
        for i, (_, title, desc, text) in enumerate(rows):
            combined = f"{title} {desc} {text}".lower()
            tokens = SURFACE_TOKEN_RE.findall(combined)
            word_counts.update(tokens)
            progress.update(i + 1)
        progress.done()

        with open(counts_path, "wb") as f:
            pickle.dump((word_counts, raw_text_bytes), f)
    else:
        print("\nLoading cached word counts")
        with open(counts_path, "rb") as f:
            word_counts, raw_text_bytes = pickle.load(f)

    print(f"Applying frequency threshold N={min_frequency}")
    filtered_counts = {
        word: count for word, count in word_counts.items() 
        if count >= min_frequency and len(word) >= 3
    }

    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    
    total_words = len(filtered_counts)
    print()
    progress = VocabProgress(total_words, "2/2 Generating SymSpell map")
    for i, (word, count) in enumerate(filtered_counts.items()):
        sym_spell.create_dictionary_entry(word, count)
        progress.update(i + 1)
    progress.done()

    with open(pkl_path, "wb") as f:
        pickle.dump(sym_spell, f)

    metadata = {
        "created_at": datetime.now(ZoneInfo("Europe/Berlin")).isoformat(timespec="seconds"),
        "word_count": len(filtered_counts),
        "symspell_size": len(sym_spell.deletes),
        "data_size_bytes": raw_text_bytes,
        "min_frequency": min_frequency,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def get_vocabulary_info(index_path: str | Path) -> dict[str, Any] | None:
    """Return the metadata for the current vocabulary if it exists."""
    meta_path = Path(index_path).parent / ".vocab" / VOCAB_META
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vocabulary(index_path: str | Path) -> SymSpell | None:
    """Load the pre-computed SymSpell dictionary from disk."""
    pkl_path = Path(index_path).parent / ".vocab" / VOCAB_PKL
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)
