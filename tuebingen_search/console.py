"""Small terminal formatting helpers used by the command-line interface."""

from __future__ import annotations

import shutil
import sys


def terminal_width(default: int = 100, maximum: int = 110) -> int:
    """Return a conservative display width for readable command output."""
    return min(shutil.get_terminal_size((default, 24)).columns, maximum)


def rule(char: str = "-") -> str:
    """Return a horizontal separator sized to the current terminal."""
    return char * terminal_width()


def heading(text: str) -> str:
    """Format a simple section heading."""
    return f"\n{text}\n{rule('=')}"


def subheading(text: str) -> str:
    """Format a simple subsection heading."""
    return f"\n{text}\n{rule('-')}"


def table(rows: list[tuple[str, object]], *, key_width: int = 22) -> str:
    """Format key-value pairs as an aligned table."""
    return "\n".join(f"{key:<{key_width}} {value}" for key, value in rows)


def supports_color(stream=None) -> bool:
    """Return whether ANSI color is likely to render correctly."""
    stream = stream or sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)())


def success(text: str) -> str:
    """Format a success message without depending on color support."""
    return f"[OK] {text}"
