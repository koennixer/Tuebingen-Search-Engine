"""Optional desktop search interface.

Run after building an index:
    python pyglet_UI.py
"""

from __future__ import annotations

import argparse
import textwrap
import webbrowser

import pyglet
from pyglet.window import key

from tuebingen_search.retrieval import retrieve


class SearchWindow(pyglet.window.Window):
    def __init__(self, index: str) -> None:
        super().__init__(900, 650, caption="Tübingen Search", resizable=True)
        self.index = index
        self.query = ""
        self.results: list[dict] = []
        self.labels: list[pyglet.text.Label] = []
        self.status = "Type a query and press Enter"
        self.rebuild_labels()

    def rebuild_labels(self) -> None:
        self.labels = [
            pyglet.text.Label(
                "Tübingen Search",
                x=24, y=self.height - 28, anchor_y="top", font_size=20, bold=True,
            ),
            pyglet.text.Label(
                f"> {self.query}_",
                x=24, y=self.height - 70, anchor_y="top", font_size=14,
            ),
            pyglet.text.Label(
                self.status,
                x=24, y=self.height - 105, anchor_y="top", color=(90, 90, 90, 255),
            ),
        ]
        y = self.height - 140
        for result in self.results[:8]:
            title = textwrap.shorten(
                f"{result['rank']}. {result['title'] or '(untitled)'}",
                width=95, placeholder="…",
            )
            snippet = textwrap.shorten(result["snippet"], width=115, placeholder="…")
            self.labels.extend(
                [
                    pyglet.text.Label(title, x=24, y=y, anchor_y="top", bold=True),
                    pyglet.text.Label(
                        result["url"], x=40, y=y - 22, anchor_y="top",
                        color=(40, 105, 65, 255),
                    ),
                    pyglet.text.Label(
                        snippet, x=40, y=y - 43, anchor_y="top",
                        color=(65, 65, 65, 255),
                    ),
                ]
            )
            y -= 68

    def on_draw(self) -> None:
        self.clear()
        for label in self.labels:
            label.draw()

    def on_text(self, text: str) -> None:
        if text.isprintable():
            self.query += text
            self.rebuild_labels()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == key.BACKSPACE:
            self.query = self.query[:-1]
        elif symbol == key.ENTER:
            self.results = retrieve(self.query, self.index, top_k=8)
            self.status = f"{len(self.results)} result(s); press 1–8 to open"
        elif key._1 <= symbol <= key._8:
            position = symbol - key._1
            if position < len(self.results):
                webbrowser.open(self.results[position]["url"])
        elif symbol == key.ESCAPE:
            self.close()
        self.rebuild_labels()

    def on_resize(self, width: int, height: int):
        result = super().on_resize(width, height)
        self.rebuild_labels()
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Desktop UI for Tübingen Search")
    parser.add_argument("--index", default="tuebingen_index.sqlite3")
    args = parser.parse_args()
    SearchWindow(args.index)
    pyglet.app.run()


if __name__ == "__main__":
    main()
