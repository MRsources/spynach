"""Generate a Jupyter notebook from each Markdown guide in docs/.

The Markdown guides are the source of truth: they render on any Git host and
their Python blocks are executed by tests/docs/. Each guide docs/<name>.md
becomes docs/notebooks/<name>.ipynb, with every ```python block as a code
cell and the prose in between as Markdown cells split at section headings.
Notebooks are written without outputs, so they diff cleanly.

    python docs/build_notebooks.py          # rewrite the notebooks
    python docs/build_notebooks.py --check  # exit 1 if any is out of date

Uses only the standard library, so it runs without Jupyter installed.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
NOTEBOOKS = DOCS / "notebooks"

# Guides that are not tutorials and get no notebook
SKIP = {"README.md"}

FENCE = re.compile(r"^```python\n(.*?)^```$", flags=re.S | re.M)
HEADING = re.compile(r"^(?=#{1,3} )", flags=re.M)
LINK = re.compile(r"\]\((?!https?://|mailto:)([^)]+)\)")

INTRO = (
    "*Generated from [`docs/{name}.md`](../{name}.md) by `docs/build_notebooks.py`;"
    " edit the Markdown guide, not this notebook.*\n\n"
    "Run the cells in order: later cells use names defined by earlier ones."
    " Plotting cells need matplotlib (`pip install -e \".[docs]\"`)."
)


def guides() -> list[Path]:
    return sorted(p for p in DOCS.glob("*.md") if p.name not in SKIP)


def notebook_path(guide: Path) -> Path:
    return NOTEBOOKS / f"{guide.stem}.ipynb"


def build(guide: Path) -> str:
    """Notebook JSON text for one guide."""
    text = guide.read_text(encoding="utf-8")
    cells: list[tuple[str, str]] = []
    position = 0
    for match in FENCE.finditer(text):
        cells.extend(("markdown", part) for part in _split_prose(text[position:match.start()]))
        cells.append(("code", match.group(1)))
        position = match.end()
    cells.extend(("markdown", part) for part in _split_prose(text[position:]))

    # The title cell comes first, then the provenance note
    intro = ("markdown", INTRO.format(name=guide.stem))
    cells.insert(1 if cells and cells[0][1].startswith("# ") else 0, intro)

    notebook = {
        "cells": [_cell(kind, source, n) for n, (kind, source) in enumerate(cells)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def _split_prose(prose: str) -> list[str]:
    """Markdown cells for the prose between code blocks, one per section heading."""
    parts = [_relink(part.strip("\n")) for part in HEADING.split(prose)]
    return [part for part in parts if part.strip()]


def _relink(markdown: str) -> str:
    """Point links at their targets as seen from docs/notebooks/.

    Links to other guides go to the sibling notebook, links to anything
    else one directory up, and section anchors are translated from
    GitHub's slugs to Jupyter's.
    """

    def target(match: re.Match[str]) -> str:
        path, _, anchor = match.group(1).partition("#")
        if path.startswith("notebooks/"):
            path = path.removeprefix("notebooks/")
        elif "/" not in path and path.endswith(".md") and path not in SKIP:
            path = Path(path).with_suffix(".ipynb").name
        elif path:
            path = f"../{path}"
        if anchor:
            anchor = "#" + _jupyter_anchors().get(anchor, anchor)
        return f"]({path}{anchor})"

    return LINK.sub(target, markdown)


@functools.cache
def _jupyter_anchors() -> dict[str, str]:
    """Map GitHub heading slugs to Jupyter heading ids, over all guides.

    GitHub lowercases the heading text and drops punctuation; Jupyter keeps
    the rendered text and replaces spaces with hyphens.
    """
    anchors = {}
    for guide in guides():
        prose = re.sub(r"^```.*?^```$", "", guide.read_text(encoding="utf-8"), flags=re.S | re.M)
        for heading in re.findall(r"^#{1,6} (.+)$", prose, flags=re.M):
            text = heading.replace("`", "").strip()
            github = re.sub(r"[^\w\- ]", "", text.lower()).replace(" ", "-")
            anchors.setdefault(github, text.replace(" ", "-"))
    return anchors


def _cell(kind: str, source: str, n: int) -> dict:
    cell = {"cell_type": kind, "id": f"cell-{n:03d}", "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    # A trailing newline in the last line is noise in Jupyter
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n")
    return cell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report stale notebooks instead of writing them")
    args = parser.parse_args()
    stale = []
    NOTEBOOKS.mkdir(exist_ok=True)
    for guide in guides():
        path = notebook_path(guide)
        text = build(guide)
        if path.exists() and path.read_text(encoding="utf-8") == text:
            continue
        stale.append(path)
        if not args.check:
            path.write_text(text, encoding="utf-8")
    for path in stale:
        print(f"{'stale' if args.check else 'wrote'}: {path.relative_to(DOCS.parent)}")
    return 1 if args.check and stale else 0


if __name__ == "__main__":
    sys.exit(main())
