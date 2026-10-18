"""Run the Python blocks of every guide in docs/ and check its notebook is current.

Blocks run in order and in one namespace per guide. Blocks that plot run
only when Plotly is installed (the "plot" or "docs" extra); they must not
define names that later blocks use.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs"
sys.path.insert(0, str(DOCS))
import build_notebooks  # noqa: E402

GUIDES = build_notebooks.guides()
HAS_PLOTLY = importlib.util.find_spec("plotly") is not None

# A block plots if it calls a plotting function or uses Plotly directly
PLOTS = re.compile(r"\bplot_\w+\(|\bplotly\b")


def python_blocks(guide: Path) -> list[str]:
    return build_notebooks.FENCE.findall(guide.read_text(encoding="utf-8"))


def plots(block: str) -> bool:
    return bool(PLOTS.search(block))


def run_blocks(guide: Path, with_plots: bool) -> None:
    namespace: dict = {}
    for n, block in enumerate(python_blocks(guide)):
        if plots(block) and not with_plots:
            continue
        exec(compile(block, f"{guide.name} block {n}", "exec"), namespace)


@pytest.mark.parametrize("guide", GUIDES, ids=lambda p: p.name)
def test_examples_run(guide):
    assert python_blocks(guide), f"{guide.name} has no Python examples"
    run_blocks(guide, with_plots=False)


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
@pytest.mark.parametrize("guide", [g for g in GUIDES if any(map(plots, python_blocks(g)))], ids=lambda p: p.name)
def test_examples_run_with_plots(guide):
    run_blocks(guide, with_plots=True)


@pytest.mark.parametrize("guide", GUIDES, ids=lambda p: p.name)
def test_notebook_is_current(guide):
    path = build_notebooks.notebook_path(guide)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    assert text and build_notebooks.strip_outputs(text) == build_notebooks.build(guide), (
        f"{path.name} is out of date; run: python docs/build_notebooks.py"
    )
    assert build_notebooks.is_executed(text), (
        f"{path.name} was saved without outputs; run: python docs/build_notebooks.py"
    )


@pytest.mark.parametrize("guide", GUIDES, ids=lambda p: p.name)
def test_figures_are_named_saved_and_static(guide):
    """Every plotting block names its figure file, the file exists, and the saved output carries a PNG copy."""
    names = build_notebooks.figure_files(guide)
    assert len(names) == len(python_blocks(guide))
    for name in filter(None, names):
        assert (build_notebooks.FIGURES / name).exists(), (
            f"docs/figures/{name} is missing; run: python docs/build_notebooks.py"
        )
    notebook = json.loads(build_notebooks.notebook_path(guide).read_text(encoding="utf-8"))
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    for name, cell in zip(names, code_cells, strict=True):
        figures = [o["data"] for o in cell["outputs"] if build_notebooks.PLOTLY in o.get("data", {})]
        assert len(figures) == (1 if name else 0), f"block with figure file {name!r} shows {len(figures)} figures"
        assert all("image/png" in data for data in figures), "a Plotly output has no PNG copy for GitHub"


def test_image_lines_stay_out_of_notebooks():
    for guide in GUIDES:
        for cell in build_notebooks.build_notebook(guide)["cells"]:
            assert "](figures/" not in "".join(cell["source"]), f"{guide.name}: an image line reached the notebook"


def test_no_orphan_notebooks():
    expected = {build_notebooks.notebook_path(g).name for g in GUIDES}
    orphans = {p.name for p in build_notebooks.NOTEBOOKS.glob("*.ipynb")} - expected
    assert not orphans, f"notebooks without a guide in docs/: {sorted(orphans)}"
