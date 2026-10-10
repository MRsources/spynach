"""Run the Python blocks of every guide in docs/ and check its notebook is current.

Blocks run in order and in one namespace per guide. Blocks that use
matplotlib run only when it is installed (the "docs" extra); they must not
define names that later blocks use.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs"
sys.path.insert(0, str(DOCS))
import build_notebooks  # noqa: E402

GUIDES = build_notebooks.guides()
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def python_blocks(guide: Path) -> list[str]:
    return build_notebooks.FENCE.findall(guide.read_text(encoding="utf-8"))


def run_blocks(guide: Path, with_plots: bool) -> None:
    namespace: dict = {}
    for n, block in enumerate(python_blocks(guide)):
        if "matplotlib" in block and not with_plots:
            continue
        exec(compile(block, f"{guide.name} block {n}", "exec"), namespace)


@pytest.mark.parametrize("guide", GUIDES, ids=lambda p: p.name)
def test_examples_run(guide):
    assert python_blocks(guide), f"{guide.name} has no Python examples"
    run_blocks(guide, with_plots=False)


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
@pytest.mark.parametrize("guide", [g for g in GUIDES if "matplotlib" in g.read_text(encoding="utf-8")],
                         ids=lambda p: p.name)
def test_examples_run_with_plots(guide, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    try:
        run_blocks(guide, with_plots=True)
    finally:
        plt.close("all")


@pytest.mark.parametrize("guide", GUIDES, ids=lambda p: p.name)
def test_notebook_is_current(guide):
    path = build_notebooks.notebook_path(guide)
    assert path.exists() and path.read_text(encoding="utf-8") == build_notebooks.build(guide), (
        f"{path.name} is out of date; run: python docs/build_notebooks.py"
    )


def test_no_orphan_notebooks():
    expected = {build_notebooks.notebook_path(g).name for g in GUIDES}
    orphans = {p.name for p in build_notebooks.NOTEBOOKS.glob("*.ipynb")} - expected
    assert not orphans, f"notebooks without a guide in docs/: {sorted(orphans)}"
