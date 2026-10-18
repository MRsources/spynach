"""The committed parity summary agrees with its rendered outputs and, with fixtures, with the suite."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "oracle"))
import parity_summary  # noqa: E402


@pytest.fixture(scope="module")
def summary():
    return json.loads(parity_summary.JSON.read_text(encoding="utf-8"))


def test_outputs_are_current(summary):
    assert parity_summary.stale(summary) == [], (
        "run: OPENBLAS_NUM_THREADS=4 PYTHONPATH=src python oracle/parity_summary.py"
    )


def test_summary_matches_fixtures_on_disk(summary):
    try:
        tag, cases = parity_summary.fixture_cases()
    except (SystemExit, FileNotFoundError):
        pytest.skip("fixtures not available")
    assert tag == summary["fixture_tag"]
    recorded = sorted(c for f in summary["families"] for c in f["cases"])
    missing = sorted(set(recorded) - set(cases))
    if missing:
        pytest.skip(f"fixture cases missing on this machine: {missing[:3]}")
    assert recorded == sorted(cases), "fixture cases changed; rerun oracle/parity_summary.py"
    assert summary["fixture_cases"] == len(cases)


def test_summary_matches_collected_parity_tests(summary):
    assert parity_summary.collected_parity_tests() == summary["parity_tests"], (
        "parity tests changed; rerun oracle/parity_summary.py"
    )
    assert sum(f["tests"] for f in summary["families"]) == summary["parity_tests"]


def test_every_family_has_relative_tolerances(summary):
    for family in summary["families"]:
        assert family["comparisons"] > 0, family["label"]
        for rtol, worst in family["worst_by_rtol"].items():
            assert worst <= float(rtol) or float(rtol) == 0 and worst == 0, (family["label"], rtol, worst)
