"""Tests of the comparison tooling: runner helpers and the report builder (no simulations)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from comparison import report, run_spinpy  # noqa: E402


def test_parse_options_and_cases():
    assert run_spinpy.parse_options(["npoints=64", "name=abc", "flag=true"]) == {"npoints": 64, "name": "abc", "flag": True}
    with pytest.raises(SystemExit):
        run_spinpy.parse_options(["npoints"])
    assert "noesy_strychnine" in run_spinpy.all_cases()
    assert run_spinpy.run_name("c", False, None) == "c" and run_spinpy.run_name("c", True, None) == "c-gpu"
    assert run_spinpy.run_name("c", True, "quick") == "c-gpu-quick" and run_spinpy.run_name("c", False, "quick") == "c-quick"
    assert (ROOT / "comparison/cases/noesy_strychnine/noesy_strychnine_spinach.m").exists()


def test_timer_records_and_rejects_nesting():
    timer = run_spinpy.Timer()
    with timer("a"):
        pass
    assert [r["label"] for r in timer.records] == ["a"] and timer.records[0]["seconds"] >= 0
    with pytest.raises(RuntimeError, match="nest"):
        with timer("outer"):
            with timer("inner"):
                pass


def test_compare_arrays():
    a = {"x": np.ones((2, 2)), "y": np.arange(3.0), "m": np.zeros(2)}
    b = {"x": np.ones((2, 2)) * (1 + 1e-12), "y": np.arange(4.0), "p": np.zeros(2)}
    rows = {r["name"]: r for r in report.compare_arrays(a, b)}
    assert rows["x"]["rel"] == pytest.approx(1e-12) and rows["x"]["max_abs"] == pytest.approx(1e-12)
    assert rows["y"]["note"] == "shape mismatch" and rows["y"]["rel"] is None
    squeezed = report.compare_arrays({"v": np.ones((1, 5))}, {"v": np.ones(5)})[0]
    assert squeezed["rel"] == 0.0 and "squeez" in squeezed["note"]
    assert rows["m"]["note"] == "only in spinach" and rows["p"]["note"] == "only in spinpy"


def _write_side(side_dir: Path, side: str, factor: float):
    side_dir.mkdir(parents=True)
    data = {"fid": np.exp(-np.arange(8.0)) * factor, "spectrum": np.ones((4, 4))}
    if side == "spinach":
        from scipy.io import savemat

        savemat(str(side_dir / "data.mat"), data, format="5")
    else:
        np.savez(side_dir / "data.npz", **data)
    (side_dir / "timings.json").write_text(json.dumps({"stages": [{"label": "basis", "seconds": 1.0 * factor},
                                                                    {"label": "liquid", "seconds": 2.0 * factor}], "total_s": 3.5 * factor}))
    (side_dir / "meta.json").write_text(json.dumps({"side": side, "case": "demo", "created_utc": "2026-10-12T00:00:00Z", "options": {"n": 4}}))
    (side_dir / "profile.json").write_text(json.dumps({"by_total": [{"function": "f", "file": "f.m", "calls": 1, "total_s": 1.0, "self_s": 0.5}],
                                                        "by_self": [{"function": "g", "file": "g.m", "calls": 2, "total_s": 0.5, "self_s": 0.5}]}))
    (side_dir / "console.log").write_text("ran\n")


def test_build_report_both_sides(tmp_path):
    run_dir = tmp_path / "demo"
    _write_side(run_dir / "spinach", "spinach", 1.0)
    _write_side(run_dir / "spinpy", "spinpy", 1.0 + 1e-10)
    out = report.build_report(run_dir)
    page = out.read_text()
    assert out.name == "report.html" and "both sides present" in page and "demo (CPU)" in page
    assert "Spinach vs spinpy: demo" in page and "e-10<" in page and "no figure" in page   # fid relative error 1e-10
    rows = report.timing_rows({s: report.load_side(run_dir, s) for s in report.SIDES})
    assert [r["label"] for r in rows] == ["basis", "liquid", "total (wall clock)"]
    assert rows[0]["spinpy"] == pytest.approx(1.0 + 1e-10)


def test_build_report_one_side(tmp_path):
    run_dir = tmp_path / "demo"
    _write_side(run_dir / "spinpy", "spinpy", 1.0)
    page = report.build_report(run_dir).read_text()
    assert "only spinpy present" in page and "no spinach/ directory" in page
    assert "only in spinpy" in page
    gpu_dir = tmp_path / "demo-gpu"
    _write_side(gpu_dir / "spinpy", "spinpy", 1.0)
    (gpu_dir / "spinpy" / "meta.json").write_text(json.dumps({"side": "spinpy", "case": "demo", "gpu": True, "error": "gpu not ported"}))
    page = report.build_report(gpu_dir).read_text()
    assert "demo (GPU)" in page and "spinpy did not run (gpu not ported)" in page


def test_profile_records_layout():
    import cProfile

    pr = cProfile.Profile()
    pr.enable()
    sum(range(1000))
    pr.disable()
    records = run_spinpy.profile_records(pr, 5)
    assert set(records) == {"by_total", "by_self"} and len(records["by_total"]) <= 5
    assert {"function", "file", "calls", "total_s", "self_s"} <= set(records["by_total"][0])
