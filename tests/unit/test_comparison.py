"""Tests of the comparison tooling: runner helpers and the report builder (no simulations)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from comparison import report, run_respinach  # noqa: E402


def test_parse_options_and_cases():
    assert run_respinach.parse_options(["npoints=64", "name=abc", "flag=true"]) == {"npoints": 64, "name": "abc", "flag": True}
    with pytest.raises(SystemExit):
        run_respinach.parse_options(["npoints"])
    assert "noesy_strychnine" in run_respinach.all_cases()
    assert run_respinach.run_name("c", False, None) == "c" and run_respinach.run_name("c", True, None) == "c-gpu"
    assert run_respinach.run_name("c", True, "quick") == "c-gpu-quick" and run_respinach.run_name("c", False, "quick") == "c-quick"
    assert (ROOT / "comparison/cases/noesy_strychnine/noesy_strychnine_spinach.m").exists()


def test_provenance_helpers(monkeypatch):
    blas = run_respinach.blas_info()
    assert set(blas) == {"numpy", "scipy"} and all(isinstance(v, str) and v for v in blas.values())
    for var in run_respinach.THREAD_VARIABLES:
        monkeypatch.delenv(var, raising=False)
    assert run_respinach.thread_env() == {}
    monkeypatch.setenv("VECLIB_MAXIMUM_THREADS", "1")
    assert run_respinach.thread_env() == {"VECLIB_MAXIMUM_THREADS": "1"}


def test_make_meta_ignores_run_outputs(tmp_path, monkeypatch):
    scratch = ROOT / "comparison/runs/_meta_test/respinach"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        (scratch / "console.log").write_text("ran\n")
        meta = run_respinach.make_meta("demo", {"n": 1})
    finally:
        (scratch / "console.log").unlink()
        scratch.rmdir()
        scratch.parent.rmdir()
    assert meta["side"] == "respinach" and meta["case"] == "demo" and meta["options"] == {"n": 1}
    assert meta["respinach_describe"].endswith("-dirty") == meta["respinach_dirty"]
    assert set(meta["blas"]) == {"numpy", "scipy"}


def test_timer_records_and_rejects_nesting():
    timer = run_respinach.Timer()
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
    assert rows["m"]["note"] == "only in spinach" and rows["p"]["note"] == "only in respinach"


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
    _write_side(run_dir / "respinach", "respinach", 1.0 + 1e-10)
    out = report.build_report(run_dir)
    page = out.read_text()
    assert out.name == "report.html" and "both sides present" in page and "demo (CPU)" in page
    assert "Spinach vs respinach: demo" in page and "e-10<" in page and "no figure" in page   # fid relative error 1e-10
    rows = report.timing_rows({s: report.load_side(run_dir, s) for s in report.SIDES})
    assert [r["label"] for r in rows] == ["basis", "liquid", "total (wall clock)"]
    assert rows[0]["respinach"] == pytest.approx(1.0 + 1e-10)


def test_build_report_one_side(tmp_path):
    run_dir = tmp_path / "demo"
    _write_side(run_dir / "respinach", "respinach", 1.0)
    page = report.build_report(run_dir).read_text()
    assert "only respinach present" in page and "no spinach/ directory" in page
    assert "only in respinach" in page
    gpu_dir = tmp_path / "demo-gpu"
    _write_side(gpu_dir / "respinach", "respinach", 1.0)
    (gpu_dir / "respinach" / "meta.json").write_text(json.dumps({"side": "respinach", "case": "demo", "gpu": True, "error": "gpu not ported"}))
    page = report.build_report(gpu_dir).read_text()
    assert "demo (GPU)" in page and "respinach did not run (gpu not ported)" in page


def test_profile_records_layout():
    import cProfile

    pr = cProfile.Profile()
    pr.enable()
    sum(range(1000))
    pr.disable()
    records = run_respinach.profile_records(pr, 5)
    assert set(records) == {"by_total", "by_self"} and len(records["by_total"]) <= 5
    assert {"function", "file", "calls", "total_s", "self_s"} <= set(records["by_total"][0])


def test_zulf_case_and_matlab_csv_loader(tmp_path):
    assert "zulf_pf6_1d" in run_respinach.all_cases()
    assert (ROOT / "comparison/cases/zulf_pf6_1d/zulf_pf6_1d_spinach.m").exists()
    example = run_respinach.load_example("user_ml/zulf_pf6_1d")
    csv = tmp_path / "rho.csv"
    csv.write_text("1\n0\n3.0231e-06+1.3835e-20i\n-2.6325e-08-1.2e-21i\n")
    rho = example.load_matlab_csv(csv)
    assert rho.shape == (4,) and rho.dtype == complex
    np.testing.assert_allclose(rho, [1, 0, 3.0231e-06 + 1.3835e-20j, -2.6325e-08 - 1.2e-21j])
    full = example.load_matlab_csv(example.RHO_PATH)
    assert full.shape == (4**7,) and full[0] == 1   # unit-state component of the adiabatic-drop state
