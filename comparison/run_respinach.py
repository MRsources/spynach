"""Runs the respinach side of the comparison cases in comparison/cases.

    OPENBLAS_NUM_THREADS=4 PYTHONPATH=src python comparison/run_respinach.py [case ...] [--gpu] [--tag NAME]
        [--option key=value ...] [--no-profile] [--top N] [--report]

Each case runs with timed stages under cProfile and writes
comparison/runs/<case>[-gpu][-<tag>]/respinach/ with data.npz, timings.json, profile.json,
meta.json, figure_<name>.html, figure_<name>.png and console.log. Without
case names every directory in comparison/cases runs. Options are passed to the
case's run() as keyword arguments (values are parsed as JSON, so
--option npoints=64 gives an int). --gpu passes gpu=True to the case (the
Spinach runner's gpu switch) and stores the run under <case>-gpu, separate from
the CPU run; respinach refuses it until the GPU backend exists. --report builds the
HTML report afterwards.
See comparison/README.md for the protocol.
"""

from __future__ import annotations

import argparse
import cProfile
import importlib.util
import json
import os
import platform
import pstats
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "comparison"
CASES = COMPARISON / "cases"
RUNS = COMPARISON / "runs"


class Timer:
    """Stage timer: ``with timer("label"):`` records and prints the wall-clock time of the block. Stages do not nest."""

    def __init__(self):
        self.records: list[dict] = []
        self._running: str | None = None

    @contextmanager
    def __call__(self, label: str):
        if self._running is not None:
            raise RuntimeError(f"stage {self._running!r} is still running, stages do not nest.")
        self._running = label
        t0 = time.perf_counter()
        try:
            yield
        finally:
            seconds = time.perf_counter() - t0
            self._running = None
            self.records.append({"label": label, "seconds": seconds})
            print(f"{label:32s} {seconds:8.1f} s", flush=True)


class _Tee:
    """Duplicates stdout/stderr into a log file."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_example(name: str) -> ModuleType:
    """Imports examples/<name>.py (e.g. ``"nmr_liquids/noesy_strychnine"``) as a module."""
    return _load(ROOT / "examples" / f"{name}.py", "respinach_example_" + name.replace("/", "_"))


def load_case(name: str) -> ModuleType:
    """Imports comparison/cases/<name>/<name>_respinach.py, which defines run(timer, **options) -> (data, figs)."""
    path = CASES / name / f"{name}_respinach.py"
    if not path.exists():
        raise FileNotFoundError(f"no case {name!r}: expected {path}")
    return _load(path, f"comparison_case_{name}")


def all_cases() -> list[str]:
    return sorted(p.name for p in CASES.iterdir() if (p / f"{p.name}_respinach.py").exists())


def parse_options(pairs: list[str]) -> dict:
    options = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if not _:
            raise SystemExit(f"--option expects key=value, got {pair!r}")
        try:
            options[key] = json.loads(value)
        except json.JSONDecodeError:
            options[key] = value
    return options


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def make_meta(case: str, options: dict) -> dict:
    import scipy

    import respinach

    status = [line for line in _git("status", "--porcelain").splitlines() if line.strip()]
    dirty = any(not line.startswith("??") or line.strip().endswith(".py") for line in status)
    return {
        "side": "respinach", "case": case,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "respinach_root": str(ROOT), "respinach_commit": _git("rev-parse", "HEAD"),
        "respinach_describe": _git("describe", "--tags", "--always", "--dirty"), "respinach_dirty": dirty,
        "respinach_version": getattr(respinach, "__version__", "0.0.1"),
        "python_version": platform.python_version(), "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "platform": platform.platform(), "hostname": socket.gethostname(),
        "cores": os.cpu_count(), "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "gpu": bool(options.get("gpu", False)), "gpu_device": "",
        "options": options,
    }


def profile_records(pr: cProfile.Profile, top_n: int) -> dict:
    """Top functions by cumulative and by self time, in the layout of the MATLAB profile.json."""
    stats = pstats.Stats(pr)
    records = []
    for (file, line, name), (_, calls, self_s, total_s, _) in stats.stats.items():
        label = f"{name} ({Path(file).name}:{line})" if file not in ("~", "") else name
        records.append({"function": label, "file": file, "calls": calls, "total_s": total_s, "self_s": self_s})
    by_total = sorted(records, key=lambda r: -r["total_s"])[:top_n]
    by_self = sorted(records, key=lambda r: -r["self_s"])[:top_n]
    return {"by_total": by_total, "by_self": by_self}


def save_figures(figs: dict, out_dir: Path) -> None:
    """Writes each figure as figure_<name>.html and .png (Kaleido if installed, otherwise a Matplotlib rendering)."""
    for name, fig in figs.items():
        fig.write_html(out_dir / f"figure_{name}.html")
        png = out_dir / f"figure_{name}.png"
        try:
            fig.write_image(png, scale=2)
        except Exception:
            try:
                _png_with_matplotlib(fig, png)
            except ImportError:
                print(f"figure_{name}.png skipped: neither Kaleido nor Matplotlib is installed")


def _png_with_matplotlib(fig, path: Path) -> None:
    """Static rendering of a respinach.plotting figure: contour traces (plot_2d) and line traces (plot_1d, plot_fid)."""
    import re

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strip = lambda text: re.sub(r"<[^>]+>", "", text or "")   # noqa: E731
    is_2d = any(trace.type == "contour" for trace in fig.data)
    mpl_fig, ax = plt.subplots(figsize=(7, 7) if is_2d else (9, 4))
    for trace in fig.data:
        if trace.type == "contour":
            start, end, size = trace.contours.start, trace.contours.end, trace.contours.size
            levels = np.arange(start, end + size / 2, size)
            scale = trace.colorscale
            colours = [scale[min(int(round(k * (len(scale) - 1) / max(len(levels) - 1, 1))), len(scale) - 1)][1]
                       for k in range(len(levels))]
            ax.contour(np.asarray(trace.x), np.asarray(trace.y), np.asarray(trace.z), levels=levels, colors=colours, linewidths=0.6)
        elif trace.type in ("scatter", "scattergl"):
            ax.plot(np.asarray(trace.x), np.asarray(trace.y), lw=0.8, label=trace.name or None)
    layout = fig.layout
    if layout.xaxis.range:
        ax.set_xlim(*layout.xaxis.range)
    if layout.yaxis.range:
        ax.set_ylim(*layout.yaxis.range)
    ax.set_xlabel(strip(layout.xaxis.title.text))
    ax.set_ylabel(strip(layout.yaxis.title.text))
    if layout.title and layout.title.text:
        ax.set_title(strip(layout.title.text))
    if not is_2d and len(fig.data) > 1:
        ax.legend()
    mpl_fig.tight_layout()
    mpl_fig.savefig(path, dpi=150)
    plt.close(mpl_fig)


def run_name(case: str, gpu: bool, tag: str | None) -> str:
    """runs/ directory name: <case>[-gpu][-<tag>], the convention shared with run_comparison.m."""
    return case + ("-gpu" if gpu else "") + (f"-{tag}" if tag else "")


def run_case(case: str, options: dict, tag: str | None, profile: bool, top_n: int) -> Path:
    gpu = bool(options.get("gpu", False))
    name = run_name(case, gpu, tag)
    out_dir = RUNS / name / "respinach"
    out_dir.mkdir(parents=True, exist_ok=True)
    module = load_case(case)
    with open(out_dir / "console.log", "w") as log:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(stdout, log), _Tee(stderr, log)
        try:
            print(f"== {case} (respinach at {ROOT}) -> {out_dir}")
            timer = Timer()
            pr = cProfile.Profile() if profile else None
            wall = time.perf_counter()
            if pr:
                pr.enable()
            try:
                data, figs = module.run(timer, **options)
            except NotImplementedError as error:
                if pr:
                    pr.disable()
                print(f"respinach cannot run this case: {error}")
                meta = make_meta(case, options)
                meta["error"] = str(error)
                (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
                for stale in ("data.npz", "timings.json", "profile.json"):
                    (out_dir / stale).unlink(missing_ok=True)
                return out_dir.parent
            finally:
                if pr:
                    pr.disable()
            total = time.perf_counter() - wall
            print(f"{'total':32s} {total:8.1f} s")
            np.savez(out_dir / "data.npz", **{k: np.asarray(v) for k, v in data.items()})
            (out_dir / "timings.json").write_text(json.dumps({"stages": timer.records, "total_s": total}, indent=2))
            if pr:
                (out_dir / "profile.json").write_text(json.dumps(profile_records(pr, top_n), indent=2))
            (out_dir / "meta.json").write_text(json.dumps(make_meta(case, options), indent=2))
            save_figures(figs or {}, out_dir)
        finally:
            sys.stdout, sys.stderr = stdout, stderr
    return out_dir.parent


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cases", nargs="*", help="case directories under comparison/cases (default: all)")
    ap.add_argument("--gpu", action="store_true", help="gpu=True for the case; results go to runs/<case>-gpu/")
    ap.add_argument("--tag", default=None, help="run name suffix: runs/<case>[-gpu]-<tag>/")
    ap.add_argument("--option", action="append", default=[], metavar="KEY=VALUE", help="keyword argument for the case's run()")
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--top", type=int, default=40, help="profiler functions to keep")
    ap.add_argument("--report", action="store_true", help="build runs/<case>/report.html afterwards")
    args = ap.parse_args(argv)
    options = parse_options(args.option)
    if args.gpu:
        options["gpu"] = True
    cases = args.cases or all_cases()
    if not cases:
        raise SystemExit(f"no cases found under {CASES}")
    for case in cases:
        run_dir = run_case(case, options, args.tag, not args.no_profile, args.top)
        if args.report:
            from comparison.report import build_report

            print(f"report: {build_report(run_dir)}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
