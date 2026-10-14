"""Builds the HTML report of a comparison run from its spinach/ and respinach/ directories.

    python comparison/report.py <run> [<run> ...]        # <run> is a name under comparison/runs or a path

Writes comparison/runs/<run>/report.html: provenance of both sides, stage timings
side by side, normwise differences of every array both sides saved, the figures,
the profiler tops, and links to the console logs. A missing side is reported as
such, so the report can be built after either half of the protocol.
"""

from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "comparison" / "runs"
SIDES = ("spinach", "respinach")
TOP_SHOWN = 25


def load_side(run_dir: Path, side: str) -> dict | None:
    side_dir = run_dir / side
    if not side_dir.is_dir():
        return None
    out = {"dir": side_dir}
    for name in ("meta", "timings", "profile"):
        path = side_dir / f"{name}.json"
        out[name] = json.loads(path.read_text()) if path.exists() else None
    out["data"] = load_data(side_dir)
    out["figures"] = {p.stem[len("figure_"):]: p for p in sorted(side_dir.glob("figure_*.png"))}
    out["figure_html"] = {p.stem[len("figure_"):]: p for p in sorted(side_dir.glob("figure_*.html"))}
    out["console"] = side_dir / "console.log" if (side_dir / "console.log").exists() else None
    return out


def load_data(side_dir: Path) -> dict[str, np.ndarray] | None:
    if (side_dir / "data.mat").exists():
        from scipy.io import loadmat

        raw = loadmat(str(side_dir / "data.mat"))
        return {k: np.asarray(v) for k, v in raw.items() if not k.startswith("__")}
    if (side_dir / "data.npz").exists():
        with np.load(side_dir / "data.npz") as raw:
            return {k: raw[k] for k in raw.files}
    return None


def compare_arrays(a: dict | None, b: dict | None) -> list[dict]:
    """Rows for every variable: shapes, normwise relative error ||respinach - spinach|| / ||spinach||, max |difference|."""
    a, b = a or {}, b or {}
    rows = []
    for name in sorted(set(a) | set(b)):
        row = {"name": name, "spinach_shape": None, "respinach_shape": None, "rel": None, "max_abs": None, "note": ""}
        if name in a:
            row["spinach_shape"] = a[name].shape
        if name in b:
            row["respinach_shape"] = b[name].shape
        if name in a and name in b:
            x, y = np.asarray(b[name]), np.asarray(a[name])
            if x.shape != y.shape and np.squeeze(x).shape == np.squeeze(y).shape:
                x, y = np.squeeze(x), np.squeeze(y)          # MATLAB stores a vector as 1 x n
                row["note"] = "compared after squeezing singleton dimensions"
            if x.shape != y.shape:
                row["note"] = "shape mismatch"
            else:
                diff = np.linalg.norm((x - y).ravel())
                ref = np.linalg.norm(y.ravel())
                row["rel"] = float(diff / ref) if ref > 0 else (0.0 if diff == 0 else float("inf"))
                row["max_abs"] = float(np.max(np.abs(x - y))) if x.size else 0.0
        else:
            row["note"] = "only in " + ("spinach" if name in a else "respinach")
        rows.append(row)
    return rows


def timing_rows(sides: dict) -> list[dict]:
    labels: list[str] = []
    per_side = {}
    for side in SIDES:
        t = (sides.get(side) or {}).get("timings")
        per_side[side] = {r["label"]: r["seconds"] for r in (t or {}).get("stages", [])}
        for label in per_side[side]:
            if label not in labels:
                labels.append(label)
    rows = [{"label": l, "spinach": per_side["spinach"].get(l), "respinach": per_side["respinach"].get(l)} for l in labels]
    totals = {side: ((sides.get(side) or {}).get("timings") or {}).get("total_s") for side in SIDES}
    rows.append({"label": "total (wall clock)", "spinach": totals["spinach"], "respinach": totals["respinach"]})
    return rows


# ---------------------------------------------------------------- HTML
CSS = """
:root { --bg:#ffffff; --ink:#1b1b1f; --ink2:#4b4b55; --line:#dcdce3; --accent:#1d4ed8; --surface:#f5f5f8; --warn:#b3261e; }
@media (prefers-color-scheme: dark) { :root { --bg:#121216; --ink:#ececf1; --ink2:#b4b4c0; --line:#33333d; --accent:#8ab4ff; --surface:#1c1c22; --warn:#ff8a80; } }
body { margin:0; padding:24px 16px 48px; background:var(--bg); color:var(--ink); font:15px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width:1200px; margin:0 auto; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:18px; margin:36px 0 10px; border-bottom:1px solid var(--line); padding-bottom:4px; }
h3 { font-size:14px; margin:14px 0 6px; color:var(--ink2); text-transform:uppercase; letter-spacing:.04em; }
p.sub { color:var(--ink2); margin:0 0 20px; }
table { border-collapse:collapse; width:100%; font-size:14px; margin:6px 0 14px; }
th, td { text-align:left; padding:5px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--ink2); font-weight:600; } td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
td.mono { font-family:ui-monospace, Menlo, Consolas, monospace; font-size:13px; word-break:break-all; }
.cols { display:grid; grid-template-columns:1fr 1fr; gap:20px; } @media (max-width:800px) { .cols { grid-template-columns:1fr; } }
.card { background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }
.card img { width:100%; height:auto; border-radius:4px; background:#fff; }
.missing { color:var(--warn); } .ok { color:var(--accent); }
a { color:var(--accent); } .small { font-size:13px; color:var(--ink2); }
"""


def esc(x) -> str:
    return html.escape(str(x))


def fmt_s(x) -> str:
    return "" if x is None else f"{x:.2f}"


def fmt_ratio(a, b) -> str:
    return "" if a is None or b is None or a == 0 else f"{b / a:.2f}x"


def fmt_rel(x) -> str:
    return "" if x is None else f"{x:.3e}"


def img_tag(path: Path | None) -> str:
    if path is None:
        return '<p class="missing">no figure</p>'
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{esc(path.name)}">'


def meta_table(sides: dict) -> str:
    keys = {
        "spinach": [("created_utc", "run date (UTC)"), ("spinach_describe", "Spinach version"), ("spinach_commit", "commit"),
                    ("spinach_dirty", "local modifications"), ("matlab_version", "MATLAB"), ("hostname", "host"),
                    ("platform", "platform"), ("cores", "cores"), ("blas", "BLAS"), ("comp_threads", "computational threads"),
                    ("pool_workers", "parallel pool workers"),
                    ("gpu", "GPU requested"), ("gpu_device", "GPU device"), ("options", "case options")],
        "respinach": [("created_utc", "run date (UTC)"), ("respinach_describe", "respinach version"), ("respinach_commit", "commit"),
                   ("respinach_dirty", "local modifications"), ("python_version", "Python"), ("numpy_version", "NumPy"),
                   ("scipy_version", "SciPy"), ("hostname", "host"), ("platform", "platform"), ("cores", "cores"),
                   ("blas", "BLAS"), ("thread_env", "thread settings"), ("gpu", "GPU requested"), ("gpu_device", "GPU device"),
                   ("options", "case options"), ("error", "not run")],
    }
    parts = []
    for side in SIDES:
        info = sides.get(side)
        rows = ""
        if info is None:
            rows = f'<tr><td colspan="2" class="missing">no {side}/ directory in this run</td></tr>'
        elif not info["meta"]:
            rows = '<tr><td colspan="2" class="missing">meta.json missing</td></tr>'
        else:
            for key, label in keys[side]:
                value = info["meta"].get(key)
                if key == "thread_env" and value == {}:
                    value = "library defaults"
                value = json.dumps(value) if isinstance(value, (dict, list)) else value
                rows += f"<tr><th>{esc(label)}</th><td class=\"mono\">{esc('' if value is None else value)}</td></tr>"
        parts.append(f'<div class="card"><h3>{side}</h3><table>{rows}</table></div>')
    return f'<div class="cols">{"".join(parts)}</div>'


def timings_table(rows: list[dict]) -> str:
    body = "".join(
        f"<tr><td>{esc(r['label'])}</td><td class=\"num\">{fmt_s(r['spinach'])}</td><td class=\"num\">{fmt_s(r['respinach'])}</td>"
        f"<td class=\"num\">{fmt_ratio(r['spinach'], r['respinach'])}</td></tr>" for r in rows)
    return ('<table><tr><th>stage</th><th class="num">Spinach / s</th><th class="num">respinach / s</th><th class="num">respinach / Spinach</th></tr>'
            f"{body}</table>")


def data_table(rows: list[dict]) -> str:
    if not rows:
        return '<p class="missing">no data arrays on either side</p>'
    body = ""
    for r in rows:
        body += (f"<tr><td class=\"mono\">{esc(r['name'])}</td><td class=\"mono\">{esc(r['spinach_shape'] or '')}</td>"
                 f"<td class=\"mono\">{esc(r['respinach_shape'] or '')}</td><td class=\"num\">{fmt_rel(r['rel'])}</td>"
                 f"<td class=\"num\">{fmt_rel(r['max_abs'])}</td><td class=\"missing\">{esc(r['note'])}</td></tr>")
    return ('<table><tr><th>variable</th><th>Spinach shape</th><th>respinach shape</th><th class="num">relative error</th>'
            f'<th class="num">max |difference|</th><th></th></tr>{body}</table>'
            '<p class="small">relative error = ||respinach - Spinach||<sub>F</sub> / ||Spinach||<sub>F</sub>, the metric of the oracle tests.</p>')


def figures_section(sides: dict) -> str:
    names: list[str] = []
    for side in SIDES:
        for name in ((sides.get(side) or {}).get("figures") or {}):
            if name not in names:
                names.append(name)
    if not names:
        return '<p class="missing">no figures</p>'
    out = ""
    for name in names:
        cols = ""
        for side in SIDES:
            info = sides.get(side) or {}
            png = (info.get("figures") or {}).get(name)
            link = (info.get("figure_html") or {}).get(name)
            extra = f'<p class="small"><a href="{esc(link.relative_to(link.parents[1]))}">interactive figure</a></p>' if link else ""
            cols += f'<div class="card"><h3>{side}: {esc(name)}</h3>{img_tag(png)}{extra}</div>'
        out += f'<div class="cols">{cols}</div>'
    return out


def profile_tables(sides: dict) -> str:
    cols = ""
    for side in SIDES:
        prof = (sides.get(side) or {}).get("profile")
        if not prof:
            cols += f'<div class="card"><h3>{side}</h3><p class="missing">no profile</p></div>'
            continue
        block = f'<div class="card"><h3>{side}</h3>'
        for key, title in (("by_self", "by self time"), ("by_total", "by total time (with callees)")):
            rows = "".join(
                f"<tr><td class=\"mono\">{esc(r['function'])}</td><td class=\"num\">{esc(r['calls'])}</td>"
                f"<td class=\"num\">{fmt_s(r['self_s'])}</td><td class=\"num\">{fmt_s(r['total_s'])}</td></tr>"
                for r in prof.get(key, [])[:TOP_SHOWN])
            block += (f"<p class=\"small\">{title}</p><table><tr><th>function</th><th class=\"num\">calls</th>"
                      f'<th class="num">self / s</th><th class="num">total / s</th></tr>{rows}</table>')
        cols += block + "</div>"
    return f'<div class="cols">{cols}</div>'


def logs_section(sides: dict, run_dir: Path) -> str:
    items = ""
    for side in SIDES:
        info = sides.get(side)
        log = info["console"] if info else None
        items += f"<li>{side}: " + (f'<a href="{esc(log.relative_to(run_dir))}">console.log</a>' if log else '<span class="missing">none</span>') + "</li>"
    return f"<ul>{items}</ul>"


def build_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    sides = {side: load_side(run_dir, side) for side in SIDES}
    case = next((s["meta"]["case"] for s in sides.values() if s and s["meta"]), run_dir.name)
    present = [side for side in SIDES if sides[side]]
    gpu = any(s and s["meta"] and s["meta"].get("gpu") for s in sides.values()) or run_dir.name.endswith("-gpu") or "-gpu-" in run_dir.name
    accel = "GPU" if gpu else "CPU"
    status = ("both sides present" if len(present) == 2 else f"only {present[0]} present" if present else "no results")
    failed = [side for side in present if sides[side]["meta"] and sides[side]["meta"].get("error")]
    if failed:
        status += "; " + ", ".join(f"{side} did not run ({sides[side]['meta']['error']})" for side in failed)
    body = f"""<main>
<h1>Spinach vs respinach: {esc(case)} ({accel})</h1>
<p class="sub">run <span class="mono">{esc(run_dir.name)}</span>, {esc(status)}. Built by comparison/report.py; protocol in comparison/README.md.</p>
<h2>Provenance</h2>{meta_table(sides)}
<h2>Timings</h2>{timings_table(timing_rows(sides))}
<h2>Data</h2>{data_table(compare_arrays((sides['spinach'] or {}).get('data'), (sides['respinach'] or {}).get('data')))}
<h2>Figures</h2>{figures_section(sides)}
<h2>Profiles</h2>{profile_tables(sides)}
<h2>Console logs</h2>{logs_section(sides, run_dir)}
</main>"""
    page = (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>Spinach vs respinach: {esc(case)} ({accel})</title><style>{CSS}</style></head><body>{body}</body></html>")
    out = run_dir / "report.html"
    out.write_text(page, encoding="utf-8")
    return out


def main(argv=None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    for arg in args:
        run_dir = Path(arg) if Path(arg).is_dir() else RUNS / arg
        if not run_dir.is_dir():
            raise SystemExit(f"no run directory {run_dir}")
        print(build_report(run_dir))


if __name__ == "__main__":
    main()
