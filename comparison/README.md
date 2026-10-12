# Spinach vs spinpy comparison runs

A reusable protocol for timing, profiling and comparing a Spinach
calculation with its spinpy translation, ending in one HTML report per run.
The Spinach side runs in MATLAB on the host; the spinpy side runs in the
container (or anywhere spinpy is installed); the report is built from the
files both sides leave behind. Nothing here is executed by the test suite
except the report and runner helpers (`tests/unit/test_comparison.py`).

## Layout

```
comparison/
  README.md                      this protocol
  cases/<case>/<case>_spinach.m  MATLAB side of a case: timed stages, returns data and figures
  cases/<case>/<case>_spinpy.py  spinpy side: run(timer, **options) -> (data, figs)
  matlab/run_comparison.m        MATLAB runner (settings block at the top): profiler, dumps, figures
  matlab/lib/                    cmp_timer, cmp_profile, cmp_meta, cmp_json, cmp_option
  run_spinach.sh                 runs run_comparison.m in MATLAB batch mode
  run_spinpy.py                  spinpy runner: cProfile, dumps, figures
  report.py                      builds runs/<run>/report.html from the two sides
  runs/<run>/spinach/            MATLAB outputs (git-ignored)
  runs/<run>/spinpy/             spinpy outputs (git-ignored)
  runs/<run>/report.html         the report
```

**What is versioned.** `comparison/runs/` is committed except for the
large files: `data.mat`, `data.npz` and the interactive `figure_*.html`
are ignored, as are runs tagged `quick`. So every report, with its
timings, profiles, provenance, PNG figures and console logs, is in git and
a rerun shows up as a diff; the arrays behind a report live only on the
machine that made them, and the report's "interactive figure" links work
there only. Rerun the side in question to regenerate them.

`<run>` is `<case>[-gpu][-<tag>]`: the case name, `-gpu` when the run was
made with the GPU switch, and `-<tag>` when a tag was given (use the same
switch and tag on both sides). CPU and GPU comparisons of the same case
therefore live next to each other, `runs/noesy_strychnine/` and
`runs/noesy_strychnine-gpu/`, each with its own report. Each side directory
holds:

| file | content |
|---|---|
| `timings.json` | `stages` (label, seconds, in order) and `total_s` (wall clock of the whole case) |
| `profile.json` | `by_total` and `by_self`: the top functions with `calls`, `total_s` (with callees), `self_s` |
| `meta.json` | provenance: Spinach commit and dirty flag, MATLAB version, pool size; spinpy commit, Python/NumPy/SciPy versions, `OPENBLAS_NUM_THREADS`; host, cores, case options |
| `data.mat` / `data.npz` | every array the case returned, same names on both sides (`-v7` MAT file, readable by SciPy) |
| `figure_<name>.png` | the figures the case returned (spinpy also writes `figure_<name>.html`, the interactive Plotly figure) |
| `console.log` | everything printed during the run, including Spinach's report output |

## Running a comparison

1. **Spinach side, on the host.** Edit the settings block at the top of
   `matlab/run_comparison.m` (repository path, Spinach checkout, cases, the
   `gpu` switch, tag, per-case options), then either

   ```
   comparison/run_spinach.sh          # MATLAB=/path/to/matlab to override the executable
   ```

   or from MATLAB: `run('<repo>/comparison/matlab/run_comparison.m')`. The
   runner puts Spinach on the path, checks that `create()` resolves to that
   checkout, runs each case under `profile on`, and writes
   `runs/<run>/spinach/`. Figures are saved from invisible figure windows,
   so batch mode works.

2. **spinpy side, in the container.**

   ```
   OPENBLAS_NUM_THREADS=4 PYTHONPATH=src python comparison/run_spinpy.py noesy_strychnine --report
   ```

   writes `runs/noesy_strychnine/spinpy/` and, with `--report`, builds the
   report. `--gpu` selects the GPU run (`runs/noesy_strychnine-gpu/`). `--tag quick --option npoints=64 --option zerofill=256` runs a
   reduced case into `runs/noesy_strychnine-quick/`; option values are
   parsed as JSON and passed to the case's `run()` as keyword arguments.
   Without case names every case runs. The PNG figures need Kaleido or
   Matplotlib (`pip install -e ".[compare]"`); without either only the
   HTML figures are written.

3. **Report.** `python comparison/report.py noesy_strychnine` (a run name
   under `runs/`, or a path) writes `runs/noesy_strychnine/report.html`:
   provenance of both sides, stage timings side by side with the
   spinpy/Spinach ratio, the normwise relative error
   `||spinpy - Spinach||_F / ||Spinach||_F` and the largest elementwise
   difference of every array (the oracle tests' metric; FIDs of ported
   sequences agree to about 1e-10), the figures next to each other, the
   profiler tops of both sides, and links to the console logs. Either side
   may be missing; the report says so. The report embeds the PNGs, so it
   can be sent on its own.

**CPU and GPU runs.** The default on both sides is CPU. `gpu=true` in
the MATLAB settings block adds `'gpu'` to `sys.enable` of every case (the
case honours `options.gpu`), records the device in `meta.json` and writes
to `runs/<case>-gpu/`; Spinach itself falls back to the CPU when no
CUDA device is present, which `gpu_device` in the provenance table shows.
`--gpu` on `run_spinpy.py` does the same for spinpy, which currently
refuses it (`porting/gpu-notes.md`): the run then leaves only `meta.json`
with the error, and the report says so in its status line. A GPU report
is still worth building for the Spinach timings alone, and the spinpy
side fills in once the backend exists. Never mix sides from different
switches in one run directory.

The two sides can run in either order, and either can be rerun alone
(for example after a spinpy optimisation) before rebuilding the report.
Keep the Spinach outputs of a run when only spinpy changes: they are the
reference.

## Reading the numbers

- **Timings** are wall clock per stage. Spinach uses its parallel pool
  (`pool_workers` in `meta.json`) and spinpy is serial, so compare like
  with like: a 4-CPU container against an 8-worker pool measures
  throughput per machine, not per core. Spinach's first run after
  starting MATLAB includes pool start-up and JIT warm-up inside `create`;
  run the cases twice and keep the second run when that matters.
- **Profiles**: the MATLAB `self_s` list is the kernel hot spots
  (`profile('info')` with built-ins included in the `by_total` list); the
  Python list comes from cProfile, where C extensions such as SciPy sparse
  kernels appear as built-in methods. Both include the profiler overhead.
- **Data**: `shape mismatch` means the two sides ran at different sizes
  (same `npoints`/`zerofill` options on both sides). A relative error of
  order 1 with matching shapes means a real discrepancy, usually a sign,
  transpose or axis-ordering convention; errors around 1e-8 or below are
  floating-point and ordering differences.

## Adding a case

A case is a directory `cases/<case>/` with two files that run the same
calculation with the same stage labels and return the same arrays and
figures under the same names.

**MATLAB** `cases/<case>/<case>_spinach.m`:

```matlab
function [data,figs]=<case>_spinach(timer,options)
npoints=cmp_option(options,'npoints',512);     % options come from the settings block
...                                              % the Spinach example, verbatim
timer.start('basis'); spin_system=basis(spin_system,bas); timer.stop();
timer.start('liquid/noesy'); fid=liquid(spin_system,@noesy,parameters,'nmr'); timer.stop();
data.fid_cos=fid.cos; data.fid_sin=fid.sin;      % arrays to save (dense; name them as on the Python side)
...
timer.start('plot'); figs.noesy=figure('Visible','off'); plot_2d(spin_system,...); timer.stop();
end
```

**Python** `cases/<case>/<case>_spinpy.py`:

```python
from comparison.run_spinpy import load_example
example = load_example("nmr_liquids/<case>")     # reuse the translation in examples/ when there is one

def run(timer, npoints=512, zerofill=2048):
    ...                                           # with timer("basis"): ss = spinpy.basis(ss, bas)
    return {"fid_cos": ..., "fid_sin": ..., "spectrum": ...}, {"noesy": fig}
```

Rules that keep the report meaningful:

- Stage labels are identical on both sides, in the same order; a label on
  one side only shows up with a blank column. Stages do not nest; keep
  them coarse (`create`, `basis`, the context call, `processing`, `plot`)
  and let the profiles show what is inside.
- Save raw FIDs before apodisation, and the final spectrum; name arrays
  identically. MATLAB `cell` and `struct` values cannot be compared; save
  dense arrays.
- Figures: the same Spinach `plot_*` call on both sides, same name.
- Take the calculation from `examples/` (the Spinach example translated
  at the Spinach size) rather than writing it twice; the `--option`
  mechanism provides reduced sizes for quick checks.
- Add the case to `cases` in `run_comparison.m` and, if the case
  mirrors a Spinach example, mention the example in `docs/README.md`.

## Cases

| case | Spinach example | notes |
|---|---|---|
| `noesy_strychnine` | `examples/nmr_liquids/noesy_strychnine.m` | 22 protons, `IK-2`, Redfield, 500 ms NOESY; options `npoints`, `zerofill`, `gpu`; stages `create`, `basis`, `liquid/noesy`, `processing`, `plot`; arrays `fid_cos`, `fid_sin`, `spectrum`; figure `noesy` |
| `zulf_pf6_1d` | `user_ml/ZULF_PF6_1D.m` (contributed) | 31P + 6 x 19F at 2.7 uT, complete basis (16 384 states), lab frame, pulse and 16 384-point acquisition (propagator path after state-space reduction); options `npoints`, `zerofill`, `rho_file`, `gpu`; stages `create`, `basis`, `hamiltonian`, `operators`, `pulse`, `acquisition`, `processing`, `plot`; arrays `rho0`, `rho_pulse`, `fid`, `spectrum`, `coil_x`, `coil_z`; figure `spectrum` |
