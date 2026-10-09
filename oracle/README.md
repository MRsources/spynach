# Oracle harness

Reference data ("fixtures") produced by running Spinach in MATLAB, and the
Python code that reads and compares against them. MATLAB runs on the host
only; see `porting/FIXTURE_REQUESTS.md` for the request workflow.

## Rules

- `/spinach` is never modified. Instrument by putting a replacement `.m` file
  in `oracle/matlab/shadow/` ahead of Spinach on the MATLAB path;
  `oracle_meta.m` records every active shadow file with its SHA-256.
- Fixtures are never overwritten. Dump scripts refuse to write into an
  existing case directory; a changed dump script means a new fixture run.
- Fixtures live read-only at `/fixtures/<spinach-tag>/<case>/` (override the
  root with `$SPINPY_FIXTURES`).

## Layout

```
oracle/matlab/lib/oracle_write.m   writes one variable into data.h5
oracle/matlab/lib/oracle_meta.m    writes meta.json (provenance)
oracle/matlab/dumps/*.m            one dump script per vertical slice
oracle/matlab/run_fixtures.m       start script: sets paths, runs all dumps (settings at top)
oracle/run_fixtures.sh             runs run_fixtures.m in MATLAB batch mode
oracle/matlab/shadow/              path-shadowing instrumentation (empty so far)
src/spinpy/oracle/                 Python loader (fixture.py) and comparison (compare.py)
```

Each case directory holds `data.h5` and `meta.json`.

**data.h5.** Every variable is an HDF5 group with attributes `kind`, `shape`
(MATLAB size, always at least 2-D) and, for numeric kinds, `complex` (0/1).

| kind     | datasets                                  | Python value               |
|----------|-------------------------------------------|----------------------------|
| `dense`  | `re`, `im` if complex; C-order shape = MATLAB shape | `np.ndarray`     |
| `sparse` | `row`, `col` (0-based int64), `re`, `im`; attribute `nnz` | `scipy.sparse.csr_array` |
| `cell`   | members `0`, `1`, ... (0-based column-major linear index) | object `np.ndarray` |

Datasets are omitted for empty arrays and for sparse matrices with `nnz=0`.
Real data stays `float64`; complex data is `complex128`. Shapes are kept as
in MATLAB, so a column vector loads as `(n, 1)`.

Every dump also writes `/probe/*` variables with known contents;
`tests/oracle/test_hilb_pa_fixtures.py::test_layout_probes` checks them, which
confirms the MATLAB-side layout on real output.

**meta.json.** `schema_version`, `created_utc`, `spinach_root`,
`spinach_commit`, `spinach_describe`, `spinach_dirty`, `matlab_version`,
`formalism`, `approximation`, `tols` (the full `spin_system.tols`),
`sys_enable`, `sys_disable`, `dump_script` (path, sha256), `harness`
(sha256 of `oracle_write.m` and `oracle_meta.m`), `shadows`, and `inputs`
(the user-level inputs of the case). MATLAB's `jsonencode` writes Inf and NaN
as `null`.

## Python usage

```python
from spinpy.oracle import find_case, load_fixture, assert_matches

fx = load_fixture(find_case("hilb_pa_ab_2spin"))
H_ref = fx["/ham/H_off"]          # scipy sparse
assert_matches(my_H, H_ref, rtol=1e-12, atol=0, name="H_off")
```

`compare`/`assert_matches` use a normwise test,
`||a - e||_F <= atol + rtol * ||e||_F`. Tolerances are always explicit.

## Dump scripts

### `dump_hilb_pa.m`: zeeman-hilb pulse-acquire

Cases `hilb_pa_ab_2spin`, `hilb_pa_abx_3spin`, `hilb_pa_hc_2spin`,
`hilb_pa_hd_2spin` (14.1 T, scalar couplings only, `assume 'nmr'`, no symmetry).
Variables:

| path | content |
|------|---------|
| `/comp/mults` | spin multiplicities |
| `/inter/basefrqs`, `/inter/zeeman_matrix`, `/inter/coupling_matrix` | `create()` output |
| `/ops/{E,Lz,Lx,Ly,Lp,Lm}` | 1×nspins cells of single-spin operators |
| `/ops/LzLp_12` | product operator Lz(1)L+(2) |
| `/ops/Lz_det`, `/ops/Lp_det` | sums over the detected isotope |
| `/ham/H`, `/ham/H_zeeman`, `/ham/H_coupling` | isotropic rotating-frame Hamiltonian and its parts |
| `/ham/H_off` | `H` after `frqoffset` |
| `/states/rho0`, `/states/coil` | `state(...,'L+',det)` |
| `/prop/P` | `propagator(H_off, 1/sweep)` |
| `/fid/explicit` | loop `fid(k)=hdot(coil,rho); rho=P*rho*P'` |
| `/fid/evolution` | `evolution(...,'observable')` |
| `/fid/liquid` | `liquid(spin_system,@acquire,pars,'nmr')` |
| `/spectrum/raw` | `fftshift(fft(fid_liquid,zerofill))` |
