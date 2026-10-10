# Fixture requests

Requests for a human to run MATLAB dump scripts on the host. The agent adds
entries; the human runs them and fills in the result. Fixtures are written
read-only under `/fixtures/<spinach-tag>/`.

## FR-001: zeeman-hilb pulse-acquire slice (`dump_hilb_pa.m`)

- **Status:** requested
- **Requested:** 2026-10-08, branch `oracle-harness`
- **Spinach:** clean checkout, expected `03a88b7e912ffbb3122dd24829cc5d216209c804` (no local modifications; `oracle_meta.m` records the dirty flag)
- **Needs:** MATLAB R2024b or later with Parallel Computing Toolbox (`create()` starts a pool). The shadow `oracle/matlab/shadow/isworkernode.m` sends `evolution()` down its serial branch, because the Hilbert-space `spmd` split uses `distributed` arrays, which MATLAB does not support on local process pools on Apple silicon Macs (`distributedutil.errorIfUnsupportedPlatformOrEnvironment`; this is a MATLAB restriction, not a Spinach one). Spinach's startup check rejects the shadow, so the dump sets `sys.disable={'hygiene'}`.
- **Before re-running:** delete the partial `hilb_pa_ab_2spin/` left by the failed run of 2026-10-10; the dump refuses to overwrite an existing case directory.
- **Produces:** `hilb_pa_ab_2spin`, `hilb_pa_abx_3spin`, `hilb_pa_hc_2spin`, `hilb_pa_hd_2spin` (each a few MB at most)

Edit the settings block at the top of `oracle/matlab/run_fixtures.m` if the
defaults do not match the host (repository worktree, Spinach directory, fixture
output root), then either run it from a shell:

```
oracle/run_fixtures.sh            # MATLAB=/path/to/matlab to override the executable
```

or from MATLAB:

```matlab
run('<repo>/oracle/matlab/run_fixtures.m')
```

It puts Spinach (`kernel`, `etc`, `experiments`, `interfaces`) and the harness
on the path, checks that `create()` resolves to that Spinach copy, writes to
`<fixtures_root>/<spinach tag>/`, and makes the new case directories read-only.

Progress: each stage of each case prints a timestamped line, e.g.
`2026-10-09 14:02:11 [abx_3spin 2/4] propagator`. The full console output,
including any error report, is saved as `<fixtures_root>/<tag>/run_<UTC time>.log`,
and Spinach's own messages for each case go to `<case>/spinach.log`.
To follow a batch run from another terminal: `tail -f <fixtures_root>/<tag>/run_*.log`.

Then check on the Python side (with `/fixtures` mapped to `fixtures_root`):

```
SPINPY_FIXTURES=/fixtures pytest tests/oracle -rs
```

All tests in `test_hilb_pa_fixtures.py` should pass, not skip. A failure in
`test_layout_probes` means the HDF5 layout is wrong, not Spinach; report it
before going further.

- **Result:** _(human: fixture path, MATLAB version, any errors or warnings)_
