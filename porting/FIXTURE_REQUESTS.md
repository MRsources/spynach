# Fixture requests

Requests for a human to run MATLAB dump scripts on the host. The agent adds
entries; the human runs them and fills in the result. Fixtures are written
read-only under `/fixtures/<spinach-tag>/`.

## FR-001: zeeman-hilb pulse-acquire slice (`dump_hilb_pa.m`)

- **Status:** requested
- **Requested:** 2026-10-08, branch `oracle-harness`
- **Spinach:** clean checkout, expected `03a88b7e912ffbb3122dd24829cc5d216209c804` (no local modifications; `oracle_meta.m` records the dirty flag)
- **Needs:** MATLAB R2024b or later. Parallel Computing Toolbox is used by `evolution.m` (Hilbert-space `spmd` split) if installed.
- **Produces:** `hilb_pa_ab_2spin`, `hilb_pa_abx_3spin`, `hilb_pa_hc_2spin`, `hilb_pa_hd_2spin` (each a few MB at most)

Run from MATLAB, with Spinach already on the path as usual on the host:

```matlab
addpath('<repo>/oracle/matlab/lib', '<repo>/oracle/matlab/dumps')
[~, tag] = system('git -C /spinach describe --tags --always'); tag = strtrim(tag);
out_root = fullfile('/fixtures', tag); mkdir(out_root)
dump_hilb_pa(out_root)
```

Then make the case directories read-only, and check on the Python side:

```
SPINPY_FIXTURES=/fixtures pytest tests/oracle -rs
```

All tests in `test_hilb_pa_fixtures.py` should pass, not skip. A failure in
`test_layout_probes` means the HDF5 layout is wrong, not Spinach; report it
before going further.

- **Result:** _(human: fixture path, MATLAB version, any errors or warnings)_
