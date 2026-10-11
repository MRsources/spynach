# spinpy

A Python reimplementation of the [Spinach](https://spindynamics.org) spin
dynamics kernel (MIT licence), verified stage by stage against the MATLAB
original. **Start with the [Hilbert-space guide](docs/hilbert-space.md)**; if you use
Spinach, also read [Coming from Spinach](docs/coming-from-spinach.md). Each
guide has a runnable Jupyter notebook in [`docs/notebooks/`](docs/notebooks/). See `HANDOFF.md` for goals and design decisions, and
`oracle/README.md` for the reference-data ("fixture") harness.

## Status

Ported and verified against Spinach 2.13 fixtures (`tests/slice/`): the
`zeeman-hilb` pulse-acquire chain for liquid-state NMR with isotropic
interactions. The `sphten-liouv` Liouville-space formalism (basis with state-space
restriction, superoperators, states, Hamiltonian and kinetics
superoperators, analytical filters, propagation with zero track
elimination, path tracing and destination state screening) runs
pulse-acquire and chemical exchange end to end; its setup is verified
against Spinach 2.13 fixtures (`tests/slice/test_liouv_setup_parity.py`)
and its propagation against Spinach 2.13 fixtures as well
(`tests/slice/test_liouv_pa_parity.py`), Hilbert space, and analytic
exchange line shapes. Anisotropic interactions (shielding and coupling
tensors, dipolar couplings from coordinates, quadrupolar couplings) with
the `crystal` and `powder` contexts are ported and verified against
Spinach 2.13 fixtures as well (`tests/slice/test_aniso_parity.py`).
Relaxation (Redfield theory, T1/T2 and damping models, thermalisation)
is ported and checked against the Solomon equations; its Spinach fixtures
are requested (FR-005).

```python
import numpy as np
import spinpy

ss = spinpy.create(
    {"isotopes": ["1H", "1H"], "magnet": 14.1},
    {"zeeman": {"scalar": [1.00, 1.20]}, "coupling": {"scalar": [[0, 12.0], [0, 0]]}},
)
ss = spinpy.basis(ss, {"formalism": "zeeman-hilb", "approximation": "none"})
pars = {"spins": ["1H"], "offset": 660, "sweep": 500, "npoints": 1024,
        "rho0": spinpy.state(ss, "L+", "1H"), "coil": spinpy.state(ss, "L+", "1H")}
fid = spinpy.liquid(ss, spinpy.acquire, pars, "nmr")
spectrum = np.fft.fftshift(np.fft.fft(fid, 4096))
```

| Spinach | spinpy | ported |
|---|---|---|
| `create.m`, `dipolar.m` | `spinpy.create` | isotopes, magnet, tolerances, Zeeman and coupling tensors as scalars, matrices, or eigenvalues with Euler angles, coordinates with point dipolar couplings and the proximity graph, chemical subsystems (`inter.chem` parts, concentrations, exchange rates) |
| `basis.m` | `spinpy.basis` | `zeeman-hilb`; `sphten-liouv` with `none`, `IK-0`, `IK-1`, `IK-2`, the `projections`/`longitudinal`/`zero_quantum`/`manual` filters, and chemical subsystems (no symmetry, no coordinates) |
| `operator.m`, `state.m` | `spinpy.operator`, `spinpy.state` | Hilbert space and `sphten-liouv` superoperators (`left`, `right`, `comm`, `acomm`); states by the `exact`, `cheap` and `chem` methods; `E`, `Lx`, `Ly`, `Lz`, `L+`, `L-`, `Tl,m` |
| `assume.m` | `spinpy.assume` | `nmr`, `esr`, `deer`, `deer-zz`, `labframe`, with `zeeman`/`couplings` retention |
| `hamiltonian.m`, `orientation.m`, `frqoffset.m` | `spinpy.hamiltonian`, `spinpy.orientation`, `spinpy.frqoffset` | isotropic part and the rank-1/rank-2 irreducible components `Q` (CSA, anisotropic couplings, quadrupolar); in `sphten-liouv` as `left`/`right`/`comm`/`acomm` superoperators |
| `euler2dcm.m`, `mat2sphten.m`, `wigner.m`, `eeqq2nqi.m` | `spinpy.conventions` | the same conventions |
| `kinetics.m` | `spinpy.kinetics` | first-order exchange between chemical subsystems |
| `correlation.m`, `coherence.m` | `spinpy.correlation`, `spinpy.coherence` | `sphten-liouv` |
| `propagator.m`, `step.m`, `krylov.m` | `spinpy.propagator`, `spinpy.step`, `spinpy.krylov` | dense Pade below `small_matrix`, scaled-squared Taylor above; Taylor action on vectors with 1-, 2-, 3-point Lie quadratures |
| `evolution.m` | `spinpy.evolution` | `zeeman-hilb`: `observable`; `sphten-liouv`: `observable`, `final`, `trajectory`, `multichannel`, destination state screening, Krylov path |
| `reduce.m`, `zte.m`, `path_trace.m` | `spinpy.reduce`, `spinpy.zte`, `spinpy.path_trace` | zero track elimination, path tracing with subspace merging; no symmetry |
| `relaxation.m`, `equilibrium.m`, `thermalize.m` | `spinpy.relaxation`, `spinpy.equilibrium`, `spinpy.thermalize` | Redfield (isotropic, axial, rhombic diffusion; diagonalisation-free integral), `t1_t2`, `damp`; `rlx_keep`, `rlx_dfs`; `zero`/`IME`/`dibari` thermalisation |
| `liquid.m`, `crystal.m`, `powder.m`, `acquire.m` | `spinpy.liquid`, `spinpy.crystal`, `spinpy.powder`, `spinpy.acquire` | both formalisms; relaxation and exchange kinetics in Liouville space; powder averages over Spinach's grids (shipped as `spinpy/grids`); no decoupling |
| `ft_axis.m`, `axis_1d.m`, `plot_1d.m` | `spinpy.plotting` | all axis units; interactive Plotly figures; no `derivative` |

Anything not ported raises `NotImplementedError`; inputs are never silently
ignored.

## Differences from Spinach

- **Spin indices are 0-based** (`operator(ss, ["Lz", "L+"], [0, 1])` is
  Spinach's `operator(spin_system, {'Lz','L+'}, {1,2})`). Isotope names and
  `'all'`/`'nuclei'`/`'electrons'` work as in Spinach.
- **Product vs. sum operators** are distinguished by the type of the first
  argument: a string (`"Lz"`) gives a sum over spins, a list (`["Lz", "L+"]`)
  a product operator.
- **Spin system objects are frozen dataclasses**; `basis()` and `assume()`
  return modified copies, as Spinach returns a modified struct.
- **Time series are 1-D arrays** (MATLAB returns `n x 1` columns).
- **`evolution()` runs serially**; Spinach's Kuprov-Edwards parallel split
  computes the same observable.

## Accuracy notes

- Spin operators match Spinach to 1 ulp: MATLAB evaluates `2^(-1/2)` one ulp
  below the correctly rounded value, so Spinach's `Lz` is `0.4999999999999999`.
- Spinach rounds Hamiltonian elements to a 1e-10 rad/s grid (`clean_up`) after
  summing terms in random order, so Spinach itself reproduces only to about
  one grid step per element; FIDs therefore agree to ~1e-10 x acquisition time.
- The Taylor propagator's error scales with `tols.prop_chop` and the number of
  squarings: about 7e-8 relative at the default 1e-10 with 7 squarings.

## Development

```
OPENBLAS_NUM_THREADS=4 PYTHONPATH=src python -m pytest -q
```

Parity tests skip when fixtures are absent (`$SPINPY_FIXTURES`, default
`/fixtures`). In this container OpenBLAS sees 12 cores but the cgroup quota
is 4 CPUs; with the default thread count, dense LAPACK calls run ~1000x
slower (a 256x256 `eigh` takes 10 s instead of 10 ms). Set
`OPENBLAS_NUM_THREADS` to the quota for tests and especially for benchmarks.

To test on a macOS host outside the container, install
[uv](https://docs.astral.sh/uv/) (`brew install uv`) and run
`scripts/mac-test.sh` (extra arguments go to pytest). The container's `.venv`
does not work on the host, so the script keeps a separate `.venv-mac` and
syncs it with `pyproject.toml` before every run. Source edits need no
reinstall. Point VS Code's interpreter at `.venv-mac/bin/python`.

Every user-facing change comes with user documentation in the same
commit; see [`docs/README.md`](docs/README.md) for the conventions. After
editing a guide, regenerate its notebook with
`python docs/build_notebooks.py`; `tests/docs/` fails if it is stale.

`src/spinpy/_isotope_data.py` is generated from Spinach's `spin.m` by
`porting/tools/gen_isotopes.py`; do not edit it by hand.
