# Changelog

Releases are tagged `vX.Y.Z` on GitHub; the version is `spynach.__version__`.

## 0.1.0 - 2026-10-18

First tagged release. Everything listed is verified against Spinach 2.13
fixtures (`docs/parity.md` lists every case, tolerance and measured
difference).

- Hilbert space (`zeeman-hilb`): spin systems, operators and states,
  isotropic Hamiltonians, propagators, pulse-acquire in the `liquid` context.
- Liouville space (`sphten-liouv`): basis with `IK-0`, `IK-1` and `IK-2`
  state-space restriction, projection, longitudinal, zero-quantum and manual
  filters, chemical subsystems and exchange kinetics, commutation and
  product superoperators, coherence and correlation filters, propagation with
  zero track elimination, path tracing, destination state screening and
  Krylov stepping, trajectory analysis (`trajan`, `stateinfo`).
- Anisotropic interactions: shielding and coupling tensors from eigenvalues
  and Euler angles, dipolar couplings from coordinates, quadrupolar couplings
  (`eeqq2nqi`), the irreducible components and `orientation`, the `crystal`
  and `powder` contexts with Spinach's spherical grids.
- Relaxation: Redfield theory for rotational diffusion with the `labframe`,
  `secular`, `kite` and `diagonal` retention policies, T1/T2 and damping
  models, thermal equilibrium with the inhomogeneous master equation and the
  DiBari-Levitt thermalisation, scalar relaxation of the first and second
  kind.
- Magic angle spinning in Fokker-Planck form (`singlerot`).
- Pulse sequences: `acquire`, `cosy`, `noesy`, `hsqc` with isotopomer
  dilution, analytical decoupling, `homospoil`, `apodisation`.
- Plotting: `plot_1d`, `plot_2d`, `plot_fid`, `plot_time_series`,
  `plot_trajan` as interactive Plotly figures, `figure_png` for static copies.
- Verification tooling: the oracle fixture reader and `compare`, 323 parity
  tests over 45 fixture cases, the parity summary generated from the suite,
  and the timed and profiled Spinach-vs-Spynach comparison protocol with HTML
  reports.
- Documentation: seven guides whose code runs in the test suite, each with a
  generated notebook, full-size example scripts, and contributed notebooks in
  `user_py/`.

Not in this release: a PyPI distribution (the name `spynach` on PyPI belongs
to an unrelated project), distribution of the Spinach fixtures, and the GPU
backend.
