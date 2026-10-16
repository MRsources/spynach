# Project handoff: context from the planning conversation

Read this once at the start of Session 1. It records decisions and findings made
before the repo existed, so they do not have to be rediscovered.

## Goal
A Python reimplementation of the Spinach kernel that is readable, maintainable,
verified against the MATLAB original, and able to run on GPU. Secondary goals,
after the verified core exists: a differentiable engine for optimal control and
fitting, and possibly a tensor-train backend. These are NOT phase 1.

## Facts established by reading the Spinach source (current master, Oct 2026)
- License is MIT. Attribution required, nothing else.
- Kernel is ~73k lines across ~1,700 .m files; ~780 example scripts under
  `examples/`. The examples are the verification corpus.
- Three formalisms exist: `zeeman-hilb`, `zeeman-liouv`, `sphten-liouv`.
  Phase 1 ports `zeeman-hilb` first (simplest, dense), then `sphten-liouv`.
- GPU use in Spinach is via MATLAB `gpuArray`, opt-in with `sys.enable={'gpu'}`:
  - `propagator.m`: for dim > 500, scaled Taylor series with sparse-SPARSE
    products on GPU, `clean_up` thresholding between terms, then squaring (P*P).
  - `step.m`, `krylov.m`, `evolution.m`: sparse-dense products (SpMV/SpMM).
  - Krylov path is used automatically above `tols.krylov_tol` (default 10,000).
  - Basis/operator construction and relaxation are CPU-only (sparse kron, rcv
    triplets, small C++ MEX: spsortrows, spunicols). No CUDA sources exist.
- `clean_up.m` switches sparse→dense above `dense_matrix` density (0.15) with a
  `small_matrix` floor (200). Propagators routinely densify after squaring, so
  dense complex performance matters as much as sparse.
- `kernel/overloads/@polyadic` is a lazy Kronecker-product operator class with
  gpuArray overloads; currently used in Fokker-Planck contexts (MAS, imaging).
  It is the natural hook for a matrix-free Zeeman backend later.
- `AGENTS.md` in the Spinach repo describes the authors' expectations for agents.

## Physics/representation decisions
- Zeeman product basis and the spherical-tensor (IST) product basis are related
  by a local unitary; the difference that matters is truncation. Spinach's
  restricted state space truncates by spin correlation order, which is a label
  on IST product-basis elements and NOT on Zeeman Liouville elements |i><j|.
  Conclusion: Zeeman-Hilbert for small systems (dense, no truncation);
  IST-Liouville with correlation/coherence-order truncation for large liquid-state
  systems. Both behind one basis abstraction. Do not try to make one encoding
  serve both regimes.
- Zero track elimination and destination state screening are basis-agnostic
  and belong in a shared layer.
- Tensor-train (MPO) compression is a published alternative (Savostyanov, Dolgov,
  Kuprov et al., Phys. Rev. B 90, 085139, 2014): wins on chain-like coupling
  topologies, loses on dense ones; strongest in the frequency domain with a
  linear solver. Candidate third backend, not phase 1.

## Numerical backend decisions
- Reference: NumPy/SciPy on CPU. GPU: CuPy, via a thin `spynach.backend` adapter
  (array module + sparse adapter). PyTorch rejected for the core (complex sparse
  support partial); JAX rejected for the sparse engine (experimental sparse,
  static shapes vs. adaptive pruning) but planned later as a separate dense,
  differentiable backend for optimal control.
- Numba for index-heavy construction loops (replaces the MEX files).
- dtype is a parameter. complex128 default and reference. Mixed precision is the
  planned optimization: build operators/propagators in double on CPU, optionally
  propagate in complex64 on GPU, accumulate observables in double. The parity
  harness run at complex64 will determine where this is acceptable.
- Spinach's thresholds (`small_matrix`, `dense_matrix`, `krylov_tol`) become
  per-backend tunables, not constants.
- Open performance risk to benchmark early: CuPy sparse-sparse product speed in
  the propagator Taylor loop (dim 500–10,000). Fallbacks: densify earlier on GPU,
  or lower the expm_multiply/Krylov threshold.

## Operating model
- MATLAB runs only on the host. The agent writes `oracle/matlab/*.m` dump scripts
  and requests fixtures in `porting/FIXTURE_REQUESTS.md`; a human runs them and
  the HDF5 output appears read-only at `/fixtures/<spinach-tag>/<example>/`.
- Never patch `/spinach`. Instrument by MATLAB path shadowing from
  `oracle/matlab/shadow/`, and record it in the fixture `meta.json`.
- Fixture format: complex arrays as separate real/imag datasets; sparse as COO
  triplets + shape; `meta.json` with Spinach commit, MATLAB version, formalism,
  active tolerances, dump-script hash.
- Agent commits on feature branches in `.worktrees/`; humans merge. Distinct git
  author identity for agent commits.

## Phase 1 scope (first 6–8 weeks)
1. Oracle harness: dump scripts + Python loader + comparison library.
2. Vertical slice in zeeman-hilb: create → basis → operator → hamiltonian (iso) →
   state → step (dense) → evolution, pulse-acquire spectrum of 2–3 spin systems.
3. sphten-liouv basis with truncation; sparse Liouvillian; the three propagation
   paths (Taylor expm with thresholding, Taylor-times-vector, expm_multiply);
   CuPy switched on; first benchmarks.
4. Consolidation and review. Decide next: relaxation vs. JAX layer vs. matrix-free Zeeman.
Out of scope for phase 1: relaxation, anisotropic interactions, optimal control, TT backend.
