# Roadmap

Phase 1 as laid out in `HANDOFF.md`, revised 2026-10-11 after the literature
and source review in `design-notes.md`. Steps 1 and 2 are done (oracle
harness, FR-001; `zeeman-hilb` pulse-acquire slice, verified against Spinach
2.13). Step 3 is narrowed to the Liouville-space *setup*: basis, operators,
states, Hamiltonian and kinetics superoperators, verified against cheap
fixtures. Propagation becomes step 4.

## Step 3: `sphten-liouv` setup and Hamiltonians

Scope: everything up to and including the Liouvillian, nothing that
propagates. Each sub-step is a commit with tests and docs.

Status (2026-10-11): done. 3a–3f on `main` (commits `68b0626` onwards);
FR-002 fulfilled, 132 parity tests pass (agreement table in
`docs/liouville-space.md`). `IK-1`/`IK-2` with proximity graphs remain
untestable until coordinates are ported. Next: step 4.

### 3a. Spin system: chemistry fields and the generalised `Basis`

- `create()`: accept `inter['chem']` with `parts` (0-based spin index
  lists), `concs`, `rates`; validate as `create.m` does (every spin in
  exactly one part, parts sorted, `concs` required for >1 part, no
  couplings between parts, `rates` square with zero column sums and equal
  part sizes). `flux_rate/flux_type`, `rp_*` and nonlinear reactions stay
  `NotImplementedError`. Default: one part with all spins, `concs = [1]`.
- `Basis` dataclass grows: `formalism`, `approximation`, `basis` (int array
  `(nstates, nspins)`; for `zeeman-hilb` keep the current `levels`),
  `tot_proj`, `tot_cord`, cached product tables. Keep frozen dataclasses.

### 3b. `basis()` for `sphten-liouv`

- Approximations `none`, `IK-0`, `IK-1`, `IK-2` with the 2.13 field names
  `inter_level`, `prox_level`, `connectivity` (`'scalar_couplings'` and
  `'full_tensors'`), `projections`, `longitudinal`, `zero_quantum`, `manual`
  — all per substance exactly as `basis.m` (design-notes §3). `sym_group`,
  `IK-DNP`, `IK-SBS` and `zeeman-liouv` raise `NotImplementedError`; the
  old field names raise the same rename errors as Spinach.
- Helpers: `dfpt`, `prune_subgraphs`, connectivity matrix (reuse
  `hamiltonian._coupled_pairs` logic for the norm), proximity matrix
  (identity; coordinates not ported).
- Unit tests without fixtures: state counts for `none` (`Π mult²`), IK-0
  (`Σ_{j≤k} C(n,j)·3^j` for n spin-1/2 at `inter_level` k), two substances (`1 + 3 + 3`), the
  7-state example, projections keeping the unit row, longitudinal reducing
  per-spin lists to `l² + l`.

### 3c. Product tables, `operator()`, `state()`

- `ist_product_table(mult)` from `irr_sph_ten`, cached; unit-tested against
  the hard-coded spin-1/2 tables in `ist_product_table.m`.
- `operator(ss, ops, spins, operator_type='comm')` dispatching on
  formalism; the sphten branch is the vectorised `superop` of design-notes
  §5.3(4). `left`, `right`, `comm`, `acomm`.
- `state(ss, ops, spins, method='exact')` with `exact`, `cheap`, `chem`.
- Cross-formalism tests without fixtures: for complete bases,
  `Pᴴ (kron(I,H) − kron(Hᵀ,I)) P / dim` equals the sphten commutation
  superoperator, with `P` from the `sphten2zeeman` construction; `exact`
  states equal `Pᴴ vec(Op)/√dim`-type images of the Hilbert-space operators.

### 3d. `hamiltonian()`, `frqoffset()`, `kinetics()`, filters

- `hamiltonian(ss, operator_type='comm')` returns the isotropic
  commutation superoperator in Liouville space (same term logic, new
  `operator_type` plumbing, term table keeps slots for `Q`). `frqoffset`
  unchanged apart from dispatch.
- `kinetics(ss)` from `chem.rates` (design-notes §5.1); unit test: column
  sums of `K` vanish (Spinach's `test_kinetics_generator_suite.m`).
- `correlation(ss, rho, orders, spins)` and `coherence(ss, rho, spec)`.
- `liquid()` still refuses `sphten-liouv` until step 4 (it would need
  `evolution`); say so in the error message.

### 3e. Fixtures: `dump_liouv_setup.m`, FR-002

All cases are seconds in MATLAB: no propagation, only
`create → basis → operator → state → hamiltonian → frqoffset → kinetics`.
Variables per case (same HDF5 conventions as FR-001):

| path | content |
|---|---|
| `/bas/basis`, `/bas/tot_proj`, `/bas/tot_cord` | basis descriptor and derived orders |
| `/bas/lpst_<mult>`, `/bas/rpst_<mult>` | product tables for every multiplicity present |
| `/ops/<name>_<type>` | `E, Lz, Lx, Ly, L+, L-` per spin and a two-spin product, for `left/right/comm/acomm` |
| `/states/<op>_<method>` | `Lz`, `L+` on the detected isotope for `exact`, `cheap`, `chem` |
| `/ham/H`, `/ham/H_zeeman`, `/ham/H_coupling`, `/ham/H_off` | commutation superoperators |
| `/ham/H_left`, `/ham/H_right` | `hamiltonian(...,'left'/'right')` |
| `/kin/K` | `kinetics(spin_system)` where `chem.rates` exists |
| `/filters/corr_<orders>`, `/filters/coh_<spec>` | filter outputs on a dense test vector |
| `/conv/P_sphten2zeeman` | for complete bases only |

Cases (Spinach examples in parentheses; all `B0 = 14.1 T` unless the
example says otherwise, `assume 'nmr'`):

| case | system | `bas` | exercises |
|---|---|---|---|
| `liouv_ab_none` | 2×1H (reuse FR-001 `ab_2spin`) | `none` | complete basis, cross-check with Hilbert fixtures |
| `liouv_hd_none` | 1H + 2H (FR-001 `hd_2spin`) | `none` | mult-3 tables, spin-1 states |
| `liouv_inad_none` | 3×13C (`nmr_liquids/inad_three_spin.m`) | `none` | 64 states, heteronuclear sums |
| `liouv_styrene_ik0_<k>` | 8×1H (`pa_styrene.m`) | `IK-0`, `inter_level` 1, 2, 4 | correlation-order truncation, state counts |
| `liouv_styrene_ik1` | 8×1H | `IK-1`, `inter_level 3`, `prox_level 1`, `scalar_couplings` | `dfpt` on the J graph |
| `liouv_styrene_ik2_proj` | 8×1H | `IK-2`, `prox_level 1`, `projections {[1]}` | star subgraphs, coherence filter |
| `liouv_chain_ik2` | 12×1H chain, J = 10 Hz nearest neighbour (`product_quadratures_1.m`, shortened from 31) | `IK-2`, `prox_level 1` | sparse large basis, timing |
| `liouv_exch_sym` | 2×1H, 2 parts (`kinetics/exchange_symmetric.m`) | `none`, `rates ±2e3`, `concs [1 1]` | segmented basis, `kinetics`, `chem` states |
| `liouv_exch_asym` | same, `rates [-5e2 2e3; 5e2 -2e3]`, `concs [2e3 5e2]` | `none` | unequal concentrations |
| `liouv_hcn_long` | 1H, 13C, 15N with J | `none`, `longitudinal {{'15N','13C'}}`, `zero_quantum {{'13C'}}` | longitudinal and zero-quantum filters |

`liouv_chain_ik2` is the only case with a non-trivial basis size; keep it
under a second. Write the dump, add FR-002 to `FIXTURE_REQUESTS.md`, and
the parity tests in `tests/slice/test_liouv_setup_parity.py`.

### 3f. Documentation

- Keep `docs/user-guide.md` as the Hilbert-space guide (rename to
  `hilbert-space.md` when the Liouville guide lands, so the two sit side by
  side in `docs/README.md`; `coming-from-spinach.md` stays the cross-cutting
  comparison). Add `docs/liouville-space.md` with its own notebook:
  basis descriptor and state counts, operators and the four superoperator
  types, states and normalisation, Hamiltonian commutation superoperator,
  restriction (`IK-0/1/2`, filters) with the per-order norm plot and
  state-count tables, two-site exchange setup. Nothing in it propagates
  until step 4.
- `coming-from-spinach.md`: basis section (field renames, per-substance
  cells, 0-based `parts`), operators (`operator_type`), states (`method`),
  kinetics; move items out of the not-ported lists; README table.

## Step 4: propagation in Liouville space

Status (2026-10-11): `step`, `krylov`, `evolution` ('observable', 'final',
'trajectory', destination state screening), `reduce`/`zte`/`path_trace`,
`liquid` with kinetics are on `main` (commit `3fd6f81`), verified against
Hilbert space and the McConnell equations; FR-003 fulfilled, 54 parity
tests pass (agreement 1e-13 or better, table in `docs/liouville-space.md`).
Deferred: the CuPy backend (no GPU in the development container; needs a
host with CUDA to test) and the benchmarks that depend on it.

`step` (Krylov-style Taylor action on a vector), `evolution` for the
Liouville formalisms with `reduce` (ZTE, path tracing, DSS; symmetry still
refused), `krylov`, Taylor propagator with thresholding and the
`krylov_tol` switch, pulse-acquire through `liquid`/`acquire` with
`H + i·K`; fixtures `dump_liouv_pa.m` (FR-003) reusing the step-3 cases plus
exchange line shapes. Then the `spinpy.backend` adapter and CuPy, with the
first benchmarks on `liouv_chain_ik2`-style systems at 500–10 000 states.

## Step 5: consolidation

Review; decide between relaxation, symmetry, anisotropic `Q`, the polyadic
container, and the JAX layer.
