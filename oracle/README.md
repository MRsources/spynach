# Oracle harness

Reference data ("fixtures") produced by running Spinach in MATLAB, and the
Python code that reads and compares against them. MATLAB runs on the host
only; see `porting/FIXTURE_REQUESTS.md` for the request workflow.

## Rules

- `/spinach` is never modified. Instrument by putting a replacement `.m` file
  in `oracle/matlab/shadow/` ahead of Spinach on the MATLAB path;
  `oracle_meta.m` records every active shadow file with its SHA-256.
- Fixtures are never overwritten. Dump scripts skip case directories that
  already exist, so `run_fixtures` can be re-run after adding a dump script
  and only produces the new cases. A changed dump script means deleting
  its old cases first (they are read-only: `chmod -R u+w` then remove).
- Fixtures live read-only at `/fixtures/<spinach-tag>/<case>/` (override the
  root with `$RESPINACH_FIXTURES`).

## Layout

```
oracle/matlab/lib/oracle_write.m   writes one variable into data.h5
oracle/matlab/lib/oracle_meta.m    writes meta.json (provenance)
oracle/matlab/dumps/*.m            one dump script per vertical slice
oracle/matlab/run_fixtures.m       start script: sets paths, runs all dumps (settings at top)
oracle/run_fixtures.sh             runs run_fixtures.m in MATLAB batch mode
oracle/matlab/shadow/              path-shadowing instrumentation (isworkernode.m: serial evolution())
src/respinach/oracle/                 Python loader (fixture.py) and comparison (compare.py)
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
`spinach_commit`, `spinach_describe`, `spinach_dirty` (modified tracked files
or untracked `.m` files; other untracked files such as `.DS_Store` are ignored),
`spinach_status` (raw `git status --porcelain` lines), `matlab_version`,
`formalism`, `approximation`, `tols` (the full `spin_system.tols`),
`sys_enable`, `sys_disable`, `dump_script` (path, sha256), `harness`
(sha256 of `oracle_write.m` and `oracle_meta.m`), `shadows`, and `inputs`
(the user-level inputs of the case). MATLAB's `jsonencode` writes Inf and NaN
as `null`.

## Python usage

```python
from respinach.oracle import find_case, load_fixture, assert_matches

fx = load_fixture(find_case("hilb_pa_ab_2spin"))
H_ref = fx["/ham/H_off"]  # scipy sparse
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

### `dump_liouv_setup.m`: sphten-liouv setup

Twelve cases `liouv_<name>` (see `porting/FIXTURE_REQUESTS.md`, FR-002)
covering complete bases (including a spin-1), `IK-0` at three levels,
`IK-1` and `IK-2` on styrene, a twelve-spin chain, two-site exchange with
equal and unequal concentrations, and the longitudinal/zero-quantum
filters. Nothing is propagated. Variables:

| path | content |
|------|---------|
| `/comp/mults` | spin multiplicities |
| `/bas/basis`, `/bas/tot_proj`, `/bas/tot_cord` | basis descriptor (sparse double), coherence and correlation orders |
| `/bas/lpst_<mult>`, `/bas/rpst_<mult>` | spherical tensor product tables of every multiplicity present |
| `/ops/<op>_<type>` | 1×nspins cells of single-spin superoperators, `op` in `E, Lz, Lx, Ly, Lp, Lm`, `type` in `left, right, comm, acomm` |
| `/ops/LzLp_12_<type>` | product superoperator Lz(1)L+(2) |
| `/ops/Lz_det_comm`, `/ops/Lp_det_left` | sums over the detected isotope |
| `/states/Lz_exact`, `/states/Lp_exact`, `/states/Lz_chem`, `/states/Lp_chem` | `state(...,method)` on the detected isotope |
| `/states/Lz_cheap`, `/states/Lp_cheap` | where the state is in the basis (`inputs.cheap` in `meta.json`) |
| `/states/LzLp_12_exact` | product state |
| `/ham/H`, `/ham/H_zeeman`, `/ham/H_coupling` | commutation superoperators of the isotropic Hamiltonian and its parts |
| `/ham/H_left`, `/ham/H_right`, `/ham/H_off` | `hamiltonian(...,'left'/'right')`, `H` after `frqoffset` |
| `/kin/K` | `kinetics(spin_system)` (zero without exchange) |
| `/filters/input`, `/filters/corr_12`, `/filters/corr_1_det`, `/filters/coh_det_p1`, `/filters/coh_all_0` | a dense test vector and `correlation`/`coherence` applied to it |

### `dump_liouv_pa.m`: sphten-liouv propagation

Nine cases `liouv_pa_<name>` (FR-003): the complete-basis and restricted
systems of FR-002, two-site exchange, one case with the Krylov path forced
(`tols.krylov_tol=1`) and one with path tracing forced (`tols.merge_dim=2`).
Variables:

| path | content |
|------|---------|
| `/bas/basis` | basis descriptor |
| `/ham/L` | `frqoffset(hamiltonian(assume(...,'nmr')))+1i*kinetics(...)` |
| `/states/rho0`, `/states/coil` | `state(...,'L+',det,'chem')` and `state(...,'L+',det)` |
| `/prop/P` | `propagator(L, 1/sweep)` |
| `/step/rho_dt`, `/step/rho_long`, `/step/stack` | `step` over one dwell time, over 7.3 dwell times, and on the stack `[rho0 0.5*coil]` |
| `/reduce/zte_projector`, `/reduce/projectors` | `zte(L',coil)` and `reduce(L',coil)` (destination state screening as `evolution` applies it) |
| `/fid/evolution`, `/fid/final`, `/fid/trajectory` | `evolution` outputs `'observable'` (npoints), `'final'` (after npoints-1 steps), `'trajectory'` (32 points) |
| `/fid/liquid`, `/spectrum/raw` | `liquid(spin_system,@acquire,pars,'nmr')` and its unapodised spectrum |

### `dump_aniso.m`: anisotropic interactions

Five cases `aniso_<name>` (FR-004): a CSA pair, a dipolar pair from
coordinates, two quadrupolar 14N (Hilbert space), a 1H-13C pair with
tensor shielding, anisotropic J and a dipolar coupling (Hilbert space),
and a nitroxide-like electron with a 14N hyperfine tensor under `'esr'`.
Variables:

| path | content |
|------|---------|
| `/inter/zeeman_matrix`, `/inter/coupling_matrix`, `/inter/proxmatrix` | `create()` output |
| `/bas/basis` | basis descriptor |
| `/ham/I`, `/ham/Q1`, `/ham/Q2` | `[I,Q]=hamiltonian(...)`: isotropic part, 3×3 and 5×5 cells of rank-1 and rank-2 components |
| `/ham/I_off`, `/ham/H_orient_1`, `/ham/H_orient_2` | `I` after `frqoffset`, and `(H+H')/2` with `H=I_off+orientation(Q,angles)` at `[0 0 0]` and `[0.3 0.9 1.2]` |
| `/conv/wigner_1..3` | `wigner(l,0.3,0.9,1.2)` |
| `/grids/<name>` | `[alphas betas gammas weights]` of `single_crystal`, `leb_2ang_rank_11`, `rep_2ang_100pts_sph` |
| `/states/rho0` | `state(...,'L+',det)` |
| `/fid/crystal` | `crystal(...)` at `[0.3 0.9 1.2]` |
| `/fid/powder_leb11`, `/fid/powder_rep100`, `/spectrum/powder_rep100` | `powder(...)` on the two grids (serial), and the unapodised spectrum |

### `dump_relax.m`: relaxation

Five cases `relax_<name>` (FR-005): heteronuclear and homonuclear NOE pairs
(Redfield, `kite`, DiBari-Levitt), a three-spin system with CSA and dipolar
couplings (`secular`, IME), axial rotational diffusion with dynamic
frequency shifts kept (`labframe`), and the extended T1/T2 model plus
anisotropic damping (`diagonal`). Variables:

| path | content |
|------|---------|
| `/bas/basis` | basis descriptor |
| `/rlx/R`, `/rlx/R_oriented` | `relaxation(spin_system)` and `relaxation(spin_system,[0.3 0.9 1.2])` |
| `/ham/HL` | lab-frame Hamiltonian as a left-side product superoperator |
| `/states/rho_eq`, `/states/rho_eq_oriented` | `equilibrium(spin_system,HL)` and with `QL` at the orientation |
| `/rlx/R_zero`, `/rlx/R_ime`, `/rlx/R_dibari` | the superoperator with `equilibrium='zero'` and its two thermalisations |
| `/states/rho_inverted`, `/noe/multichannel` | spin 1 inverted, and `evolution(...,1i*R,coil,rho,dt,nsteps,'multichannel')` with `Lz` of every spin as coils |

### `dump_scalar.m`: scalar relaxation

Three cases `scalar_<name>` (FR-008): SRFK alone on two protons with a
15 Hz modulated coupling (`kite`), Redfield plus SRSK from a fast-relaxing
quadrupolar 14N onto a scalar-coupled proton (`secular`), and all three
theories on a three-spin system with CSA and a quadrupolar tensor at an
orientation (`kite`). Variables:

| path | content |
|------|---------|
| `/bas/basis` | basis descriptor |
| `/rlx/R` | `relaxation(spin_system)` with every theory |
| `/rlx/R_base` | the same without `SRFK`/`SRSK` (absent for the SRFK-only case) |
| `/ham/HL`, `/states/rho_eq` | lab-frame left-side Hamiltonian and `equilibrium(spin_system,HL)` |
| `/states/rho_inverted`, `/noe/multichannel` | spin 1 inverted, and the `multichannel` evolution under `1i*R` with `Lz` of every spin as coils |

### `dump_mas.m`: magic angle spinning (`singlerot`)

Four cases `mas_<name>` (FR-006): the CSA, dipolar and 2H quadrupolar MAS
examples at reduced rotor rank and grid, and a single crystal spinning
about an axis off the magic angle with a transmitter offset. Variables:

| path | content |
|------|---------|
| `/bas/basis` | basis descriptor |
| `/fp/rotor_phases`, `/fp/d_dphi` | `fourdif(2*max_rank+1,1)` |
| `/fp/rotor_angles`, `/fp/first_orientation` | rotor polar angles from `parameters.axis`; first grid orientation |
| `/fp/G_first_orientation` | the Fokker-Planck generator `blkdiag(H(phi))+1i*M` at the first orientation, as `singlerot` builds it |
| `/states/rho0_fp_crystal`, `/states/coil_fp` | the expanded states handed to the sequence (`single_crystal` grid for `rho0`) |
| `/fid/singlerot`, `/spectrum/raw` | `singlerot(spin_system,@acquire,pars,'nmr')` and its unapodised spectrum |

### `dump_seq.m`: pulse sequences

Three cases (FR-007). `seq_cosy_derome`: `cosy` FID, its apodised form and
spectrum, apodisation of known vectors/matrices (`/apod/*`), `'trajectory'`
and `'refocus'` outputs (`/evo/*`), `acquire` with `dead_time`, and
`homospoil` on a stack. `seq_noesy_pair`: `noesy` cos/sin FIDs, the
`oldschool` variant, `rho_eq`. `seq_hsqc_dilute`: per-isotopomer `hsqc`
pos/neg FIDs and bases (`/sub/*`, `/fid/hsqc_*_n`), the summed spectrum,
and `decouple` of a Liouvillian and a state.
