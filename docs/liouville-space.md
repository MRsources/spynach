# Liouville space: the spherical-tensor formalism

This guide covers the `sphten-liouv` formalism: Liouville space built on
products of irreducible spherical tensors, with the state-space restriction
that lets Spinach simulate large spin systems. Read
[Liquid-state NMR in Hilbert space](hilbert-space.md) first; `create`,
`assume`, the experiment parameters and the processing are the same. The
companion notebook is
[`notebooks/liouville-space.ipynb`](notebooks/liouville-space.ipynb).

Ported: the basis set with its restrictions, operators (superoperators),
states, the Hamiltonian and kinetics superoperators, the analytical
filters, and propagation with trajectory-level reduction (`step`,
`evolution`, `liquid`), so pulse-acquire experiments and chemical
exchange run end to end.

Every Python block is executed by `tests/docs/`, in order and in one
namespace.

## Contents

1. [Why Liouville space](#why-liouville-space)
2. [The basis descriptor](#the-basis-descriptor)
3. [State-space restriction](#state-space-restriction)
4. [Filters](#filters)
5. [Chemical subsystems](#chemical-subsystems)
6. [Operators: superoperators in the restricted basis](#operators-superoperators-in-the-restricted-basis)
7. [States](#states)
8. [The Liouvillian: Hamiltonian and kinetics superoperators](#the-liouvillian-hamiltonian-and-kinetics-superoperators)
9. [Analytical filters](#analytical-filters)
10. [Propagation: `step`, `evolution`, and the reductions](#propagation-step-evolution-and-the-reductions)
11. [Pulse-acquire and chemical exchange through `liquid`](#pulse-acquire-and-chemical-exchange-through-liquid)
12. [What is not ported yet](#what-is-not-ported-yet)
13. [Checking results against Spinach](#checking-results-against-spinach)

## Why Liouville space

In Hilbert space the density matrix of `n` spin-1/2 particles has `4^n`
elements and nothing can be left out. In Liouville space the density matrix
is a vector in the space of operators, and a basis of that space can be
chosen so that each basis operator has a physical label: how many spins it
correlates and what coherence order it carries. Spinach's observation
(Kuprov et al. 2007; Karabanov et al. 2011) is that in liquid-state NMR
high spin correlations are never populated to a significant extent, so
dropping them makes the matrix dimension polynomial rather than exponential
in the number of spins, while relaxation and kinetics superoperators can be
added to the same generator.

```python
import numpy as np
import spinpy


def spin_system(isotopes, **inter):
    return spinpy.create({"isotopes": isotopes, "magnet": 14.1}, inter)
```

## The basis descriptor

Each spin of multiplicity `2S + 1` has `(2S + 1)^2` single-spin irreducible
spherical tensor operators `T(l, m)`, numbered by the linear index
`l*l + l - m`: 0 is the unit operator, 1, 2, 3 are `T(1,+1)`, `T(1,0)`,
`T(1,-1)` (proportional to `L+`, `Lz`, `L-`), 4 to 8 the rank-2 tensors of a
spin-1, and so on. A basis state is a product of one tensor per spin,
normalised to unit Frobenius norm, and is described by one integer per spin.
`basis()` stores all of them as the rows of `ss.basis.basis`, sorted
lexicographically with the all-zero unit state first:

```python
ab = spinpy.basis(spin_system(["1H", "1H"]), {"formalism": "sphten-liouv", "approximation": "none"})
assert ab.basis.nstates == 16                    # (2*2)^2: the complete Liouville space
assert ab.basis.basis[0].tolist() == [0, 0]      # the unit state
assert ab.basis.basis[6].tolist() == [1, 2]      # T(1,+1) on spin 0 times T(1,0) on spin 1
print(ab.basis.basis.T)
```

Two labels are derived from the descriptor and used by the filters below:
`tot_proj`, the total coherence order (sum of `m` over spins), and
`tot_cord`, the spin correlation order (number of spins carrying a
non-unit tensor):

```python
assert ab.basis.tot_proj[6] == 1 and ab.basis.tot_cord[6] == 2
assert sorted(set(ab.basis.tot_cord)) == [0, 1, 2]
hd = spinpy.basis(spin_system(["1H", "2H"]), {"formalism": "sphten-liouv", "approximation": "none"})
assert hd.basis.nstates == 4 * 9                 # spin-1 has nine tensors, ranks 0 to 2
```

## State-space restriction

`bas["approximation"]` chooses which states are kept. The complete basis
(`'none'`) is practical up to about ten spins; beyond that, Spinach's
"IK" basis sets keep only correlations between spins that are coupled to
each other. All of them are unions of complete bases of small subgraphs of
the coupling graph, so every state in them is a product of tensors on a
few coupled spins and the unit operator elsewhere.

| approximation | keeps | parameters |
|---|---|---|
| `none` | everything | |
| `IK-0` | all correlations of up to `inter_level` spins, regardless of coupling | `inter_level` |
| `IK-1` | correlations of up to `inter_level` spins connected through the coupling graph, and of up to `prox_level` spatially proximate spins | `inter_level`, `prox_level`, `connectivity` |
| `IK-2` | each spin together with all spins directly coupled to it, plus proximity correlations up to `prox_level` | `prox_level`, `connectivity` |

`connectivity` is `'scalar_couplings'` (the isotropic coupling builds the
graph) or `'full_tensors'` (the full tensor norm does); a coupling counts
when its norm exceeds `tols.inter_cutoff`. Spatial proximity needs atomic
coordinates, which are not ported, so for now no spins are proximate and
`prox_level` only has to be present.

For `n` spin-1/2 particles, `IK-0` at level `k` keeps
`sum_j C(n, j) 3^j` for `j <= k` states; with `IK-1` and `IK-2` the count
depends on the coupling topology. A chain with nearest-neighbour couplings
shows the difference:

```python
from math import comb

five = spin_system(["1H"] * 5)
for level in (1, 2, 3, 5):
    ik0 = spinpy.basis(five, {"formalism": "sphten-liouv", "approximation": "IK-0", "inter_level": level})
    assert ik0.basis.nstates == sum(comb(5, j) * 3**j for j in range(level + 1))
    print(f"IK-0 level {level}: {ik0.basis.nstates} states")

J = np.zeros((8, 8))
for n in range(7):
    J[n, n + 1] = 10.0                       # nearest-neighbour couplings only
chain = spin_system(["1H"] * 8, coupling={"scalar": J.tolist()})
ik2 = spinpy.basis(chain, {"formalism": "sphten-liouv", "approximation": "IK-2",
                           "connectivity": "scalar_couplings", "prox_level": 1})
ik1 = spinpy.basis(chain, {"formalism": "sphten-liouv", "approximation": "IK-1",
                           "connectivity": "scalar_couplings", "inter_level": 4, "prox_level": 1})
print(f"8-spin chain: complete {4**8}, IK-2 {ik2.basis.nstates}, IK-1(4) {ik1.basis.nstates}")
assert ik2.basis.tot_cord.max() == 3         # a chain spin has at most two neighbours
assert ik1.basis.tot_cord.max() == 4
```

The states kept by `IK-2` on the chain are products on up to three
consecutive spins. The correlation-order histogram makes the restriction
visible:

```python
import plotly.graph_objects as go

orders, counts = np.unique(ik2.basis.tot_cord, return_counts=True)
fig = go.Figure(go.Bar(x=orders, y=counts))
fig.update_layout(title="IK-2 basis of an 8-spin chain", xaxis_title="spin correlation order",
                  yaxis_title="number of states", width=500, height=350)
fig
```

## Filters

Three filters remove states that a given experiment never populates; each
is a list with one entry per chemical subsystem (see below), so for a
single molecule they are lists of one element.

- `projections`: coherence orders to keep. The unit state is always kept.
  A pulse-acquire experiment on an ideal 90-degree pulse only needs
  `[+1]` (or `[-1]`, depending on the detection convention).
- `longitudinal`: spins (0-based numbers or an isotope name) that only
  carry `m = 0` tensors, for example undetected heteronuclei in a
  proton experiment.
- `zero_quantum`: spins whose total coherence order must be zero.

```python
hc = spin_system(["1H", "13C"], zeeman={"scalar": [2.0, 30.0]}, coupling={"scalar": [[0, 145.0], [0, 0]]})
full = spinpy.basis(hc, {"formalism": "sphten-liouv", "approximation": "none"})
plus_one = spinpy.basis(hc, {"formalism": "sphten-liouv", "approximation": "none", "projections": [[1]]})
carbon_z = spinpy.basis(hc, {"formalism": "sphten-liouv", "approximation": "none", "longitudinal": [["13C"]]})
carbon_zq = spinpy.basis(hc, {"formalism": "sphten-liouv", "approximation": "none", "zero_quantum": [[1]]})
assert full.basis.nstates == 16
assert plus_one.basis.nstates == 5 and set(plus_one.basis.tot_proj[1:]) == {1}
assert carbon_z.basis.nstates == 8 and set(carbon_z.basis.basis[:, 1]) == {0, 2}
assert carbon_zq.basis.nstates == 8
```

`bas["manual"]` adds subgraphs of your own: a boolean matrix with one row
per subgraph and one column per spin, whose complete product bases are
added to whatever the approximation generates.

```python
manual = spinpy.basis(chain, {"formalism": "sphten-liouv", "approximation": "IK-0", "inter_level": 1,
                              "manual": [[True, False, False, False, False, False, False, True]]})
assert manual.basis.nstates == 1 + 8 * 3 + 9     # singles, plus the two-spin states on spins 0 and 7
```

## Chemical subsystems

A spin system with several molecules (`inter["chem"]["parts"]`, see the
[Hilbert-space guide](hilbert-space.md#1-create-describe-the-spin-system))
gets one sub-basis per molecule; the sub-bases share the unit state and
nothing else, because operators and states never correlate spins of
different molecules. This is the layout Spinach's kinetics superoperator
relies on: with identical molecules on both sides of a reaction, the
sub-bases are identical blocks and exchange maps each state of one onto
the same state of the other.

```python
exchanging = spin_system(["1H", "1H"], zeeman={"scalar": [0.0, 3.0]},
                         chem={"parts": [[0], [1]], "rates": [[-2e3, 2e3], [2e3, -2e3]], "concs": [1.0, 1.0]})
two_sites = spinpy.basis(exchanging, {"formalism": "sphten-liouv", "approximation": "none"})
assert two_sites.basis.nstates == 7              # 1 + 3 + 3, not 16
print(two_sites.basis.basis.T)
```

Filters are specified per molecule: `projections=[[1], [1]]` keeps the
`+1` coherences of both sites.

## Operators: superoperators in the restricted basis

In Liouville space an operator `Q` acts on the density matrix from the
left (`Q rho`), from the right (`rho Q`), as a commutator (`[Q, rho]`, what
Hamiltonians do) or as an anticommutator. `operator(ss, ops, spins,
operator_type)` returns the chosen superoperator as a sparse
`nstates x nstates` matrix; `operator_type` is `'left'`, `'right'`,
`'comm'` (the default, as in Spinach) or `'acomm'`. The operator
descriptions are the same as in Hilbert space:

```python
ab_hilb = spinpy.basis(spin_system(["1H", "1H"]), {"formalism": "zeeman-hilb", "approximation": "none"})
Lz_comm = spinpy.operator(ab, "Lz", "1H")                   # [Lz, .] on both protons
Lz_left = spinpy.operator(ab, "Lz", "1H", "left")
Lz_right = spinpy.operator(ab, "Lz", "1H", "right")
assert Lz_comm.shape == (16, 16)
assert abs(Lz_comm - (Lz_left - Lz_right)).max() == 0
```

Nothing is ever expanded into the full `dim^2 x dim^2` space: the product
of two spherical tensors on one spin is a fixed linear combination of
spherical tensors on that spin (`spinpy.operators.ist_product_table`), so
the action on a product state factorises over the spins and the matrix
elements are looked up directly in the basis descriptor. In a restricted
basis the result is the exact superoperator projected onto the kept
states; states that an operator would create outside the basis are simply
dropped. The matrix elements themselves can be checked against the
Hilbert-space definition, `A[k, j] = <B_k | Q B_j>` for the orthonormal
basis operators `B_k`:

```python
from spinpy.operators import irr_sph_ten, lin2lm

def basis_operator(ss, row):
    t = np.array([[1.0 + 0j]])
    for mult, index in zip(ss.mults, row):
        rank, proj = lin2lm(int(index))
        t = np.kron(t, irr_sph_ten(int(mult), rank)[rank - proj].toarray())
    return t / np.sqrt(np.vdot(t, t).real)

B = [basis_operator(ab, row) for row in ab.basis.basis]
Q = spinpy.operator(ab_hilb, "Lz", "1H").toarray()
reference = np.array([[np.vdot(bk, Q @ bj) for bj in B] for bk in B])
assert abs(Lz_left.toarray() - reference).max() < 1e-15
```

## States

`state(ss, ops, spins, method)` returns a dense complex vector of
coefficients in the orthonormal basis; its first element is the
coefficient of the unit state. Spinach's three methods are available:

- `'exact'` (default): the operator applied to the unit state, which is
  the expansion of `Op / sqrt(dim)`. A spin-1/2 `Lz` has coefficient 1/2 on
  its `T(1,0)` state, `L+` has `-1/sqrt(2)` on `T(1,+1)`.
- `'cheap'`: the coefficients placed directly on the basis states (`Lz`
  gives 1, `L+` gives `-sqrt(2)`); an error if the state is not in the
  basis. For products of `p` spin-1/2 operators the two differ by `2^p`.
- `'chem'`: `'exact'` weighted by the concentration of the molecule each
  term belongs to, for exchanging systems.

```python
rho_exact = spinpy.state(ab, "Lz", "1H")
rho_cheap = spinpy.state(ab, "Lz", "1H", "cheap")
assert rho_exact.shape == (16,)
assert np.count_nonzero(rho_exact) == 2 and set(rho_exact[rho_exact != 0]) == {0.5}
assert set(rho_cheap[rho_cheap != 0]) == {1.0}
coil = spinpy.state(ab, "L+", "1H")
assert np.isclose(np.vdot(coil, rho_exact), 0)      # Lz and L+ are orthogonal
weighted = spinpy.state(two_sites, "L+", "1H", "chem")
assert np.count_nonzero(weighted) == 2             # one T(1,+1) state per site
```

Expectation values are Frobenius products, `spinpy.hdot(coil, rho)`, so
the observable of a `'exact'` state against a `'exact'` coil carries a
factor `1/dim` relative to the Hilbert-space trace; Spinach has the same
convention, and only ratios of observables matter in practice.

## The Liouvillian: Hamiltonian and kinetics superoperators

`hamiltonian(ss, operator_type)` uses the same term table as in Hilbert
space (`assume` first) and returns, by default, the commutation
superoperator `[H, .]`, the generator of coherent evolution. The
rotating-frame offsets of `frqoffset` are added as `2 pi offset [Lz, .]`:

```python
hc_nmr = spinpy.assume(spinpy.basis(hc, {"formalism": "sphten-liouv", "approximation": "none"}), "nmr")
H = spinpy.hamiltonian(hc_nmr)
assert H.shape == (16, 16)
H_off = spinpy.frqoffset(hc_nmr, H, {"spins": ["1H"], "offset": [1200.0]})
Lz_comm_H = spinpy.operator(hc_nmr, "Lz", "1H")
assert abs(H_off - H - 2 * np.pi * 1200.0 * Lz_comm_H).max() < 1e-9
```

A commutation superoperator is traceless and its eigenvalues are
differences of Hamiltonian eigenvalues, so the coherence frequencies of
the spectrum can be read off directly; as a check against the Hilbert-space
Hamiltonian:

```python
hc_hilb = spinpy.assume(spinpy.basis(hc, {"formalism": "zeeman-hilb", "approximation": "none"}), "nmr")
E = np.linalg.eigvalsh(np.asarray(spinpy.hamiltonian(hc_hilb)))
differences = np.sort((E[:, None] - E[None, :]).ravel())
assert np.allclose(np.sort(np.linalg.eigvals(np.asarray(H)).real), differences, atol=1e-6)
```

In a restricted basis the Liouvillian is the projection of the complete
one onto the kept states; Karabanov et al. (2011) show that the error this
makes decays as a Gaussian in the correlation order for liquid-state NMR.

`kinetics(ss)` builds the first-order exchange superoperator from
`inter["chem"]["rates"]`: every state of one molecule is mapped onto the
identical state of the other at the given rate, the diagonal rates drain
the source states, and the unit state is untouched. It enters the
Liouvillian as `L = H + 1j * R + 1j * K`, Spinach's convention for
dissipative superoperators:

```python
exch_nmr = spinpy.assume(two_sites, "nmr")
K = spinpy.kinetics(exch_nmr)
assert K.shape == (7, 7)
assert np.allclose(np.asarray(K.sum(axis=0)), 0)          # conservation of matter
assert K[0, 0] == 0                                        # the unit state is untouched
L = spinpy.hamiltonian(exch_nmr) + 1j * K
```

## Analytical filters

In this basis the correlation order and the coherence order of every
state are labels of the descriptor, so selecting coherences (what phase
cycles do in the spectrometer) is a matter of zeroing coefficients.
`correlation(ss, rho, orders, spins)` keeps the states whose number of
correlated spins (among `spins`) is in `orders`; `coherence(ss, rho,
spec)` keeps the states whose total coherence order over each spin set in
`spec` is in the listed orders. Both accept arrays with the spin state
space on the last axis, so they work on stacks of state vectors too.

```python
rho = spinpy.state(hc_nmr, "L+", "all") + spinpy.state(hc_nmr, ["Lz", "L+"], [0, 1])
single = spinpy.correlation(hc_nmr, rho, [1])
assert np.count_nonzero(single) == 2                      # L+ on 1H and on 13C
proton = spinpy.coherence(hc_nmr, rho, [("1H", [1]), ("13C", [0])])
assert np.count_nonzero(proton) == 1                      # only L+ on the proton
stack = np.stack([rho, 2 * rho])
assert spinpy.correlation(hc_nmr, stack, [2]).shape == (2, 16)
```

## Propagation: `step`, `evolution`, and the reductions

A Liouville-space state is a vector, so one time step is a
matrix-vector product with `expm(-1j * L * dt)`. `step(ss, L, rho, dt)`
computes it without forming the exponential: a Taylor series of the
generator acting on the vector, in substeps short enough for the series
to converge quickly. It takes a single vector or a stack of column
vectors, and a list of two or three generators sampled across the step
(the Lie-group quadratures Spinach uses for shaped pulses):

```python
import scipy.linalg as sla

L = spinpy.frqoffset(hc_nmr, spinpy.hamiltonian(hc_nmr), {"spins": ["1H"], "offset": [1200.0]})
rho = spinpy.state(hc_nmr, "L+", "1H")
dt = 1e-3
rho_dt = spinpy.step(hc_nmr, L, rho, dt)
assert np.allclose(rho_dt, sla.expm(-1j * np.asarray(L) * dt) @ rho, rtol=1e-12)
```

`evolution(ss, L, coil, rho, dt, nsteps, output)` runs a whole free
evolution period. Before propagating, it reduces the problem exactly for
the trajectory at hand (`spinpy.reduce`):

- **destination state screening** (Krzystyniak et al. 2011): for the
  `'observable'` output the reduction is run backwards from the coil
  state, since only states that can reach the detector matter;
- **zero track elimination** (Kuprov 2008, `spinpy.zte`): a short trial
  propagation marks the states that are ever populated above
  `tols.zte_tol`; the others are dropped;
- **path tracing** (Hogben et al. 2010, `spinpy.path_trace`): the
  generator is split into non-interacting subspaces (connected components
  of its graph), unpopulated ones are dropped, small ones merged into
  batches of `tols.merge_dim` states. Skipped below `tols.merge_dim`
  states.

Each subspace is then propagated with the exponential propagator, or, above
`tols.krylov_tol` states, by repeated `step` calls (`spinpy.krylov`).
`sys["disable"]` accepts Spinach's `'zte'`, `'pt'`, `'dss'`, `'merge'`,
`'krylov'` and `'trajlevel'` tokens, `sys["enable"]` accepts `'krylov'`;
none of them changes the answer:

```python
coil = spinpy.state(hc_nmr, "L+", "1H")
fid = spinpy.evolution(hc_nmr, L, coil, rho, 1 / 500, 63, "observable")
hc_plain = spinpy.create({"isotopes": ["1H", "13C"], "magnet": 14.1, "disable": ["trajlevel"]},
                         {"zeeman": {"scalar": [2.0, 30.0]}, "coupling": {"scalar": [[0, 145.0], [0, 0]]}})
hc_plain = spinpy.assume(spinpy.basis(hc_plain, {"formalism": "sphten-liouv", "approximation": "none"}), "nmr")
fid_plain = spinpy.evolution(hc_plain, L, coil, rho, 1 / 500, 63, "observable")
assert np.allclose(fid, fid_plain, rtol=1e-9)
projectors = spinpy.reduce(hc_nmr, L.conj().T, coil)
assert sum(p.shape[1] for p in projectors) == 2     # L+(H) and L+(H)Lz(C) are all that reaches the coil
```

Outputs: `'observable'` returns `coil^H rho(t)` at `nsteps + 1` points
(a `(nsteps + 1, k)` array for a stack of `k` vectors), `'final'` the
state after `nsteps` steps, `'trajectory'` the `(nstates, nsteps + 1)`
array of states. Because `'exact'` states carry a factor `1/sqrt(dim)`
each, a Liouville-space observable is the Hilbert-space one divided by
`dim`:

```python
hc_hilb_nmr = spinpy.assume(spinpy.basis(hc, {"formalism": "zeeman-hilb", "approximation": "none"}), "nmr")
L_hilb = spinpy.frqoffset(hc_hilb_nmr, spinpy.hamiltonian(hc_hilb_nmr), {"spins": ["1H"], "offset": [1200.0]})
rho_hilb = spinpy.state(hc_hilb_nmr, "L+", "1H")
fid_hilb = spinpy.evolution(hc_hilb_nmr, L_hilb, rho_hilb, rho_hilb, 1 / 500, 63, "observable")
assert np.allclose(4 * fid, fid_hilb, rtol=1e-9)
```

## Pulse-acquire and chemical exchange through `liquid`

`liquid` builds `H`, `R` (zero: relaxation is not ported) and `K`
(`spinpy.kinetics`) and hands them to the pulse sequence, which forms the
Liouvillian as `L = H + 1j * R + 1j * K`; `acquire` then calls
`evolution`. For an exchanging system the initial state is weighted by
the concentrations (`state(..., "chem")`) and the coil is not, exactly as
in Spinach's `examples/kinetics/exchange_symmetric.m`:

```python
from spinpy.plotting import plot_1d

exch_pars = {"spins": ["1H"], "offset": 900.0, "sweep": 5000.0, "npoints": 512, "zerofill": 1024,
             "rho0": spinpy.state(two_sites, "L+", "1H", "chem"), "coil": spinpy.state(two_sites, "L+", "1H"),
             "axis_units": "ppm", "invert_axis": True}
fid_exch = spinpy.liquid(two_sites, spinpy.acquire, exch_pars, "nmr")
fid_exch *= np.exp(-6 * np.arange(exch_pars["npoints"]) / exch_pars["npoints"])   # exponential apodisation
spectrum_exch = np.fft.fftshift(np.fft.fft(fid_exch, exch_pars["zerofill"]))
plot_1d(two_sites, spectrum_exch.real, exch_pars)
```

The two lines at 0 and 3 ppm, 1800 Hz apart at 14.1 T, exchange at
2000 Hz: close to coalescence, they merge into one broad line. The same
FID follows the McConnell equations of the two sites, which
`tests/unit/test_propagation_liouv.py` checks against a `2 x 2` matrix
exponential.

## What is not ported yet

- `evolution` outputs `'total'`, `'refocus'` and `'multichannel'`,
  Lie-group quadratures in Hilbert space, relaxation, magnetisation flux
  and radical-pair kinetics, `decouple`, `homodec_*` and `dead_time` in
  `acquire`;
- permutation symmetry (`sym_group`, `sym_spins`);
- the `zeeman-liouv` formalism, and the `IK-DNP` and `IK-SBS` basis sets;
- proximity-based correlations, which need `inter.coordinates`.

## Checking results against Spinach

Everything on this page is checked in two ways. `tests/unit/` compares
the superoperators with direct Frobenius products of the Hilbert-space
operators (`<B_k | Q B_j>`), which needs no reference data and holds to
1e-15 for operators and to the `liouv_zero` rounding grid (1e-10 rad/s)
for Hamiltonians. `tests/slice/test_liouv_setup_parity.py` compares the
basis descriptors, product tables, superoperators, states, Hamiltonians,
kinetics and filters of twelve reference cases, and
`tests/slice/test_liouv_pa_parity.py` the propagators, steps, reductions,
`evolution` outputs and pulse-acquire FIDs of nine (including two-site
exchange, and cases with the Krylov path and path tracing forced), with
the output of Spinach 2.13 stored as fixtures (`oracle/README.md`,
requests FR-002 and FR-003 in `porting/FIXTURE_REQUESTS.md`); those tests
skip while the fixtures are absent. Measured agreement (largest
elementwise difference relative to the largest element):

| stage | difference from Spinach |
|---|---|
| basis descriptors, coherence and correlation orders | 0 |
| product tables: spin-1/2 | 0 |
| product tables: spin-1 | 2e-16 |
| superoperators (all four types, 5 to 11 441 states) | 2e-16, identical sparsity |
| states (`exact`, `cheap`, `chem`) | 0 |
| Hamiltonian superoperators and offsets | 2e-14 |
| kinetics superoperators | 0 |
| `correlation` and `coherence` | 0 |
| Liouvillians `H + iK` with offsets | 2e-14 |
| dwell-time propagators (16 to 1789 states) | 2e-14 |
| `step` over one and over 7.3 dwell times | 1e-16, 7e-16 |
| zero-track and `reduce` projectors | identical sets of kept states |
| `evolution`: `observable`, `final`, `trajectory` | 3e-13, 8e-13, 5e-14 |
| `liquid`/`acquire` FIDs and spectra (256 to 512 points) | 3e-13, 8e-13 |

The Hamiltonian figure is the same `liouv_zero` grid effect as in Hilbert
space: Spinach sums the terms in random order before rounding, so single
elements can land one grid step (1e-10 rad/s) away.
