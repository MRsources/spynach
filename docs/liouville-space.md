# Liouville space: the spherical-tensor formalism

This guide covers the `sphten-liouv` formalism: Liouville space built on
products of irreducible spherical tensors, with the state-space restriction
that lets Spinach simulate large spin systems. Read
[Liquid-state NMR in Hilbert space](hilbert-space.md) first; `create`,
`assume`, the experiment parameters and the processing are the same. The
companion notebook is
[`notebooks/liouville-space.ipynb`](notebooks/liouville-space.ipynb).

Ported so far: the basis set with its restrictions and filters, operators
(superoperators) and states. The Liouvillian and propagation in this
formalism follow in the next steps; until then `hamiltonian` raises
`NotImplementedError` for a `sphten-liouv` system.

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
8. [What is not ported yet](#what-is-not-ported-yet)

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

## What is not ported yet

- `hamiltonian`, `frqoffset` and everything after them for `sphten-liouv`
  systems (next steps of the port), and the `kinetics` superoperator;
- permutation symmetry (`sym_group`, `sym_spins`);
- the `zeeman-liouv` formalism, and the `IK-DNP` and `IK-SBS` basis sets;
- proximity-based correlations, which need `inter.coordinates`.
