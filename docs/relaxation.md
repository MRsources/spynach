# Relaxation

This guide covers spin relaxation: Redfield theory for rotational
diffusion (chemical shift anisotropy, dipolar and quadrupolar mechanisms,
with cross-correlations), the phenomenological T1/T2 and damping models,
the retention policies, and thermal equilibrium with the two
thermalisation methods. Read the [Liouville-space](liouville-space.md) and
[solid-state](solid-state.md) guides first: relaxation lives in Liouville
space and is driven by the same anisotropic interaction tensors. The
companion notebook is
[`notebooks/relaxation.ipynb`](notebooks/relaxation.ipynb).

Every Python block is executed by `tests/docs/`, in order and in one
namespace.

## Contents

1. [Specifying relaxation](#specifying-relaxation)
2. [Redfield theory](#redfield-theory)
3. [Retention policies and dynamic frequency shifts](#retention-policies-and-dynamic-frequency-shifts)
4. [T1/T2 and damping models](#t1t2-and-damping-models)
5. [Thermal equilibrium and thermalisation](#thermal-equilibrium-and-thermalisation)
6. [Transient NOE](#transient-noe)
7. [Relaxation in the contexts](#relaxation-in-the-contexts)
8. [What is not ported yet](#what-is-not-ported-yet)
9. [Checking results against Spinach](#checking-results-against-spinach)

## Specifying relaxation

Relaxation is switched on by `inter["relaxation"]`, a list of theories,
and always needs two policies alongside it: `inter["rlx_keep"]`, which
terms of the superoperator to keep, and `inter["equilibrium"]`, where the
system relaxes to. The fields and their meanings are Spinach's:

| key | meaning |
|---|---|
| `relaxation` | `["redfield"]`, `["t1_t2"]`, `["damp"]`, or combinations |
| `rlx_keep` | `"labframe"` (everything), `"secular"`, `"kite"`, `"diagonal"` |
| `equilibrium` | `"zero"` (relax to nothing), `"IME"`, `"dibari"` |
| `tau_c` | one list of 1, 2 or 3 rotational correlation times (s) per chemical subsystem, for Redfield theory |
| `temperature` | Kelvin, default 298 |
| `rlx_dfs` | `"ignore"` (default) or `"keep"` the dynamic frequency shifts |
| `r1_rates`, `r2_rates` | per spin, Hz, scalar or 3x3 tensor, for `t1_t2` |
| `damp_rate` | Hz, scalar or 3x3 tensor, for `damp` |

The settings are kept in `ss.rlx`. `relaxation(ss)` then returns the
superoperator `R` in Hz; it enters the Liouvillian as `L = H + 1j * R +
1j * K`, the convention of every Spinach pulse sequence.

```python
import numpy as np
import scipy.linalg as sla
import respinach


def dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


hc = respinach.create({"isotopes": ["1H", "13C"], "magnet": 14.1},
                   {"zeeman": {"scalar": [0.0, 0.0]}, "coordinates": [[0, 0, 0], [0, 0, 1.03]],
                    "relaxation": ["redfield"], "equilibrium": "dibari", "rlx_keep": "kite",
                    "temperature": 298, "tau_c": [[100e-12]]})
hc = respinach.basis(hc, {"formalism": "sphten-liouv", "approximation": "none"})
R = respinach.relaxation(hc)
assert R.shape == (16, 16)
```

## Redfield theory

Bloch-Redfield-Wangsness theory treats the anisotropic part of the
Hamiltonian, modulated by rotational diffusion, as a perturbation of the
lab-frame isotropic Hamiltonian. respinach uses Spinach's
diagonalisation-free form (Kuprov 2011): every pair of irreducible
components `Q[r][k][m]`, `Q[r][p][q]` of the anisotropic Hamiltonian (the
same `Q` as in the solid-state guide, here as commutation superoperators
in the lab frame) contributes

    R -= w * Q[r][k][m] * int_0^T exp(i L0 t) Q[r][p][q]^H exp(-i (L0 - i/tau) t) dt

where `L0` is the lab-frame Liouvillian, the weights `w` and decay rates
`1/tau` come from the rotational diffusion model (isotropic, axial or
rhombic, from one, two or three correlation times), and the integral is a
block of one matrix exponential (the auxiliary-matrix formalism of
Goodwin and Kuprov 2015, `respinach.relaxation.expmint`). No eigenvectors are
ever needed, so the method scales to restricted bases.

For a dipolar-coupled pair the result is the Solomon theory: with
`J(w) = tau_c / (1 + w^2 tau_c^2)` and the dipolar constant `b`, the auto-
and cross-relaxation rates are `rho_I = b^2/10 [J(wI-wS) + 3J(wI) +
6J(wI+wS)]` and `sigma = b^2/10 [6J(wI+wS) - J(wI-wS)]`. Reading them off
`R` between normalised `Lz` states:

```python
Lz_H, Lz_C = respinach.state(hc, ["Lz"], [0]), respinach.state(hc, ["Lz"], [1])
rate = lambda bra, ket: -np.vdot(bra, dense(R) @ ket).real / np.sqrt(np.vdot(bra, bra).real * np.vdot(ket, ket).real)
b = hc.gammas[0] * hc.gammas[1] * hc.tols.hbar * hc.tols.mu0 / (4 * np.pi * 1.03e-10**3)
J = lambda w: 100e-12 / (1 + (w * 100e-12) ** 2)
wH, wC = abs(hc.basefrqs[0]), abs(hc.basefrqs[1])
assert np.isclose(rate(Lz_H, Lz_H), b**2 / 10 * (J(wH - wC) + 3 * J(wH) + 6 * J(wH + wC)), rtol=1e-4)
assert np.isclose(rate(Lz_H, Lz_C), b**2 / 10 * (6 * J(wH + wC) - J(wH - wC)), rtol=1e-4)
print(f"R1(1H) = {rate(Lz_H, Lz_H):.3f} Hz, sigma(HC) = {rate(Lz_H, Lz_C):.3f} Hz")
```

The agreement is limited by the integral's truncation
(`tols.rlx_integration`, default 1e-4), as in Spinach. Redfield theory
requires `T1, T2 >> tau_c`; `relaxation` raises an error when a rate
violates it. Two correlation times give axial rotational diffusion
(`tau_c = [[tau_parallel, tau_perpendicular]]`), three give rhombic
diffusion (rank 2 only).

## Retention policies and dynamic frequency shifts

The full superoperator (`rlx_keep = "labframe"`) is only correct for lab
frame simulations. In the rotating frame, `"secular"` keeps the elements
between states of the same Zeeman frequency, `"kite"` keeps the
longitudinal block and the diagonal (what a NOESY needs), and
`"diagonal"` keeps only the self-relaxation rates. `rlx_dfs = "ignore"`
(the default) discards the imaginary part of `R`, the dynamic frequency
shifts:

```python
def pair(keep, **extra):
    ss = respinach.create({"isotopes": ["1H", "13C"], "magnet": 14.1},
                       {"zeeman": {"scalar": [0.0, 0.0]}, "coordinates": [[0, 0, 0], [0, 0, 1.03]],
                        "relaxation": ["redfield"], "equilibrium": "zero", "rlx_keep": keep,
                        "temperature": 298, "tau_c": [[100e-12]], **extra})
    return respinach.basis(ss, {"formalism": "sphten-liouv", "approximation": "none"})

R_lab = dense(respinach.relaxation(pair("labframe", rlx_dfs="keep")))
R_diag = dense(respinach.relaxation(pair("diagonal")))
assert np.abs(R_lab.imag).max() > 0                       # dynamic frequency shifts
assert np.allclose(R_diag, np.diag(np.diag(R_diag)))
assert np.allclose(np.diag(R_diag), np.diag(R_lab).real, atol=1e-9)
```

## T1/T2 and damping models

`"t1_t2"` is the extended T1/T2 model: a diagonal superoperator in which
every basis state relaxes with the sum of the `r1_rates` of the spins it
carries longitudinally and the `r2_rates` of the spins it carries
transversely (3x3 tensors give orientation-dependent rates, projected on
the field direction at the orientation handed to `relaxation`).
`"damp"` damps every state except the unit state at `damp_rate`
(in Hilbert space, every element at half the rate):

```python
t1t2 = respinach.create({"isotopes": ["1H", "13C"], "magnet": 14.1},
                     {"relaxation": ["t1_t2", "damp"], "equilibrium": "zero", "rlx_keep": "diagonal",
                      "r1_rates": [1.0, 2.0], "r2_rates": [5.0, 7.0], "damp_rate": 0.5})
t1t2 = respinach.basis(t1t2, {"formalism": "sphten-liouv", "approximation": "none"})
R_t = dense(respinach.relaxation(t1t2))
row = lambda spec: np.flatnonzero((t1t2.basis.basis == spec).all(axis=1))[0]
assert R_t[row([2, 0]), row([2, 0])] == -1.5          # Lz(H): R1 + damping
assert R_t[row([1, 3]), row([1, 3])] == -12.5         # L+(H) L-(C): R2(H) + R2(C) + damping
assert R_t[0, 0] == 0                                  # the unit state never relaxes
```

## Thermal equilibrium and thermalisation

`equilibrium(ss)` returns the thermal state at `inter["temperature"]`
from the lab-frame Hamiltonian; in Liouville space it is `exp(-H beta)`
applied to the unit state (through `step` with an imaginary time), with
unit coefficient on the unit state, in Hilbert space the Boltzmann density
matrix with unit trace. A relaxation superoperator built with `equilibrium
= "zero"` drives everything to zero; the two thermalisation methods make
`rho_eq` its stationary state instead. `"IME"`, the inhomogeneous master
equation, adds a drive from the unit state (`R -> R - (R rho_eq) u^T`);
`"dibari"`, the DiBari-Levitt method, multiplies by the Boltzmann factor
(`R -> R exp(-H beta)`). `relaxation` applies the one chosen in
`inter["equilibrium"]`; `thermalize` applies either to a given `R`:

```python
rho_eq = respinach.equilibrium(hc)
assert rho_eq[0] == 1
zero = pair("kite")
R0 = respinach.relaxation(zero)
HL = respinach.hamiltonian(respinach.assume(zero, "labframe"), "left")
for R_th in (respinach.thermalize(zero, R0, None, None, rho_eq, "IME"), respinach.thermalize(zero, R0, HL, 298.0, None, "dibari")):
    assert np.linalg.norm(dense(R_th) @ rho_eq) < 1e-9       # rho_eq is stationary
```

## Transient NOE

Spinach's `examples/nmr_liquids/noe_two_spin_het.m`: invert the proton of
a 1H-13C pair and follow the longitudinal magnetisation of both spins
under `R` alone, with the `"multichannel"` output of `evolution` (one
trace per coil column):

```python
Lz1, Lz2 = respinach.state(hc, ["Lz"], [0]), respinach.state(hc, ["Lz"], [1])
rho = rho_eq - 2 * Lz1 * np.vdot(Lz1, rho_eq) / np.vdot(Lz1, Lz1).real   # proton inverted
coils = np.stack([Lz1, Lz2], axis=1)
dt, nsteps = 1e-2, 400
noe = respinach.evolution(hc, 1j * R, coils, rho, dt, nsteps, "multichannel").real
assert noe.shape == (2, nsteps + 1)
```

```python
import plotly.graph_objects as go

t = dt * np.arange(nsteps + 1)
scale = np.vdot(Lz2, rho_eq).real
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=noe[0] / scale, name="1H"))
fig.add_trace(go.Scatter(x=t, y=noe[1] / scale, name="13C"))
fig.update_layout(title="Transient heteronuclear NOE", xaxis_title="time, s",
                  yaxis_title="Lz / Lz(13C) at equilibrium", width=600, height=380)
fig
```

The carbon magnetisation rises above its equilibrium value before
returning to it: the heteronuclear Overhauser effect. The curves follow
the two-spin Solomon equations, which `tests/unit/test_relaxation.py`
checks against a matrix exponential.

## Relaxation in the contexts

`liquid`, `crystal` and `powder` build `R` with `relaxation(ss)` (at each
grid orientation in `powder`, so anisotropic rate tensors and the thermal
equilibrium of anisotropic systems follow the orientation) and hand it to
the pulse sequence together with `H` and `K`. A system without
`inter["relaxation"]` gets `R = 0`, as in Spinach.

## What is not ported yet

- the Lindblad, Nottingham and Weizmann DNP theories, scalar relaxation
  of the first and second kind (`SRFK`, `SRSK`), the Nakajima-Zwanzig
  evaluation of the Redfield kernel, and orientation functions for rates;
- relaxation in Hilbert space other than `damp`;
- bosonic mode dissipators.

## Checking results against Spinach

`tests/unit/test_relaxation.py` checks Redfield rates against the Solomon
equations, the T1/T2 and damping models against their definitions, the
equilibrium state against the Boltzmann distribution, both thermalisation
methods, and the transient NOE against the two-spin Solomon solution.
`tests/slice/test_relax_parity.py` compares `R` (isotropic and at an
orientation), equilibrium states, both thermalised superoperators and
NOE trajectories of five systems (heteronuclear and homonuclear NOE
pairs, a three-spin system with CSA, dipolar couplings and secular
retention under IME, axial diffusion with dynamic frequency shifts, and
the T1/T2 model plus anisotropic damping) with Spinach 2.13
fixtures (FR-005 in `porting/FIXTURE_REQUESTS.md`); those tests skip
while the fixtures are absent. Measured agreement over the five cases
(largest elementwise difference relative to the largest element):

| stage | difference from Spinach |
|---|---|
| relaxation superoperators, isotropic and at an orientation | 0 (identical after the `rlx_zero` rounding) |
| equilibrium states | 3e-25 |
| thermalised superoperators (IME, DiBari-Levitt) | 1e-15 |
| NOE trajectories (`multichannel`, up to 1000 steps) | 5e-15 |
