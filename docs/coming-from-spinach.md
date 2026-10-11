# Coming from Spinach

spinpy keeps Spinach's structure: the same functions, called in the same
order, with the same arguments, field names, units, and tolerances. A
Spinach script translates into spinpy almost line by line. This page lists,
topic by topic, what carries over unchanged and what is different, and
what is not ported yet. The [Hilbert-space guide](hilbert-space.md) explains each
function in full.

Every Python block on this page is executed by `tests/docs/`. MATLAB blocks
are Spinach code, shown for comparison and not run. The companion notebook
is [`notebooks/coming-from-spinach.ipynb`](notebooks/coming-from-spinach.ipynb).

## Contents

1. [A Spinach script, translated](#a-spinach-script-translated)
2. [General differences](#general-differences)
3. [Spin system: `sys` and `inter`](#spin-system-sys-and-inter)
4. [Basis: `bas`](#basis-bas)
5. [Operators and states](#operators-and-states)
6. [Assumptions and the Hamiltonian](#assumptions-and-the-hamiltonian)
7. [Experiment parameters](#experiment-parameters)
8. [Calculation modalities](#calculation-modalities)
9. [Processing and plotting](#processing-and-plotting)
10. [Messages and errors](#messages-and-errors)

## A Spinach script, translated

Spinach's `examples/nmr_liquids/pa_styrene.m` simulates the 1H spectrum of
styrene at 7.046 T:

```matlab
sys.isotopes={'1H','1H','1H','1H','1H','1H','1H','1H'};
sys.magnet=7.046;
inter.zeeman.scalar={6.720  6.720  6.587  6.587  6.527  6.063   5.085  4.555};
inter.coupling.scalar={ 0.0  1.9170  7.7884  0.6061  1.2447 -0.5347  0.0390  0.1582
                        0.0  0.0     0.6061  7.7884  1.2447 -0.5347  0.0390  0.1582
                        0.0  0.0     0.0     1.4184  7.4390  0.3693 -0.0416  0.0169
                        0.0  0.0     0.0     0.0     7.4390  0.3693 -0.0416  0.0169
                        0.0  0.0     0.0     0.0     0.0    -0.2279  0.2349  0.2905
                        0.0  0.0     0.0     0.0     0.0     0.0    17.6002 10.8954
                        0.0  0.0     0.0     0.0     0.0     0.0     0.0     1.0382
                        0.0  0.0     0.0     0.0     0.0     0.0     0.0     0.0};
bas.formalism='zeeman-hilb';
bas.approximation='none';
spin_system=create(sys,inter);
spin_system=basis(spin_system,bas);
parameters.spins={'1H'};
parameters.rho0=state(spin_system,'L+','1H');
parameters.coil=state(spin_system,'L+','1H');
parameters.decouple={};
parameters.offset=1700;
parameters.sweep=1000;
parameters.npoints=16384;
parameters.zerofill=65536;
parameters.axis_units='ppm';
parameters.invert_axis=1;
fid=liquid(spin_system,@acquire,parameters,'nmr');
fid=apodisation(spin_system,fid,{{'gauss',6}});
spectrum=fftshift(fft(fid,parameters.zerofill));
kfigure(); plot_1d(spin_system,real(spectrum),parameters);
```

In spinpy:

```python
import numpy as np
import spinpy

sys = {"isotopes": ["1H"] * 8, "magnet": 7.046}
inter = {
    "zeeman": {"scalar": [6.720, 6.720, 6.587, 6.587, 6.527, 6.063, 5.085, 4.555]},
    "coupling": {"scalar": [
        [0.0, 1.9170, 7.7884, 0.6061, 1.2447, -0.5347,  0.0390,  0.1582],
        [0.0, 0.0,    0.6061, 7.7884, 1.2447, -0.5347,  0.0390,  0.1582],
        [0.0, 0.0,    0.0,    1.4184, 7.4390,  0.3693, -0.0416,  0.0169],
        [0.0, 0.0,    0.0,    0.0,    7.4390,  0.3693, -0.0416,  0.0169],
        [0.0, 0.0,    0.0,    0.0,    0.0,    -0.2279,  0.2349,  0.2905],
        [0.0, 0.0,    0.0,    0.0,    0.0,     0.0,    17.6002, 10.8954],
        [0.0, 0.0,    0.0,    0.0,    0.0,     0.0,     0.0,     1.0382],
        [0.0, 0.0,    0.0,    0.0,    0.0,     0.0,     0.0,     0.0],
    ]},
}
bas = {"formalism": "zeeman-hilb", "approximation": "none"}
spin_system = spinpy.create(sys, inter)
spin_system = spinpy.basis(spin_system, bas)

parameters = {
    "spins": ["1H"],
    "rho0": spinpy.state(spin_system, "L+", "1H"),
    "coil": spinpy.state(spin_system, "L+", "1H"),
    "decouple": [],
    "offset": 1700,
    "sweep": 1000,
    "npoints": 2048,     # Spinach example: 16384, see below
    "zerofill": 8192,    # Spinach example: 65536
    "axis_units": "ppm",
    "invert_axis": 1,
}
fid = spinpy.liquid(spin_system, spinpy.acquire, parameters, "nmr")

# apodisation(spin_system,fid,{{'gauss',6}}): halve the first point, Gaussian window
fid[0] = fid[0] / 2
fid = fid * np.exp(-6 * np.linspace(0, 1, fid.size) ** 2)
spectrum = np.fft.fftshift(np.fft.fft(fid, parameters["zerofill"]))

assert spectrum.shape == (8192,)
```

The run uses 2048 points instead of 16384 to keep this page quick: with
eight spins the Hilbert space has dimension 256, and the full-length run
takes about 30 s on four CPU cores. Set `"npoints": 16384` and
`"zerofill": 65536` to reproduce the Spinach example exactly.

`kfigure(); plot_1d(spin_system,real(spectrum),parameters)` becomes a
call to `spinpy.plotting.plot_1d`, which reads the same parameters and
returns an interactive figure (see
[Processing and plotting](#processing-and-plotting)). Hover over a line to
read its shift, and drag across the aromatic region to zoom in on the
multiplets:

```python
from spinpy.plotting import plot_1d

plot_1d(spin_system, spectrum.real, parameters, title="Styrene, 1H, 7.046 T")
```

## General differences

| topic | Spinach | spinpy |
|---|---|---|
| spin numbers | 1-based | **0-based**: Spinach spin 1 is spinpy spin 0 |
| input structures | MATLAB structs: `sys.magnet=14.1;` | dictionaries: `sys = {"magnet": 14.1}` |
| lists | cell arrays `{'1H','13C'}` | lists `["1H", "13C"]` |
| spin system object | struct, copied and returned by each function | frozen dataclass, copied and returned by each function |
| matrix product | `A*B` | `A @ B` (`A * B` is element-wise for NumPy and SciPy arrays) |
| conjugate transpose | `A'` | `A.conj().T` |
| sparse matrices | MATLAB sparse | `scipy.sparse.csr_array` |
| time-domain results | `n x 1` column vectors | 1-D arrays of length `n` |
| function handles | `@acquire` | `spinpy.acquire` |
| units | ppm, Hz, Tesla, Bohr magneton; rad/s internally | the same |
| tolerances | `sys.tols.<name>` | `sys["tols"]["<name>"]`, same names and defaults |

The `*` versus `@` difference is the most common translation mistake,
because both run without error:

```python
Lp = spinpy.operator(spin_system, "L+", "1H")
Lm = spinpy.operator(spin_system, "L-", "1H")
matrix_product = Lp @ Lm                 # MATLAB Lp*Lm
elementwise = Lp * Lm                    # MATLAB Lp.*Lm, a different matrix
assert (matrix_product != elementwise).nnz > 0
```

## Spin system: `sys` and `inter`

`create(sys, inter)` reads the same fields as in Spinach, with the same
meaning and units. Shown for a 1H-13C pair:

```matlab
sys.isotopes={'1H','13C'};
sys.magnet=14.1;
inter.zeeman.scalar={2.0 30.0};
inter.coupling.scalar=cell(2,2);
inter.coupling.scalar{1,2}=145;
spin_system=create(sys,inter);
```

```python
sys = {"isotopes": ["1H", "13C"], "magnet": 14.1}
inter = {
    "zeeman": {"scalar": [2.0, 30.0]},
    "coupling": {"scalar": [[None, 145.0],
                            [None, None]]},
}
hc = spinpy.create(sys, inter)
```

- **Same:** isotope names (from Spinach's `spin.m`, including `'E'` for
  electrons), units (ppm for nuclear shifts, Bohr magneton units for
  g-values, Hz for couplings, Tesla for the field), the rule that a
  coupling entered both above and below the diagonal is summed, dropping
  of interactions below `tols.inter_cutoff`, and the internal storage of
  every interaction as a 3x3 tensor in rad/s.
- **Empty entries:** where Spinach uses empty cells (`cell(2,2)`), use
  `None` or `0`. A nested list, or a NumPy array for scalar couplings,
  are both accepted.
- **Tolerances:** `sys["tols"] = {"prop_chop": 1e-14}` does what
  `sys.tols.prop_chop=1e-14` does. The values are in `spin_system.tols`.
- **Immutability:** Spinach scripts that change a spin system after
  `create` (`spin_system.inter.coupling.matrix{1,2}=...`) must build a new
  system instead. A parameter scan is a loop over `create`:

```python
for J in (140.0, 145.0, 150.0):
    scanned = spinpy.create(sys, {**inter, "coupling": {"scalar": [[0, J], [0, 0]]}})
    assert np.isclose(scanned.coupling_matrix[0][1][0, 0], 2 * np.pi * J)
```

- **Chemical subsystems:** `inter.chem.parts`, `inter.chem.concs`, and
  `inter.chem.rates` carry over with 0-based spin indices in `parts`; the
  validation is Spinach's (every spin in one part, no couplings between
  parts, `rates` square with zero column sums, equal isotope sequences in
  exchange mode). They are kept in `ss.chem` (`parts`, `concs`, `rates`).

```matlab
inter.chem.parts={1,2};
inter.chem.rates=[-2e3 2e3; 2e3 -2e3];
inter.chem.concs=[1.0 1.0];
```

```python
inter_exch = {"zeeman": {"scalar": [0.0, 3.0]},
              "chem": {"parts": [[0], [1]], "rates": [[-2e3, 2e3], [2e3, -2e3]], "concs": [1.0, 1.0]}}
exch = spinpy.create({"isotopes": ["1H", "1H"], "magnet": 14.1}, inter_exch)
assert exch.chem.parts == ((0,), (1,))
```

**Not ported yet** (each raises `NotImplementedError` naming the field):
`sys.output`, `sys.enable` (GPU and other options), `sys.disable`,
`sys.parallel`, `sys.scratch`, `inter.coordinates` and `inter.pbc`
(dipolar couplings), `inter.zeeman.eigs`/`euler`,
`inter.coupling.eigs`/`euler`, `inter.giant`, `inter.suscept`,
`inter.order_matrix`, `inter.relaxation` and all relaxation fields,
`inter.chem.flux_rate`/`flux_type` (magnetisation flux), `inter.chem.rp_*`
(radical pair kinetics), `inter.temperature`, and bosonic modes
(cavities, phonons, transmons). The `kinetics` superoperator itself is
not ported yet either; see [Calculation modalities](#calculation-modalities).

## Basis: `bas`

```python
bas = {"formalism": "zeeman-hilb", "approximation": "none"}
hc = spinpy.basis(hc, bas)
```

- **Same:** field names and meanings: `formalism`, `approximation`,
  `inter_level`, `prox_level`, `connectivity`, `projections`,
  `longitudinal`, `zero_quantum`, `manual`. `zeeman-hilb` requires
  `approximation='none'`; basis states are Zeeman product states ordered as
  `kron(spin 1, spin 2, ...)`, so matrices are element for element
  identical to Spinach's. In `sphten-liouv` the basis descriptor
  `ss.basis.basis` is Spinach's `spin_system.bas.basis` (same integer
  encoding `l*l + l - m`, same lexicographic row order, unit state first),
  and `ss.basis.tot_proj`/`tot_cord` are `bas.tot_proj`/`bas.tot_cord`.
  Spinach 2.13 renamed `bas.level` to `inter_level`, `bas.space_level` to
  `prox_level` and `bas.longitudinals` to `longitudinal`; spinpy uses the
  new names and raises the same rename errors for the old ones.
- **Different:** spin numbers in `longitudinal`, `zero_quantum` and
  `manual` are 0-based; the per-substance cell arrays become lists with one
  entry per chemical subsystem (`projections=[[1]]` for one substance,
  `longitudinal=[["13C"], []]` for two). Without `inter.coordinates` the
  proximity graph is empty, so `prox_level` has no effect yet.

```matlab
bas.formalism='sphten-liouv';
bas.approximation='IK-2';
bas.connectivity='scalar_couplings';
bas.prox_level=1;
bas.projections={+1};
```

```python
bas_ik2 = {"formalism": "sphten-liouv", "approximation": "IK-2",
           "connectivity": "scalar_couplings", "prox_level": 1, "projections": [[1]]}
hc_liouv = spinpy.basis(hc, bas_ik2)
assert hc_liouv.basis.basis[0].tolist() == [0, 0]     # unit state first
assert set(hc_liouv.basis.tot_proj[1:]) == {1}        # coherence order +1 only
```

- **Not ported yet:** `zeeman-liouv`, the `IK-DNP` and `IK-SBS`
  approximations, permutation symmetry (`sym_group`, `sym_spins`), and
  proximity-based correlations (they need `inter.coordinates`). In
  Spinach, switching a script from Hilbert to Liouville space changes only
  `bas`, and states become vectors instead of matrices; spinpy keeps that
  behaviour.

## Operators and states

The three Spinach call forms carry over; only the brackets and the spin
numbers change:

| Spinach | spinpy | result |
|---|---|---|
| `operator(spin_system,'Lz','1H')` | `spinpy.operator(ss, "Lz", "1H")` | sum of Lz over all 1H |
| `operator(spin_system,'Lz',[1 3])` | `spinpy.operator(ss, "Lz", [0, 2])` | Lz(1) + Lz(3) in Spinach numbering |
| `operator(spin_system,{'Lz','L+'},{1,2})` | `spinpy.operator(ss, ["Lz", "L+"], [0, 1])` | product Lz(1) L+(2) |
| `operator(spin_system,'Lx','all')` | `spinpy.operator(ss, "Lx", "all")` | also `'nuclei'`, `'electrons'` |
| `state(spin_system,'L+','13C')` | `spinpy.state(ss, "L+", "13C")` | density matrix |

In Spinach a cell array of spin numbers, `{1,2}`, selects the product form.
In spinpy the type of the *first* argument does: a string gives a sum, a
list gives a product.

```python
Lz_sum = spinpy.operator(hc, "Lz", [0, 1])         # Lz(H) + Lz(C)
LzSz = spinpy.operator(hc, ["Lz", "Lz"], [0, 1])   # Lz(H) Lz(C)
assert np.allclose((LzSz @ LzSz).toarray(), np.eye(4) / 16)
```

- **Same:** operator labels `E`, `Lx`, `Ly`, `Lz`, `L+`, `L-`, `Tl,m`;
  matrix elements (Spinach's `Lz` for spin-1/2 is `0.4999999999999999`,
  one ulp below 0.5, and spinpy reproduces it); in Hilbert space a state
  is the operator itself, unnormalised.
- **Same in Liouville space:** `operator(ss, ops, spins, operator_type)`
  takes Spinach's `'left'`, `'right'`, `'comm'` (default) and `'acomm'`
  and builds the superoperator in the restricted basis by the same
  structure-coefficient algorithm (`superop.m`), so matrix elements and
  the basis order match Spinach's. `state(ss, ops, spins, method)` has the
  `'exact'` (default), `'cheap'` and `'chem'` methods with the same
  normalisation: in `sphten-liouv` the `'exact'` coefficient of a spin-1/2
  `Lz` is 1/2, of `L+` it is `-1/sqrt(2)`.
- **Different:** `operator` has no `format` argument (`'xyz'` triplets are
  internal); `operator_type` and `method` are ignored in Hilbert space, as
  in Spinach. Liouville-space states are dense 1-D NumPy vectors rather
  than sparse `n x 1` columns.

```matlab
bas.formalism='sphten-liouv'; bas.approximation='none';
spin_system=basis(spin_system,bas);
Lz_L=operator(spin_system,'Lz','1H','left');
rho=state(spin_system,'L+','1H','cheap');
```

```python
hc_none = spinpy.basis(hc, {"formalism": "sphten-liouv", "approximation": "none"})
Lz_L = spinpy.operator(hc_none, "Lz", "1H", "left")
rho_cheap = spinpy.state(hc_none, "L+", "1H", "cheap")
assert Lz_L.shape == (16, 16) and rho_cheap.shape == (16,)
assert np.isclose(rho_cheap[np.flatnonzero(rho_cheap)[0]], -np.sqrt(2))
```

- **Not ported yet:** labels `CT*` (central transition) and `ZL*`
  (Zeeman energy level projectors), operators on bosonic modes, and the
  `zeeman-liouv` formalism (`hilb2liouv`).

## Assumptions and the Hamiltonian

```python
H = spinpy.hamiltonian(spinpy.assume(hc, "nmr"))
H_zeeman = spinpy.hamiltonian(spinpy.assume(hc, "nmr", "zeeman"))
assert np.allclose(H - H_zeeman, 2 * np.pi * 145.0 * LzSz.toarray())  # weak 1H-13C coupling
```

- **Same:** `assume(spin_system, 'nmr')` and the retention options
  `'zeeman'` and `'couplings'`; the `'nmr'` truncation rules (strong
  coupling within an isotope, weak `LzSz` coupling between isotopes);
  rounding of every element to `tols.liouv_zero`; the dense-or-sparse
  choice by `tols.small_matrix` and `tols.dense_matrix`.
- **Same in Liouville space:** `hamiltonian(ss, operator_type)` with
  `'left'`, `'right'`, `'comm'` (default) or `'acomm'` returns the
  superoperator built from the same term table; `frqoffset` adds
  `2*pi*offset*[Lz, .]`; `kinetics(ss)` is the exchange superoperator of
  `kinetics.m` with the `L = H + 1i*R + 1i*K` convention; `correlation`
  and `coherence` zero the same states.
- **Different:** Spinach's `[H,Q]=hamiltonian(...)` also returns the
  anisotropic components `Q`; spinpy returns `H` alone. Spinach sums the
  Hamiltonian terms in a random order, so its last digits change from run
  to run; spinpy sums in a fixed order and is reproducible. `coherence`
  takes a list of `(spins, orders)` pairs instead of nested cell arrays,
  and both filters expect the spin state space on the *last* axis of a
  stacked state array.

```python
hc_liouv = spinpy.assume(hc_none, "nmr")
H_comm = spinpy.hamiltonian(hc_liouv)                  # [H, .] in Liouville space
H_left = spinpy.hamiltonian(hc_liouv, "left")
H_right = spinpy.hamiltonian(hc_liouv, "right")
assert abs(H_comm - (H_left - H_right)).max() < 1e-9   # each is rounded to liouv_zero separately
```

- **Not ported yet:** the other assumption sets (`'esr'`, `'deer'`,
  `'labframe'`, `'qnmr'`, the DNP sets, ...), anisotropic interactions,
  magnetisation flux and radical-pair kinetics, and the Hilbert-space
  branches of `correlation` and `coherence`.

## Experiment parameters

`parameters` is a dictionary with Spinach's field names. `liquid` copies it
before filling in defaults, so the caller's dictionary is never changed,
as with MATLAB's value semantics.

| field | meaning | ported |
|---|---|---|
| `spins` | isotopes of the working channels, e.g. `["1H"]` | yes |
| `offset` | transmitter offset per channel, Hz; a number or a list | yes; default 0 |
| `sweep` | spectral width, Hz | yes |
| `npoints` | number of complex points | yes |
| `rho0`, `coil` | initial state, detection state | yes |
| `decouple` | isotopes to decouple | only empty |
| `rframes`, `needs` | extra rotating frames, context requests | only empty |
| `homodec_oper`, `homodec_pwr`, `dead_time` | `acquire` options | no |
| `zerofill`, `axis_units`, `invert_axis` | read by the plotting functions | yes, by `spinpy.plotting` |

Fields that only processing and plotting functions read are accepted and
ignored by the simulation, as in Spinach.

## Calculation modalities

| Spinach | spinpy |
|---|---|
| contexts: `liquid`, `powder`, `crystal`, `singlerot`, `doublerot`, `floquet`, `gridfree`, `imaging`, ... | `liquid` only |
| pulse sequences in `experiments/`: `acquire`, `hsqc`, `cosy`, `noesy`, ... | `acquire`; write others yourself |
| `evolution` outputs `'observable'`, `'final'`, `'total'`, `'trajectory'`, `'refocus'`, `'multichannel'` | `'observable'` in Hilbert space; `'observable'`, `'final'`, `'trajectory'` in Liouville space |
| `evolution(...,destination)` destination state screening | the same, Liouville space (`destination=` keyword) |
| `reduce`, `zte`, `path_trace` (trajectory-level reduction), `sys.disable={'zte','pt','dss','merge','krylov','trajlevel'}`, `sys.enable={'krylov'}` | the same, Liouville space; permutation symmetry not ported |
| `step` (single generator and 2/3-point Lie quadratures), `krylov` | the same in Liouville space; Hilbert-space `step` for one generator |
| `pulse`, `shaped_pulse_*` | not ported; use `propagator` or `step` |
| `propagator`: Pade `expm` below `small_matrix`, scaled and squared Taylor series above | the same |
| relaxation (`relaxation`) | not ported; `R` is zero |
| chemical kinetics (`kinetics`) | first-order exchange; `liquid` passes it as `K` and `acquire` propagates under `H + 1i*K` |
| parallel `evolution` (Kuprov-Edwards split), `parfor` | serial; the same numbers |
| GPU via `sys.enable={'gpu'}` | CPU only; CuPy backend planned |
| Liouville-space states | dense 1-D NumPy vectors; stacks are `(nstates, k)` arrays, trajectories `(nstates, nsteps + 1)` as in Spinach |

**Pulse sequences** have Spinach's signature, `sequence(spin_system,
parameters, H, R, K)`, and are passed to `liquid` the same way. Inside,
use `spinpy.operator`, `spinpy.propagator`, and `spinpy.evolution` as you
would in MATLAB; the Hilbert-space guide has a
[worked example](hilbert-space.md#writing-your-own-pulse-sequence).
Spinach sequences usually combine the generator as `L = H + 1i*R + 1i*K`;
in Python that is `L = H + 1j * R + 1j * K`.

**Numerical agreement** with Spinach is checked stage by stage against
stored Spinach 2.13 output; see
[Checking results against Spinach](hilbert-space.md#checking-results-against-spinach).

## Processing and plotting

`axis_1d`, `ft_axis`, and `plot_1d` are in `spinpy.plotting`, with
Spinach's arguments; the other processing helpers are a line of NumPy.

| Spinach | spinpy / NumPy |
|---|---|
| `fftshift(fft(fid,zerofill))` | `np.fft.fftshift(np.fft.fft(fid, zerofill))` |
| `apodisation(spin_system,fid,{{'exp',k}})` | `fid[0] /= 2; fid *= np.exp(-k * np.linspace(0, 1, fid.size))` |
| `apodisation(spin_system,fid,{{'gauss',k}})` | `fid[0] /= 2; fid *= np.exp(-k * np.linspace(0, 1, fid.size) ** 2)` |
| `[ax,ax_label]=axis_1d(spin_system,parameters)` | `ax, ax_label = axis_1d(spin_system, parameters)` |
| `ft_axis(offset,sweep,npoints)` | `ft_axis(offset, sweep, npoints)` |
| `kfigure(); plot_1d(spin_system,spectrum,parameters)` | `plot_1d(spin_system, spectrum, parameters)` |
| `plot(fid)` of a time-domain signal | `plot_fid(fid, parameters)` |

How `plot_1d` differs from Spinach's:

- It returns a Plotly figure instead of drawing into the current MATLAB
  axes. Display it as the last expression of a notebook cell, change it
  with Plotly's methods (`fig.update_xaxes(range=[7, 6])` for `xlim`), and
  save it with `fig.write_html(...)`. There is no `kfigure`, `hold on`, or
  `varargin` of line options; to overlay spectra, pass them together as a
  list, with `names=[...]` for the legend.
- `zerofill` defaults to the length of the spectrum (Spinach: to
  `npoints`), and a `zerofill` that disagrees with the spectrum length is
  an error.
- Spinach's defaults (`axis_units='ppm'`, `invert_axis=1`, `offset=0`)
  apply without the log messages; `derivative` is not ported.
- `axis_1d` labels axes in plain text (`1H chemical shift / ppm`) rather
  than LaTeX; `plot_1d` typesets the mass number as a superscript.

Spinach's `apodisation` halves the first point of the FID by default;
apply that step yourself when comparing spectra with Spinach's.

## Messages and errors

- Spinach prints a running log (`report`) and summary tables. spinpy is
  silent; warnings, such as a time step much longer than the system's
  dynamics, go to Python's `logging` module under the name `"spinpy"`.
- Spinach's input checks (`grumble`) become `ValueError` for invalid input.
- Anything that Spinach supports and spinpy does not yet raises
  `NotImplementedError` with a message naming the missing piece. spinpy
  never silently ignores an input it does not understand.

```python
try:
    spinpy.create({"isotopes": ["1H"], "magnet": 14.1, "output": "hush"}, {})
except NotImplementedError as error:
    print(error)
```
