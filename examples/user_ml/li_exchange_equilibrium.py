"""Equilibration of a two-site 7Li exchange system: translation of user_ml/Li_exchange_equilibrium.m.

    OPENBLAS_NUM_THREADS=4 PYTHONPATH=src python examples/user_ml/li_exchange_equilibrium.py \
        [--nsteps-ex N] [--nsteps-eq N] [--out DIR] [--no-plot]

The contributed script (Florin Teleanu, 2026): two 7Li sites A and B at 1.93 T with quadrupolar
couplings of 30 and 90 kHz, axial shift tensors (45 and 135 ppm anisotropy), two-site exchange
A <-> B with pA = 0.95 and kex = 1 kHz, Redfield relaxation (1 ns on either site, secular
retention). The thermal state of the lab-frame Hamiltonian is equilibrated under H + 1i K for
10 s (1e6 steps of 10 us, 'final'), the relaxation superoperator is thermalised to that state
with the inhomogeneous master equation, and a 2 s trajectory (2e5 steps) under H + 1i R + 1i K
follows from the (null, zero-angle) perturbation step(Lx, 0). The trajectory is analysed
through the T(1,0), T(2,0), T(3,0) projections of the 'chem' states and the correlation-order,
coherence-order and per-spin amplitudes of Spinach's trajan, and the three states are listed
with stateinfo.

Adaptations, applied identically to the MATLAB side in comparison/cases/li_exchange_equilibrium:
  - the script's time axis for the 2 s trajectory is linspace(0, 1, npts); it is kept as written;
  - the four subplots become four figures (tensors, correlation, coherence, spins);
  - the number of steps of the two evolutions are options (nsteps_ex, nsteps_eq) for quick checks.
Stage labels (create, basis, hamiltonian, equilibrium, relaxation, trajectory, analysis, plot)
are those of the comparison case; keep them in step. The two sites are separate chemical
subsystems, so the complete basis has 31 states (1 + 15 + 15, no cross-site correlations); both
evolutions take the exponential propagator path and the whole script runs in about a second.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import spynach
from spynach.conventions import eeqq2nqi


class Timer:
    """Stage timer: ``with timer("label"):`` records and prints the wall-clock time of the block."""

    def __init__(self, verbose: bool = True):
        self.records: list[tuple[str, float]] = []
        self.verbose = verbose

    @contextmanager
    def __call__(self, label: str):
        t0 = time.perf_counter()
        yield
        seconds = time.perf_counter() - t0
        self.records.append((label, seconds))
        if self.verbose:
            print(f"{label:32s} {seconds:8.1f} s", flush=True)


def simulate(nsteps_ex: int = 1_000_000, nsteps_eq: int = 200_000, timer: Timer | None = None):
    """Runs the script; returns (spin_system, time_eq, data) with the states, projections and trajan amplitudes."""
    timer = Timer() if timer is None else timer

    # Magnet and spins
    sys_ = {"magnet": 1.93, "isotopes": ["7Li", "7Li"]}  # nu_7Li = 32 MHz

    # Quadrupolar tensors for A and B (Cq in Hz, eta, I = 3/2, Euler angles in rad)
    Cq = 30e3
    coupling_matrix = [
        [eeqq2nqi(Cq, 0, 3 / 2, [0, 0, 0]).tolist(), None],
        [None, eeqq2nqi(3 * Cq, 0, 3 / 2, [0, 0, 0]).tolist()],
    ]

    # Chemical shifts (ppm)
    sigma_iso, delta_sigma, eta = -1.0, 45.0, 0.0
    zeeman_matrix = [
        np.diag(
            [sigma_iso - delta_sigma / 2 * (1 + eta), sigma_iso - delta_sigma / 2 * (1 - eta), sigma_iso + delta_sigma]
        ).tolist(),
        np.diag(
            [
                sigma_iso - 3 * delta_sigma / 2 * (1 + eta),
                sigma_iso - 3 * delta_sigma / 2 * (1 - eta),
                sigma_iso + 3 * delta_sigma,
            ]
        ).tolist(),
    ]

    # Chemical exchange A <-> B
    pA = 0.95
    pB = 1 - pA
    kex = 1e3
    kAB, kBA = pB * kex, pA * kex
    rates = np.array([[-kAB, kBA], [kAB, -kBA]])
    concs = spynach.equilibrate(rates, np.array([1.0, kAB / kBA]))  # sanity check, as in the script

    inter = {
        "coupling": {"matrix": coupling_matrix},
        "zeeman": {"matrix": zeeman_matrix},
        "chem": {"parts": [[0], [1]], "rates": rates.tolist(), "concs": concs.tolist()},
        "temperature": 298,
        # Relaxation model
        "relaxation": ["redfield"],
        "rlx_keep": "secular",
        "equilibrium": "zero",
        "tau_c": [[1e-9], [1e-9]],
    }

    # Basis and formalism
    bas = {"formalism": "sphten-liouv", "approximation": "none"}

    # Spinach init
    with timer("create"):
        spin_system = spynach.create(sys_, inter)
    with timer("basis"):
        spin_system = spynach.basis(spin_system, bas)

    # Liouvillian and equilibrium state
    with timer("hamiltonian"):
        H0 = spynach.hamiltonian(spynach.assume(spin_system, "labframe"), "left")
        Lx = spynach.operator(spin_system, "Lx", "7Li")
        H = spynach.hamiltonian(spynach.assume(spin_system, "nmr"))
        K = spynach.kinetics(spin_system)
    with timer("equilibrium"):
        rho_eq = spynach.equilibrium(spin_system, H0)
        rho_eq_ex = spynach.evolution(
            spin_system, H + 1j * K, rho_eq, 1e-5, nsteps_ex, "final"
        )  # 10 s of chemical equilibration
    with timer("relaxation"):
        R = spynach.thermalize(spin_system, spynach.relaxation(spin_system), "IME", rho_eq=rho_eq_ex)
        L0 = H + 1j * R + 1j * K
    with timer("trajectory"):
        rho_pert = spynach.step(spin_system, Lx, rho_eq_ex, 0)  # apply perturbation
        traj_eq = spynach.evolution(
            spin_system, L0, rho_pert, 1e-5, nsteps_eq, "trajectory"
        )  # 2 s for full equilibration

    # Analyse the trajectory during equilibration
    with timer("analysis"):
        time_eq = np.linspace(0, 1, traj_eq.shape[1])
        T10 = spynach.state(spin_system, "T1,0", "7Li", "chem")
        T20 = spynach.state(spin_system, "T2,0", "7Li", "chem")
        T30 = spynach.state(spin_system, "T3,0", "7Li", "chem")
        tensors = np.stack([np.conj(T) @ traj_eq for T in (T10, T20, T30)])
        corr_order, corr_labels = spynach.trajan(spin_system, traj_eq, "correlation_order")
        coh_order, coh_labels = spynach.trajan(spin_system, traj_eq, "coherence_order")
        spin_total, spin_labels = spynach.trajan(spin_system, traj_eq, "total_each_spin")

        # Density operator analysis
        print("Density Operator after equilibrium(H0)")
        spynach.stateinfo(spin_system, rho_eq, 16)
        print("Density Operator after H+1i*K (targeted state)")
        spynach.stateinfo(spin_system, rho_eq_ex, 16)
        print("Density Operator after perturbation + H+1i*R+1i*K")
        spynach.stateinfo(spin_system, traj_eq[:, -1], 16)
        unit = spynach.state(spin_system, "E", "all")
        unit_population = np.conj(unit) @ rho_eq_ex / np.linalg.norm(unit)
        print(f"Unity State population in the targeted state: {unit_population.real:g}")

    data = {
        "rho_eq": rho_eq,
        "rho_eq_ex": rho_eq_ex,
        "rho_final": traj_eq[:, -1],
        "tensors": tensors,
        "corr_order": corr_order,
        "coh_order": coh_order,
        "spin_total": spin_total,
        "unit_population": np.array(unit_population),
    }
    labels = {"corr_order": corr_labels, "coh_order": coh_labels, "spin_total": spin_labels}
    return spin_system, time_eq, data, labels


def plot(spin_system, time_eq, data, labels, timer: Timer | None = None) -> dict:
    """The four panels of the script's figure 1 as separate figures: tensors, correlation, coherence, spins."""
    from spynach.plotting import plot_time_series

    timer = Timer() if timer is None else timer
    with timer("plot"):
        figs = {
            "tensors": plot_time_series(
                time_eq,
                data["tensors"],
                names=["T10", "T20", "T30"],
                xlabel="Equilibration time (s)",
                ylabel="projection",
                title="T(l,0) tensors",
            ),
            "correlation": plot_time_series(
                time_eq,
                data["corr_order"],
                names=labels["corr_order"],
                xlabel="Equilibration time (s)",
                ylabel="correlation order amplitude",
                title="correlation orders",
            ),
            "coherence": plot_time_series(
                time_eq,
                data["coh_order"],
                names=labels["coh_order"],
                xlabel="Equilibration time (s)",
                ylabel="coherence order amplitude",
                title="coherence orders",
            ),
            "spins": plot_time_series(
                time_eq,
                data["spin_total"],
                names=labels["spin_total"],
                xlabel="Equilibration time (s)",
                ylabel="density touching each spin",
                title="spin populations",
            ),
        }
    return figs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nsteps-ex", type=int, default=1_000_000, help="steps of 10 us for the chemical equilibration")
    ap.add_argument("--nsteps-eq", type=int, default=200_000, help="steps of 10 us for the trajectory")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("out"))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")  # stateinfo reports, as Spinach prints them

    spin_system, time_eq, data, labels = simulate(args.nsteps_ex, args.nsteps_eq, Timer())
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "li_exchange_equilibrium.npz", time_eq=time_eq, **data)
    print(f"states, projections and trajectory amplitudes written to {args.out}/")
    if not args.no_plot:
        for name, fig in plot(spin_system, time_eq, data, labels).items():
            fig.write_html(args.out / f"li_exchange_equilibrium_{name}.html")
        print(f"figures written to {args.out}/")


if __name__ == "__main__":
    main(sys.argv[1:])
