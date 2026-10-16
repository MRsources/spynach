"""ZULF spectrum of PF6- after an adiabatic field drop and a selective pulse: translation of user_ml/ZULF_PF6_1D.m.

    OPENBLAS_NUM_THREADS=4 PYTHONPATH=src python examples/user_ml/zulf_pf6_1d.py \
        [--npoints N] [--zerofill N] [--out DIR] [--no-plot]

The contributed script (Florin Teleanu, 2024): 31P with six equivalent 19F (1J_PF = 710 Hz, octahedral
coordinates at 1.6 Angstrom) in a 2.7 uT solenoid field, complete sphten-liouv basis (16 384 states),
lab-frame Hamiltonian. The initial state, read from user_ml/rho_adiabatic_drop.csv, is the density
matrix after the adiabatic drop in a ZULF BELT; a 150 us gamma-weighted y-pulse of 30 uT follows, then
16 384 points of gamma-weighted Lz detection at 6 kHz sweep, exponential apodisation and the magnitude
spectrum on a Hz axis.

Adaptations, applied identically to the MATLAB side in comparison/cases/zulf_pf6_1d (the original
targets a pre-2.9 Spinach and does not run on 2.13 as written):
  - inter.coordinates=rPF*{...} (a number times a cell array) is written out as the scaled coordinates;
  - apodization(fid-mean(fid),'exp-1d',1) is the legacy function; 2.13 uses apodisation(spin_system,
    fid-mean(fid),{{'exp',1}}), which also halves the first point, as respinach's apodisation does;
  - sys.output='hush' and sys.enable={'greedy'} only affect console output and parallelisation and
    are dropped; the relaxation fields are kept (create validates them) but, as in the script, R is
    never built: the pulse and the acquisition evolve under H alone.
Stage labels (create, basis, hamiltonian, operators, pulse, acquisition, processing, plot) are those of
the comparison case; keep them in step. Full size takes about 36 s on four cores: evolution() reduces
the 16 384-state problem (zero track elimination and path tracing, as in Spinach) to subspaces below
tols.krylov_tol, so both evolutions use the exponential propagator.
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import scipy.sparse as sps

import respinach
from respinach.isotopes import spin

ROOT = Path(__file__).resolve().parents[2]
RHO_PATH = ROOT / "user_ml" / "rho_adiabatic_drop.csv"


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


def load_matlab_csv(path: Path | str) -> np.ndarray:
    """Reads a column written by MATLAB's csvwrite: real or complex numbers in ``a+bi`` form."""
    return np.array(
        [complex(line.strip().replace("i", "j")) for line in Path(path).read_text().split() if line.strip()]
    )


def _dense(v) -> np.ndarray:
    return np.asarray(v.toarray() if sps.issparse(v) else v).reshape(-1)


def simulate(
    npoints: int = 4 * 4096, zerofill: int = 4 * 4096, timer: Timer | None = None, rho_path: Path | str = RHO_PATH
):
    """Runs the experiment; returns (spin_system, parameters, data) with data = rho0, rho_pulse, fid, spectrum."""
    timer = Timer() if timer is None else timer

    # Solenoid field
    sys_ = {"magnet": 2.7e-6}

    # Spin system
    sys_["isotopes"] = ["31P", "19F", "19F", "19F", "19F", "19F", "19F"]

    # 1J_PF coupling system for 2M LiPF6
    J_PF, rPF = 710.0, 1.6
    scalar = [[0.0] * 7 for _ in range(7)]
    for k in range(1, 7):
        scalar[0][k] = J_PF
    directions = [[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
    inter = {"coupling": {"scalar": scalar}, "coordinates": [[rPF * x for x in d] for d in directions]}

    # Basis set
    bas = {"formalism": "sphten-liouv", "approximation": "none"}

    # Relaxation theory (validated by create; R is never built, as in the script)
    inter.update(
        relaxation=["redfield"], equilibrium="dibari", temperature=298, rlx_keep="labframe", tau_c=[[0.065e-9]]
    )

    # Spinach housekeeping
    with timer("create"):
        spin_system = respinach.create(sys_, inter)
    with timer("basis"):
        spin_system = respinach.basis(spin_system, bas)
    with timer("hamiltonian"):
        H = respinach.hamiltonian(respinach.assume(spin_system, "labframe"), "comm")

    # Density operator after the adiabatic drop
    rho = load_matlab_csv(rho_path)
    if rho.size != spin_system.basis.nstates:
        raise ValueError(f"{rho_path} has {rho.size} entries, the basis has {spin_system.basis.nstates} states")

    # Sequence parameters
    parameters = {
        "sweep": 6000,
        "npoints": npoints,
        "zerofill": zerofill,
        "offset": 0,
        "spins": ["19F"],
        "axis_units": "Hz",
        "invert_axis": 0,
        "detection": "quadrature",
    }

    with timer("operators"):
        # Magnetogyric ratio weights relative to 19F
        weights = spin_system.gammas / spin("19F")[0]

        # Gamma-weighted detection states
        coilPF6X = sum(
            weights[n]
            * (_dense(respinach.state(spin_system, "L+", n)) + _dense(respinach.state(spin_system, "L-", n)))
            / 2
            for n in range(7)
        )
        coilPF6Z = sum(weights[n] * _dense(respinach.state(spin_system, "Lz", n)) for n in range(7))

        # Gamma-weighted pulse operators
        Sy = sum(
            weights[n] * (respinach.operator(spin_system, "L+", n) - respinach.operator(spin_system, "L-", n)) / 2j
            for n in range(7)
        )
        Sx = sum(
            weights[n] * (respinach.operator(spin_system, "L+", n) + respinach.operator(spin_system, "L-", n)) / 2
            for n in range(7)
        )

    with timer("pulse"):
        rho_pulse = respinach.evolution(spin_system, H + spin("19F")[0] * 30e-6 * Sy, None, rho, 1e-6, 150, "final")
    with timer("acquisition"):
        fid = respinach.evolution(
            spin_system, H, coilPF6Z, rho_pulse, 1 / parameters["sweep"], parameters["npoints"] - 1, "observable"
        )

    with timer("processing"):
        # Apodisation (legacy apodization(fid-mean(fid),'exp-1d',1))
        fid_ap = respinach.apodisation(fid - fid.mean(), [("exp", 1)])

        # Fourier transform
        spectrum = np.fft.fftshift(np.fft.fft(fid_ap, parameters["zerofill"]))

    data = {
        "rho0": rho,
        "rho_pulse": rho_pulse,
        "fid": fid,
        "spectrum": spectrum,
        "coil_x": coilPF6X,
        "coil_z": coilPF6Z,
    }
    del Sx  # built as in the script, unused in the 1D experiment
    return spin_system, parameters, data


def plot(spin_system, parameters, spectrum, timer: Timer | None = None):
    """The figure of the script: plot_1d(spin_system,abs(spectrum),parameters)."""
    from respinach.plotting import plot_1d

    timer = Timer() if timer is None else timer
    with timer("plot"):
        return plot_1d(
            spin_system,
            np.abs(spectrum),
            parameters,
            title="PF6- ZULF spectrum after adiabatic drop and selective pulse",
        )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npoints", type=int, default=4 * 4096)
    ap.add_argument("--zerofill", type=int, default=4 * 4096)
    ap.add_argument("--rho", type=Path, default=RHO_PATH, help="initial state CSV (MATLAB csvwrite format)")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("out"))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    spin_system, parameters, data = simulate(args.npoints, args.zerofill, Timer(), args.rho)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("fid", "spectrum", "rho_pulse"):
        np.save(args.out / f"zulf_pf6_1d_{name}.npy", data[name])
    print(f"FID, spectrum and pulsed state written to {args.out}/")
    if not args.no_plot:
        fig = plot(spin_system, parameters, data["spectrum"])
        fig.write_html(args.out / "zulf_pf6_1d.html")
        print(f"figure written to {args.out / 'zulf_pf6_1d.html'}")


if __name__ == "__main__":
    main(sys.argv[1:])
