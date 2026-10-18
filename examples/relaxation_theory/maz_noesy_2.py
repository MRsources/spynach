"""Methylaziridine NOESY with scalar relaxation: translation of Spinach's examples/relaxation_theory/maz_noesy_2.m.

    OPENBLAS_NUM_THREADS=4 PYTHONPATH=src python examples/relaxation_theory/maz_noesy_2.py [options]

Options:
    --npoints N      points per dimension (default 256, as in Spinach)
    --zerofill N     zero-filling per dimension (default 1024, as in Spinach)
    --out DIR        output directory (default: out/ next to this script, git-ignored)
    --compare FILE   MATLAB .mat file with the Spinach result (see below)
    --no-plot        skip the Plotly figure

Seven protons and the 14N of methylaziridine with vacuum DFT shielding tensors
(CSA), the 14N quadrupolar tensor and coordinates; Redfield theory with scalar
relaxation of the first kind (the H-N couplings modulated by the nitrogen
inversion, srfk_mdepth 15 Hz, srfk_tau_c 1 ms) and of the second kind (the fast
quadrupolar relaxation of 14N transferred to the protons through J), IK-2 basis,
2 s mixing time; the effect described in http://dx.doi.org/10.1002/ange.201410271.

Outputs in DIR: maz_noesy_2_fid_cos.npy, maz_noesy_2_fid_sin.npy (raw FIDs, shape
(npoints, npoints) with the direct dimension along axis 0, as Spinach's fid.cos),
maz_noesy_2_spectrum.npy (the States spectrum, (zerofill, zerofill)), and
maz_noesy_2.html (the plot_2d figure).

Comparison with Spinach. Add these lines to maz_noesy_2.m, before the apodisation
and after the F1 Fourier transform respectively, then run it in MATLAB:

    fid_cos=fid.cos; fid_sin=fid.sin;                                % before 'Apodisation'
    save('maz_noesy_2.mat','fid_cos','fid_sin','spectrum','-v7');    % after 'F1 Fourier transform'

and run this script with --compare maz_noesy_2.mat. It prints the normwise relative
error ||spynach - spinach||_F / ||spinach||_F of the two FIDs and the spectrum, the
metric of spynach.oracle.compare. The sizes must match: run both at the same
npoints and zerofill.

Differences from the MATLAB script: spin indices are 0-based, so the 14N source
spin is 3 and the modulated couplings are (0,4), (1,4), (2,4); the shielding
tensors are converted to shift tensors with the experimental isotropic shifts
exactly as the MATLAB loop does, before create(). spynach runs serially.
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import spynach

# Absolute shielding tensors (vacuum DFT), ppm, in the order H3, H2, H4, N, HN, Me, Me, Me
SHIELDING = [
    [[32.7075, 6.1977, -3.6061], [3.3757, 29.2462, -0.8672], [-6.8603, -3.1603, 29.2097]],
    [[27.5505, -0.4604, -3.9670], [2.4491, 30.8072, 3.5941], [0.0951, 5.9399, 33.5623]],
    [[27.8325, -1.3662, -3.4664], [-0.3195, 23.8084, -0.9356], [1.9567, -0.5013, 39.0966]],
    [[257.9722, 16.9286, -28.3305], [22.7951, 157.2701, 18.5357], [-13.8903, -20.6605, 275.1264]],
    [[28.3494, 3.7981, -3.9745], [-1.0065, 31.8697, -4.2119], [-1.1997, -10.1395, 38.9642]],
    [[33.2953, 4.3402, 0.8099], [3.3922, 29.9675, 1.2225], [-0.7891, 0.1926, 26.6461]],
    [[33.2302, -4.7306, 1.2366], [-3.6419, 30.0443, -1.1180], [-0.3469, -0.7420, 26.5551]],
    [[31.4875, 0.1078, -2.9339], [0.1534, 27.9566, 0.6704], [-3.2188, 0.7748, 35.7251]],
]
EXPT_SHIFT = [1.3, 1.7, 1.9, 0.0, 0.1, 1.2, 1.2, 1.2]  # isotropic components from the experiment, ppm
QUADRUPOLE_14N = [[-1.2932, 0.6251, 1.8700], [0.6251, 1.7170, -2.3127], [1.8700, -2.3127, -0.4238]]  # x 1e6 Hz
SCALAR_COUPLINGS = {
    (0, 1): -11.6,
    (0, 2): 5.5,
    (0, 4): 7.5,
    (1, 4): 12.7,
    (2, 5): 4.5,
    (2, 6): 4.5,
    (2, 7): 4.5,
    (3, 0): 4.4,
    (3, 2): 5.2,
    (3, 4): 44.8,
}  # Hz, Spinach's (n,k)
COORDINATES = [
    [-1.812977, -1.098554, 0.444452],
    [-0.842255, -1.116468, -1.122812],
    [0.176983, -0.023347, 1.610045],
    [-0.907675, 0.811819, -0.038237],
    [-0.590140, 1.188036, -0.936471],
    [2.093648, 0.832363, 0.088912],
    [2.084298, -0.944150, 0.159225],
    [1.387021, -0.108214, -1.250363],
]


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


def spin_system_inputs(gpu: bool = False) -> tuple[dict, dict, dict]:
    """The sys, inter and bas dictionaries of the Spinach script."""
    nspins = 8
    sys_ = {
        "isotopes": ["1H", "1H", "1H", "14N", "1H", "1H", "1H", "1H"],
        "magnet": 11.75,
        "tols": {"inter_cutoff": 2.0, "prox_cutoff": 4.0},
        "disable": ["krylov"],
    }
    if gpu:
        sys_["enable"] = ["gpu"]

    # Assign isotropic components from the experiment: shift = expt_shift*1 - (shielding - trace/3)
    shift_tensors = []
    for shielding, shift in zip(SHIELDING, EXPT_SHIFT):
        M = np.asarray(shielding, dtype=float)
        M = M - np.eye(3) * np.trace(M) / 3
        shift_tensors.append((np.eye(3) * shift - M).tolist())

    coupling_matrix = [[None] * nspins for _ in range(nspins)]
    coupling_matrix[3][3] = (np.asarray(QUADRUPOLE_14N) * 1e6).tolist()
    scalar = [[None] * nspins for _ in range(nspins)]
    for (n, k), J in SCALAR_COUPLINGS.items():
        scalar[n][k] = J
    mdepth = [[None] * nspins for _ in range(nspins)]
    for n in (0, 1, 2):
        mdepth[n][4] = 15.0

    inter = {
        "zeeman": {"matrix": shift_tensors},
        "coupling": {"matrix": coupling_matrix, "scalar": scalar},
        "coordinates": COORDINATES,
        "relaxation": ["redfield", "SRFK", "SRSK"],
        "equilibrium": "zero",
        "srsk_sources": [3],
        "tau_c": [[25e-12]],
        "rlx_keep": "kite",
        "srfk_tau_c": [[1.0, 1e-3]],
        "srfk_mdepth": mdepth,
    }
    bas = {"formalism": "sphten-liouv", "approximation": "IK-2", "prox_level": 4, "connectivity": "scalar_couplings"}
    return sys_, inter, bas


def simulate(npoints: int = 256, zerofill: int = 1024, timer: Timer | None = None, gpu: bool = False):
    """Run the NOESY experiment; returns (spin_system, parameters, fid, spectrum).

    The stages are timed with ``timer`` (default: one that prints); the labels
    are those of comparison/cases/maz_noesy_2, keep them in step. ``gpu``
    adds ``'gpu'`` to ``sys['enable']``; spynach refuses it until the GPU
    backend exists (porting/gpu-notes.md).
    """
    timer = Timer() if timer is None else timer
    sys_, inter, bas = spin_system_inputs(gpu)

    # Spinach housekeeping
    with timer("create"):
        spin_system = spynach.create(sys_, inter)
    with timer("basis"):
        spin_system = spynach.basis(spin_system, bas)

    # Sequence parameters
    parameters = {
        "spins": ["1H"],
        "offset": 500,
        "sweep": [1400, 1400],
        "tmix": 2.0,
        "npoints": [npoints, npoints],
        "zerofill": [zerofill, zerofill],
        "axis_units": "ppm",
        "rho0": spynach.state(spin_system, "Lz", "1H"),
    }

    # Simulation
    with timer("liquid/noesy"):
        fid = spynach.liquid(spin_system, spynach.noesy, parameters, "nmr")

    with timer("processing"):
        # Apodisation
        cos = spynach.apodisation(fid["cos"], [("cos",), ("cos",)])
        sin = spynach.apodisation(fid["sin"], [("cos",), ("cos",)])

        # F2 Fourier transform
        f1_cos = np.real(np.fft.fftshift(np.fft.fft(cos, parameters["zerofill"][1], axis=0), axes=0))
        f1_sin = np.real(np.fft.fftshift(np.fft.fft(sin, parameters["zerofill"][1], axis=0), axes=0))

        # States signal
        f1_states = f1_cos - 1j * f1_sin

        # F1 Fourier transform
        spectrum = np.fft.fftshift(np.fft.fft(f1_states, parameters["zerofill"][0], axis=1), axes=1)

    return spin_system, parameters, fid, spectrum


def plot(spin_system, parameters, spectrum, timer: Timer | None = None):
    """The figure of the Spinach script:

    plot_2d(spin_system,-real(spectrum),parameters,40,[0.025 0.50 0.025 0.50],2,256,6,'both')
    """
    from spynach.plotting import plot_2d

    timer = Timer() if timer is None else timer
    with timer("plot"):
        return plot_2d(
            spin_system,
            -spectrum.real,
            parameters,
            ncont=40,
            delta=(0.025, 0.5, 0.025, 0.5),
            k=2,
            ncol=256,
            m=6,
            signs="both",
            title="NOESY of methylaziridine with scalar relaxation, 2 s mixing",
        )


def compare_with_spinach(path: Path, fid: dict, spectrum: np.ndarray) -> None:
    """Normwise relative error of each saved array against the Spinach run (spynach.oracle.compare)."""
    from scipy.io import loadmat

    from spynach.oracle import compare

    ref = loadmat(str(path))
    print(f"comparison with {path}:")
    for name, mine in (("fid_cos", fid["cos"]), ("fid_sin", fid["sin"]), ("spectrum", spectrum)):
        if name not in ref:
            print(f"  {name:10s} missing from the .mat file")
            continue
        print(f"  {name:10s} relative error {compare(mine, ref[name], rtol=1e-9, atol=0, name=name).rel_err:.3e}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npoints", type=int, default=256)
    ap.add_argument("--zerofill", type=int, default=1024)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("out"))
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    timer = Timer()
    spin_system, parameters, fid, spectrum = simulate(args.npoints, args.zerofill, timer)

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "maz_noesy_2_fid_cos.npy", fid["cos"])
    np.save(args.out / "maz_noesy_2_fid_sin.npy", fid["sin"])
    np.save(args.out / "maz_noesy_2_spectrum.npy", spectrum)
    print(f"FIDs and spectrum written to {args.out}/")

    if args.compare is not None:
        compare_with_spinach(args.compare, fid, spectrum)

    # Plotting
    if not args.no_plot:
        fig = plot(spin_system, parameters, spectrum)
        fig.write_html(args.out / "maz_noesy_2.html")
        print(f"figure written to {args.out / 'maz_noesy_2.html'}")


if __name__ == "__main__":
    main(sys.argv[1:])
