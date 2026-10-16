"""NOESY spectrum of strychnine: translation of Spinach's examples/nmr_liquids/noesy_strychnine.m.

    OPENBLAS_NUM_THREADS=4 PYTHONPATH=src python examples/nmr_liquids/noesy_strychnine.py [options]

Options:
    --npoints N      points per dimension (default 512, as in Spinach)
    --zerofill N     zero-filling per dimension (default 2048, as in Spinach)
    --out DIR        output directory (default: out/ next to this script, git-ignored)
    --compare FILE   MATLAB .mat file with the Spinach result (see below)
    --no-plot        skip the Plotly figure

Outputs in DIR: noesy_strychnine_fid_cos.npy, noesy_strychnine_fid_sin.npy (raw FIDs,
shape (npoints, npoints) with the direct dimension along axis 0, as Spinach's fid.cos),
noesy_strychnine_spectrum.npy (the States spectrum, (zerofill, zerofill)), and
noesy_strychnine.html (the plot_2d figure).

Comparison with Spinach. Add these lines to noesy_strychnine.m, before the apodisation
and after the F1 Fourier transform respectively, then run it in MATLAB:

    fid_cos=fid.cos; fid_sin=fid.sin;                                % before 'Apodisation'
    save('noesy_strychnine.mat','fid_cos','fid_sin','spectrum','-v7');  % after 'F1 Fourier transform'

and run this script with --compare noesy_strychnine.mat. It prints the normwise relative
error ||respinach - spinach||_F / ||spinach||_F of the two FIDs and the spectrum, the metric of
respinach.oracle.compare. The sizes must match: run both at the same npoints and zerofill.

Differences from the MATLAB script: sys.enable={'greedy'} is a Spinach parallelisation
hint with no effect on the numbers and is omitted (respinach refuses unknown enable tokens);
respinach runs serially. Full size (512 x 512, 12 316 states) takes 2 min 50 s on four cores in the
development container (Spinach: "minutes"); 64 x 64 takes 15 s. The figure is drawn by
respinach.plotting.plot_2d, which bins the 2048 x 2048 spectrum to 512 x 512 for the HTML file.
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import respinach
from respinach.molecules import strychnine


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


def simulate(npoints: int = 512, zerofill: int = 2048, timer: Timer | None = None, gpu: bool = False):
    """Run the NOESY experiment; returns (spin_system, parameters, fid, spectrum).

    The stages are timed with ``timer`` (default: one that prints); the labels
    are those of comparison/cases/noesy_strychnine, keep them in step. ``gpu``
    adds ``'gpu'`` to ``sys['enable']`` as Spinach's ``sys.enable={'gpu'}``
    would; respinach refuses it until the GPU backend exists (porting/gpu-notes.md).
    """
    timer = Timer() if timer is None else timer

    # Spin system properties
    sys_, inter = strychnine(["1H"])

    # Magnet field
    sys_["magnet"] = 5.9

    # Basis set
    bas = {"formalism": "sphten-liouv", "approximation": "IK-2", "connectivity": "scalar_couplings", "prox_level": 3}

    # Relaxation theory parameters
    inter["relaxation"] = ["redfield"]
    inter["equilibrium"] = "IME"
    inter["temperature"] = 298
    inter["rlx_keep"] = "kite"
    inter["tau_c"] = [[200e-12]]

    # Algorithmic options ('greedy' is a parallelisation hint in Spinach and has no equivalent here)
    sys_["disable"] = ["krylov"]
    sys_["tols"] = {"prox_cutoff": 4.0}
    if gpu:
        sys_["enable"] = ["gpu"]

    # Spinach housekeeping
    with timer("create"):
        spin_system = respinach.create(sys_, inter)
    with timer("basis"):
        spin_system = respinach.basis(spin_system, bas)

    # Sequence parameters
    parameters = {
        "tmix": 0.5,
        "offset": 1200,
        "sweep": [2500, 2500],
        "npoints": [npoints, npoints],
        "zerofill": [zerofill, zerofill],
        "spins": ["1H"],
        "axis_units": "ppm",
        "needs": ["rho_eq"],
    }

    # Simulation
    with timer("liquid/noesy"):
        fid = respinach.liquid(spin_system, respinach.noesy, parameters, "nmr")

    with timer("processing"):
        # Apodisation
        cos = respinach.apodisation(fid["cos"], [("sqcos",), ("sqcos",)])
        sin = respinach.apodisation(fid["sin"], [("sqcos",), ("sqcos",)])

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

    plot_2d(spin_system,-real(spectrum),parameters,20,[0.01 0.1 0.01 0.1],2,256,6,'both')
    """
    from respinach.plotting import plot_2d

    timer = Timer() if timer is None else timer
    with timer("plot"):
        return plot_2d(
            spin_system,
            -spectrum.real,
            parameters,
            20,
            (0.01, 0.1, 0.01, 0.1),
            2,
            256,
            6,
            "both",
            title="NOESY of strychnine, 500 ms mixing",
        )


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    """||actual - expected||_F / ||expected||_F, the normwise metric of respinach.oracle.compare."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: respinach {actual.shape}, Spinach {expected.shape}")
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def compare_with_spinach(path: Path, fid: dict, spectrum: np.ndarray) -> None:
    from scipy.io import loadmat

    ref = loadmat(str(path))
    print(f"comparison with {path}:")
    for name, mine in (("fid_cos", fid["cos"]), ("fid_sin", fid["sin"]), ("spectrum", spectrum)):
        if name not in ref:
            print(f"  {name:10s} missing from the .mat file")
            continue
        print(f"  {name:10s} relative error {relative_error(mine, ref[name]):.3e}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npoints", type=int, default=512)
    ap.add_argument("--zerofill", type=int, default=2048)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("out"))
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    timer = Timer()
    spin_system, parameters, fid, spectrum = simulate(args.npoints, args.zerofill, timer)

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "noesy_strychnine_fid_cos.npy", fid["cos"])
    np.save(args.out / "noesy_strychnine_fid_sin.npy", fid["sin"])
    np.save(args.out / "noesy_strychnine_spectrum.npy", spectrum)
    print(f"FIDs and spectrum written to {args.out}/")

    if args.compare is not None:
        compare_with_spinach(args.compare, fid, spectrum)

    # Plotting
    if not args.no_plot:
        fig = plot(spin_system, parameters, spectrum)
        fig.write_html(args.out / "noesy_strychnine.html")
        print(f"figure written to {args.out / 'noesy_strychnine.html'}")


if __name__ == "__main__":
    main(sys.argv[1:])
