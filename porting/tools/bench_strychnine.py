"""Timing benchmark on the 22-proton strychnine system of Spinach's etc/molecules/strychnine.m.

    OPENBLAS_NUM_THREADS=4 PYTHONPATH=src python porting/tools/bench_strychnine.py [--profile]

Builds the IK-2 basis with proximity level 3 (as examples/nmr_liquids/noesy_strychnine.m),
the Hamiltonian, the Redfield superoperator, and runs a pulse-acquire and a COSY, printing
wall-clock times and dimensions. With --profile, prints the cProfile top of each stage.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time

import numpy as np

import spinpy

# Spin numbering of strychnine.m (0-based here): H1 H2 H3 H4 H8 H11a H11b H12 H13 H14 H15a H15b H16 H17a H17b H18a H18b H20a H20b H22 H23a H23b
H1, H2, H3, H4, H8, H11a, H11b, H12, H13, H14, H15a, H15b, H16, H17a, H17b, H18a, H18b, H20a, H20b, H22, H23a, H23b = range(22)
SHIFTS = {H3: 7.255, H22: 5.915, H2: 7.098, H1: 7.167, H4: 8.092, H12: 4.288, H23b: 4.066, H23a: 4.148, H16: 3.963,
          H8: 3.860, H20b: 2.745, H20a: 3.716, H18a: 3.219, H18b: 2.878, H13: 1.276, H17a: 1.890, H17b: 1.890,
          H11a: 3.132, H11b: 2.670, H14: 3.150, H15b: 1.462, H15a: 2.360}
COUPLINGS = {(H3, H4): 7.90, (H22, H23a): 7.00, (H22, H23b): 6.10, (H2, H3): 7.44, (H2, H4): 0.98, (H1, H2): 7.49,
             (H1, H3): 1.08, (H1, H4): 0.23, (H12, H13): 3.30, (H23a, H23b): -13.80, (H8, H13): 10.41,
             (H20a, H20b): -14.80, (H20a, H22): 1.79, (H18a, H18b): -13.90, (H13, H14): 3.29, (H17a, H17b): -13.90,
             (H17a, H18a): 5.50, (H17a, H18b): 7.20, (H17b, H18a): 3.20, (H17b, H18b): 10.70, (H11a, H11b): -17.34,
             (H11a, H12): 3.34, (H11b, H12): 8.47, (H14, H15a): 4.11, (H14, H15b): 1.96, (H14, H22): 0.47,
             (H14, H20a): 1.61, (H15a, H15b): -14.35, (H15a, H16): 4.33, (H15b, H16): 2.42}
COORDINATES = {
    H1: [2.7336, -2.8961, -0.4256], H2: [5.0747, -2.1373, -0.7803], H3: [5.6630, 0.2547, -0.5015],
    H4: [3.9144, 1.9280, 0.1137], H8: [-0.6004, 0.4789, 1.3185], H11a: [-0.3262, 4.0926, -0.0781],
    H11b: [-0.6863, 3.1492, 1.3616], H12: [-1.7489, 2.7068, -1.3360], H13: [-0.0861, 1.0611, -1.5871],
    H14: [-2.2234, 0.1869, -2.2835], H15a: [0.0099, -1.0345, -2.4296], H15b: [-1.3580, -2.1373, -2.4297],
    H16: [0.3886, -2.9088, -0.8537], H17a: [1.0318, -2.7083, 1.7742], H17b: [0.6059, -1.1921, 2.5799],
    H18a: [-1.2874, -3.1432, 2.3544], H18b: [-1.7510, -1.4610, 2.0500], H20a: [-2.9738, -2.8430, -1.0448],
    H20b: [-3.4330, -2.5457, 0.6228], H22: [-4.4597, -0.3928, 0.8115], H23a: [-3.8532, 1.8278, -1.0768],
    H23b: [-4.5333, 1.9818, 0.5498],
}


def strychnine(relaxation: bool = False) -> dict:
    scalar = [[0.0] * 22 for _ in range(22)]
    for (n, k), J in COUPLINGS.items():
        scalar[n][k] = J
    inter = {"zeeman": {"scalar": [SHIFTS[n] for n in range(22)]}, "coupling": {"scalar": scalar},
             "coordinates": [COORDINATES[n] for n in range(22)]}
    if relaxation:
        inter.update(relaxation=["redfield"], equilibrium="IME", temperature=298, rlx_keep="kite", tau_c=[[200e-12]])
    return {"isotopes": ["1H"] * 22, "magnet": 5.9, "tols": {"prox_cutoff": 4.0}}, inter


def timed(label: str, fn, profile: bool):
    pr = cProfile.Profile() if profile else None
    t0 = time.perf_counter()
    if pr:
        pr.enable()
    out = fn()
    if pr:
        pr.disable()
    print(f"{label:40s} {time.perf_counter() - t0:8.2f} s")
    if pr:
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(10)
        print("\n".join(line for line in s.getvalue().splitlines()[6:22]))
    return out


def main(profile: bool) -> None:
    sys_, inter = strychnine(relaxation=True)
    ss = spinpy.create(sys_, inter)
    bas = {"formalism": "sphten-liouv", "approximation": "IK-2", "connectivity": "scalar_couplings", "prox_level": 3}
    ss = timed("basis IK-2 prox 3", lambda: spinpy.basis(ss, bas), profile)
    print(f"{'states':40s} {ss.basis.nstates:8d}")
    nmr = spinpy.assume(ss, "nmr")
    H = timed("hamiltonian", lambda: spinpy.hamiltonian(nmr), profile)
    R = timed("relaxation (Redfield, kite, IME)", lambda: spinpy.relaxation(ss), profile)
    pars = {"spins": ["1H"], "offset": 1200, "sweep": 2500, "npoints": 256, "rho0": spinpy.state(ss, "L+", "1H"),
            "coil": spinpy.state(ss, "L+", "1H")}
    timed("liquid/acquire 256 points", lambda: spinpy.liquid(ss, spinpy.acquire, pars, "nmr"), profile)
    cosy_pars = {"angle": np.pi / 2, "offset": 1200, "sweep": 2500, "npoints": [32, 64], "spins": ["1H"]}
    timed("liquid/cosy 32 x 64", lambda: spinpy.liquid(ss, spinpy.cosy, cosy_pars, "nmr"), profile)


if __name__ == "__main__":
    main("--profile" in sys.argv)
