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

import respinach

from respinach.molecules import strychnine as strychnine_molecule


def strychnine(relaxation: bool = False) -> tuple[dict, dict]:
    """sys/inter of the NOESY example: the 1H strychnine molecule at 5.9 T, optionally with its relaxation fields."""
    sys_, inter = strychnine_molecule(["1H"])
    sys_.update(magnet=5.9, tols={"prox_cutoff": 4.0})
    if relaxation:
        inter.update(relaxation=["redfield"], equilibrium="IME", temperature=298, rlx_keep="kite", tau_c=[[200e-12]])
    return sys_, inter


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
    ss = respinach.create(sys_, inter)
    bas = {"formalism": "sphten-liouv", "approximation": "IK-2", "connectivity": "scalar_couplings", "prox_level": 3}
    ss = timed("basis IK-2 prox 3", lambda: respinach.basis(ss, bas), profile)
    print(f"{'states':40s} {ss.basis.nstates:8d}")
    nmr = respinach.assume(ss, "nmr")
    H = timed("hamiltonian", lambda: respinach.hamiltonian(nmr), profile)
    R = timed("relaxation (Redfield, kite, IME)", lambda: respinach.relaxation(ss), profile)
    pars = {"spins": ["1H"], "offset": 1200, "sweep": 2500, "npoints": 256, "rho0": respinach.state(ss, "L+", "1H"),
            "coil": respinach.state(ss, "L+", "1H")}
    timed("liquid/acquire 256 points", lambda: respinach.liquid(ss, respinach.acquire, pars, "nmr"), profile)
    cosy_pars = {"angle": np.pi / 2, "offset": 1200, "sweep": 2500, "npoints": [32, 64], "spins": ["1H"]}
    timed("liquid/cosy 32 x 64", lambda: respinach.liquid(ss, respinach.cosy, cosy_pars, "nmr"), profile)


if __name__ == "__main__":
    main("--profile" in sys.argv)
