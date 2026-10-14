"""Comparison case: 7Li two-site exchange equilibration, examples/user_ml/li_exchange_equilibrium.py with timed stages.

Stage labels must match li_exchange_equilibrium_spinach.m: create, basis, hamiltonian, equilibrium,
relaxation, trajectory, analysis, plot.
"""

from __future__ import annotations

import logging
import sys

from comparison.run_respinach import load_example

example = load_example("user_ml/li_exchange_equilibrium")


def run(timer, nsteps_ex: int = 1_000_000, nsteps_eq: int = 200_000, gpu: bool = False) -> tuple[dict, dict]:
    """Returns (data, figs): the states, T(l,0) projections, trajan amplitudes and unit population; four figures."""
    if gpu:
        raise NotImplementedError("sys['enable'] 'gpu' is not available yet: the GPU backend is planned, see porting/gpu-notes.md")
    # stateinfo reports go to the console log, as Spinach's do
    logger = logging.getLogger("respinach")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        spin_system, time_eq, data, labels = example.simulate(nsteps_ex, nsteps_eq, timer)
    finally:
        logger.removeHandler(handler)
    data = {**data, "time_eq": time_eq}
    return data, example.plot(spin_system, time_eq, data, labels, timer)
