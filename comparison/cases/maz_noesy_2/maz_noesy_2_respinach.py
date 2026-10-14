"""Comparison case: methylaziridine NOESY with scalar relaxation, examples/relaxation_theory/maz_noesy_2.py with timed stages.

Stage labels must match maz_noesy_2_spinach.m: create, basis, liquid/noesy, processing, plot.
"""

from __future__ import annotations

from comparison.run_respinach import load_example

example = load_example("relaxation_theory/maz_noesy_2")


def run(timer, npoints: int = 256, zerofill: int = 1024, gpu: bool = False) -> tuple[dict, dict]:
    """Returns (data, figs): data has fid_cos, fid_sin (raw FIDs) and spectrum; figs has the plot_2d figure."""
    spin_system, parameters, fid, spectrum = example.simulate(npoints, zerofill, timer, gpu=gpu)
    data = {"fid_cos": fid["cos"], "fid_sin": fid["sin"], "spectrum": spectrum}
    figs = {"noesy": example.plot(spin_system, parameters, spectrum, timer)}
    return data, figs
