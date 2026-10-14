"""Comparison case: NOESY of strychnine, examples/nmr_liquids/noesy_strychnine.py with timed stages.

Stage labels must match noesy_strychnine_spinach.m: create, basis, liquid/noesy, processing, plot.
"""

from __future__ import annotations

from comparison.run_respinach import load_example

example = load_example("nmr_liquids/noesy_strychnine")


def run(timer, npoints: int = 512, zerofill: int = 2048, gpu: bool = False) -> tuple[dict, dict]:
    """Returns (data, figs): data has fid_cos, fid_sin (raw FIDs) and spectrum; figs has the plot_2d figure."""
    spin_system, parameters, fid, spectrum = example.simulate(npoints, zerofill, timer, gpu=gpu)
    data = {"fid_cos": fid["cos"], "fid_sin": fid["sin"], "spectrum": spectrum}
    figs = {"noesy": example.plot(spin_system, parameters, spectrum, timer)}
    return data, figs
