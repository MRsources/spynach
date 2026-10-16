"""Comparison case: ZULF spectrum of PF6-, examples/user_ml/zulf_pf6_1d.py with timed stages.

Stage labels must match zulf_pf6_1d_spinach.m: create, basis, hamiltonian, operators, pulse,
acquisition, processing, plot.
"""

from __future__ import annotations

from comparison.run_spynach import load_example

example = load_example("user_ml/zulf_pf6_1d")


def run(
    timer, npoints: int = 4 * 4096, zerofill: int = 4 * 4096, rho_file: str | None = None, gpu: bool = False
) -> tuple[dict, dict]:
    """Returns (data, figs): data has rho0, rho_pulse, fid, spectrum, coil_x, coil_z; figs has the plot_1d figure."""
    if gpu:
        raise NotImplementedError(
            "sys['enable'] 'gpu' is not available yet: the GPU backend is planned, see porting/gpu-notes.md"
        )
    spin_system, parameters, data = example.simulate(npoints, zerofill, timer, rho_file or example.RHO_PATH)
    figs = {"spectrum": example.plot(spin_system, parameters, data["spectrum"], timer)}
    return data, figs
