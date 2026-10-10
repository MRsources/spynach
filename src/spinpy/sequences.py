"""Pulse sequences (port of ``experiments/acquire.m``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from spinpy.propagation import evolution
from spinpy.system import SpinSystem
from spinpy.utilities import Matrix


def acquire(spin_system: SpinSystem, parameters: dict[str, Any], H: Matrix, R: Matrix, K: Matrix) -> np.ndarray:
    """Free induction decay: evolve ``parameters['rho0']`` and detect ``parameters['coil']``.

    Required parameters: ``'sweep'`` (Hz, sets the dwell time 1/sweep),
    ``'npoints'``, ``'rho0'``, ``'coil'``. Returns ``npoints`` complex points.
    Decoupling, homodecoupling, and dead time are not ported yet.
    """
    for key in ("sweep", "npoints", "rho0", "coil"):
        if key not in parameters:
            raise ValueError(f"parameters['{key}'] must be present.")
    if parameters.get("decouple"):
        raise NotImplementedError("decoupling is not ported yet.")
    for key in ("homodec_oper", "dead_time"):
        if key in parameters:
            raise NotImplementedError(f"parameters['{key}'] is not ported yet.")
    L = H + 1j * R + 1j * K
    return evolution(
        spin_system, L, parameters["coil"], parameters["rho0"],
        1 / parameters["sweep"], parameters["npoints"] - 1, "observable",
    )
