"""Simulation contexts (port of ``liquid.m``).

A context builds the Hamiltonian, relaxation, and kinetics for a physical
setting and hands them to a pulse sequence function.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.sparse as sps

from spinpy.assume import assume
from spinpy.hamiltonian import frqoffset, hamiltonian
from spinpy.system import SpinSystem
from spinpy.utilities import Matrix

PulseSequence = Callable[[SpinSystem, dict[str, Any], Matrix, Matrix, Matrix], Any]


def liquid(
    spin_system: SpinSystem, pulse_sequence: PulseSequence, parameters: dict[str, Any], assumptions: str
) -> Any:
    """Run ``pulse_sequence(spin_system, parameters, H, R, K)`` in the isotropic liquid context.

    ``parameters`` must contain ``'spins'`` (isotopes of the working channels)
    and may contain ``'offset'`` (Hz per channel). Relaxation and chemical
    kinetics are not ported yet; R and K are zero, which is what Spinach
    produces when the system specifies neither.
    """
    parameters = dict(parameters)
    parameters.setdefault("decouple", [])
    parameters.setdefault("rframes", [])
    parameters.setdefault("offset", np.zeros(len(parameters["spins"])))
    parameters.setdefault("needs", [])
    if parameters["rframes"]:
        raise NotImplementedError("additional rotating frame transformations are not ported yet.")
    if parameters["needs"]:
        raise NotImplementedError(f"parameters['needs'] {parameters['needs']} not ported yet.")

    spin_system = assume(spin_system, assumptions)
    H = hamiltonian(spin_system)
    dim = H.shape[0]
    R = sps.csr_array((dim, dim), dtype=complex)
    K = sps.csr_array((dim, dim), dtype=complex)
    H = frqoffset(spin_system, H, parameters)

    parameters["spc_dim"] = 1
    parameters["spn_dim"] = dim
    return pulse_sequence(spin_system, parameters, H, R, K)
