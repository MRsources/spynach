"""Simulation contexts (ports of ``liquid.m``, ``crystal.m`` and ``powder.m``).

A context builds the Hamiltonian, relaxation, and kinetics for a physical
setting and hands them to a pulse sequence function with the signature
``sequence(spin_system, parameters, H, R, K)``. Relaxation is not ported
yet, so ``R`` is zero.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import resources
from typing import Any

import numpy as np
import scipy.sparse as sps

from spinpy.assume import assume
from spinpy.hamiltonian import frqoffset, hamiltonian, orientation
from spinpy.kinetics import kinetics
from spinpy.system import SpinSystem
from spinpy.utilities import Matrix

PulseSequence = Callable[[SpinSystem, dict[str, Any], Matrix, Matrix, Matrix], Any]


def liquid(
    spin_system: SpinSystem, pulse_sequence: PulseSequence, parameters: dict[str, Any], assumptions: str
) -> Any:
    """Run ``pulse_sequence(spin_system, parameters, H, R, K)`` in the isotropic liquid context.

    ``parameters`` must contain ``'spins'`` (isotopes of the working channels)
    and may contain ``'offset'`` (Hz per channel). ``K`` is
    :func:`spinpy.kinetics` in Liouville space and zero in Hilbert space,
    where exchange cannot be described.
    """
    parameters = _defaults(spin_system, parameters)
    spin_system = assume(spin_system, assumptions)
    H = hamiltonian(spin_system)
    H = frqoffset(spin_system, H, parameters)
    R, K = _dissipators(spin_system, H.shape[0])
    parameters["spc_dim"], parameters["spn_dim"] = 1, H.shape[0]
    return pulse_sequence(spin_system, parameters, H, R, K)


def crystal(
    spin_system: SpinSystem, pulse_sequence: PulseSequence, parameters: dict[str, Any], assumptions: str
) -> Any:
    """Run a pulse sequence for a single orientation, ``parameters['orientation']`` (ZYZ Euler angles, rad).

    The Hamiltonian is the isotropic part plus the anisotropic components
    rotated to that orientation (:func:`spinpy.orientation`), Hermitised.
    """
    parameters = _defaults(spin_system, parameters)
    if "orientation" not in parameters:
        raise ValueError("system orientation must be specified in parameters['orientation'] as three Euler angles.")
    angles = np.asarray(parameters["orientation"], dtype=float).reshape(-1)
    if angles.size != 3 or not np.all(np.isfinite(angles)):
        raise ValueError("parameters['orientation'] must be three finite Euler angles.")
    spin_system = assume(spin_system, assumptions)
    I, Q = hamiltonian(spin_system, aniso=True)
    I = frqoffset(spin_system, I, parameters)
    H = _hermitise(I + orientation(Q, angles))
    R, K = _dissipators(spin_system, H.shape[0])
    parameters["spc_dim"], parameters["spn_dim"] = 1, H.shape[0]
    return pulse_sequence(spin_system, parameters, H, R, K)


def powder(
    spin_system: SpinSystem, pulse_sequence: PulseSequence, parameters: dict[str, Any], assumptions: str
) -> Any:
    """Powder average of a pulse sequence over the spherical grid ``parameters['grid']``.

    The grid names are Spinach's (``'rep_2ang_6400pts_sph'``, ``'leb_2ang_rank_23'``,
    ...; see :func:`load_grid`). With ``parameters['sum_up']`` true (the
    default) the weighted sum over orientations is returned; otherwise the
    list of per-orientation results. Orientations run serially; Spinach
    uses a parallel pool and computes the same sum.
    """
    parameters = _defaults(spin_system, parameters)
    parameters.setdefault("sum_up", True)
    if not parameters.get("grid"):
        raise ValueError("spherical averaging grid must be specified in parameters['grid'].")
    if callable(parameters.get("rho0")):
        raise NotImplementedError("orientation-dependent initial states (callable rho0) are not ported yet.")
    grid = load_grid(parameters["grid"])
    spin_system = assume(spin_system, assumptions)
    I, Q = hamiltonian(spin_system, aniso=True)
    I = frqoffset(spin_system, I, parameters)
    R, K = _dissipators(spin_system, I.shape[0])
    parameters["spc_dim"], parameters["spn_dim"] = 1, I.shape[0]
    results = []
    for alpha, beta, gamma in zip(grid["alphas"], grid["betas"], grid["gammas"]):
        local = dict(parameters)
        local["current_angles"] = np.array([alpha, beta, gamma])
        H = _hermitise(I + orientation(Q, [alpha, beta, gamma]))
        results.append(pulse_sequence(spin_system, local, H, R, K))
    if not parameters["sum_up"]:
        return results
    answer = grid["weights"][0] * results[0]
    for weight, result in zip(grid["weights"][1:], results[1:]):
        answer = answer + weight * result
    return answer


def load_grid(name: str) -> dict[str, np.ndarray]:
    """Spherical averaging grid by its Spinach name: arrays ``alphas``, ``betas``, ``gammas`` (rad) and ``weights``.

    The grids are Spinach's ``kernel/grids`` files converted to NumPy
    archives (``porting/tools/gen_grids.py``): Lebedev (``leb_*``),
    repulsion (``rep_*``), icosahedral (``icos_*``) and ``single_crystal``.
    """
    path = resources.files("spinpy").joinpath("grids", f"{name}.npz")
    if not path.is_file():
        raise ValueError(f"unknown spherical grid {name!r}; see spinpy/grids for the available names.")
    with resources.as_file(path) as file, np.load(file) as data:
        return {key: np.array(data[key]) for key in ("alphas", "betas", "gammas", "weights")}


def _defaults(spin_system: SpinSystem, parameters: dict[str, Any]) -> dict[str, Any]:
    if spin_system.basis is None:
        raise ValueError("basis set information is missing, run basis() before calling this function.")
    if "spins" not in parameters or not parameters["spins"]:
        raise ValueError("working spins must be specified in parameters['spins'].")
    parameters = dict(parameters)
    parameters.setdefault("decouple", [])
    parameters.setdefault("rframes", [])
    parameters.setdefault("offset", np.zeros(len(parameters["spins"])))
    parameters.setdefault("needs", [])
    if parameters["rframes"]:
        raise NotImplementedError("additional rotating frame transformations are not ported yet.")
    if parameters["needs"]:
        raise NotImplementedError(f"parameters['needs'] {parameters['needs']} not ported yet.")
    return parameters


def _dissipators(spin_system: SpinSystem, dim: int) -> tuple[Matrix, Matrix]:
    R = sps.csr_array((dim, dim), dtype=complex)
    if spin_system.basis.formalism == "sphten-liouv":
        return R, kinetics(spin_system)
    return R, sps.csr_array((dim, dim), dtype=complex)


def _hermitise(H: Matrix) -> Matrix:
    H = (H + H.conj().T) / 2
    return sps.csr_array(H) if sps.issparse(H) else np.asarray(H)
