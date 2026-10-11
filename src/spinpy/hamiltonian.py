"""Isotropic Hamiltonian (ports of the isotropic part of ``hamiltonian.m`` and ``frqoffset.m``).

The anisotropic components (Spinach's second output ``Q``) are not ported yet.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sps

from spinpy.operators import operator
from spinpy.system import SpinSystem
from spinpy.utilities import Matrix, clean_up


def hamiltonian(spin_system: SpinSystem) -> Matrix:
    """Isotropic part of the Hamiltonian in rad/s, truncated as set by :func:`spinpy.assume`."""
    if spin_system.assumptions is None:
        raise ValueError("assumption information is missing, run assume() before calling this function.")
    if spin_system.basis is None or spin_system.basis.formalism != "zeeman-hilb":
        raise NotImplementedError("hamiltonian() is only ported for the zeeman-hilb formalism so far.")
    tols = spin_system.tols
    terms: list[tuple[complex, Sequence[str], Sequence[int]]] = []

    for n in range(spin_system.nspins):
        strength = spin_system.assumptions.zeeman[n]
        if strength == "full":
            zeeman_iso = np.trace(spin_system.zeeman_matrix[n]) / 3
        elif strength == "secular":
            # Rotating frame: subtract the carrier frequency
            zeeman_iso = np.trace(spin_system.zeeman_matrix[n]) / 3 - spin_system.basefrqs[n]
        elif strength == "ignore":
            continue
        else:
            raise ValueError(f"unknown strength specification for the Zeeman interaction of spin {n}")
        terms.append((zeeman_iso, ["Lz"], [n]))

    for n, k in _coupled_pairs(spin_system):
        coupling_iso = np.trace(spin_system.coupling_matrix[n][k]) / 3
        if abs(coupling_iso) <= 2 * np.pi * tols.inter_cutoff:
            continue
        strength = spin_system.assumptions.coupling[n][k]
        if strength in ("strong", "secular"):
            terms.append((coupling_iso, ["Lz", "Lz"], [n, k]))
            terms.append((coupling_iso / 2, ["L+", "L-"], [n, k]))
            terms.append((coupling_iso / 2, ["L-", "L+"], [n, k]))
        elif strength in ("z*", "*z", "zz"):
            terms.append((coupling_iso, ["Lz", "Lz"], [n, k]))
        elif strength not in ("ignore", "z+", "z-", "+z", "-z", "T(L,+1)", "T(L,-1)"):
            raise ValueError(f"unknown strength specification for the bilinear coupling between spins {n} and {k}")

    dim = int(np.prod(spin_system.mults))
    total = sps.csr_array((dim, dim), dtype=complex)
    for coeff, ops, spins in terms:
        if abs(coeff) > tols.liouv_zero:
            total = total + coeff * operator(spin_system, ops, spins)
    return clean_up(tols, total, tols.liouv_zero)


def frqoffset(spin_system: SpinSystem, H: Matrix, parameters: dict[str, Any]) -> Matrix:
    """Add rotating-frame offsets ``parameters['offset']`` (Hz) on ``parameters['spins']`` (isotopes)."""
    spins = list(parameters["spins"])
    offsets = np.atleast_1d(np.asarray(parameters["offset"], dtype=float))
    if not spins:
        raise ValueError("parameters['spins'] cannot be empty.")
    if any(s not in spin_system.isotopes for s in spins):
        raise ValueError("parameters['spins'] refers to a spin that is not present in the system.")
    if offsets.size != len(spins):
        raise ValueError("parameters['offset'] must have the same number of elements as parameters['spins'].")
    unique: dict[str, float] = {}
    for isotope, offset in zip(spins, offsets):
        if unique.setdefault(isotope, offset) != offset:
            raise ValueError("offsets on channels referring to the same spin must be the same.")
    for isotope, offset in unique.items():
        if offset != 0:
            H = H + 2 * np.pi * offset * operator(spin_system, "Lz", isotope)
    return _as_array_type(H)


def _coupled_pairs(spin_system: SpinSystem) -> list[tuple[int, int]]:
    """Spin pairs with a significant coupling tensor, excluding quadratic (n == k) terms."""
    cutoff = 2 * np.pi * spin_system.tols.inter_cutoff
    return [
        (n, k)
        for k in range(spin_system.nspins)
        for n in range(spin_system.nspins)
        if n != k
        and spin_system.coupling_matrix[n][k] is not None
        and np.linalg.norm(spin_system.coupling_matrix[n][k], 2) > cutoff
    ]


def _as_array_type(matrix: Any) -> Matrix:
    """Normalise mixed dense/sparse arithmetic results to ndarray or csr_array."""
    if sps.issparse(matrix):
        return sps.csr_array(matrix)
    return np.asarray(matrix)
