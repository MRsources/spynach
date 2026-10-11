"""Analytical state filters (ports of ``correlation.m`` and ``coherence.m``).

Ported for ``sphten-liouv``, where correlation and coherence orders are
labels of the basis states and filtering means zeroing coefficients.
The Hilbert-space branches (which need projection superoperators) are not
ported yet.

State arrays may hold several vectors or a Fokker-Planck stack: the last
axis must be the spin state space (``(..., nstates)``), which is spinpy's
layout for composite spaces with the spin index varying fastest.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from spinpy.operators import lin2lm
from spinpy.system import SpinSystem


def correlation(
    spin_system: SpinSystem, rho: np.ndarray, orders: int | Sequence[int], spins: str | Sequence[int] = "all"
) -> np.ndarray:
    """Keep only the states whose spin correlation order over ``spins`` is in ``orders``.

    ``spins`` is ``'all'``, an isotope name, or a list of 0-based spin
    numbers; ``orders`` are non-negative integers.
    """
    _check(spin_system, rho)
    orders = np.atleast_1d(np.asarray(orders))
    if orders.ndim != 1 or np.any(orders < 0) or np.any(orders != orders.astype(int)):
        raise ValueError("orders must be a vector of non-negative integers.")
    selected = _spins(spin_system, spins)
    present = np.count_nonzero(spin_system.basis.basis[:, selected], axis=1)
    return _apply(rho, np.isin(present, orders))


def coherence(spin_system: SpinSystem, rho: np.ndarray, spec: Sequence[tuple[Any, Any]]) -> np.ndarray:
    """Keep only the states whose coherence orders match every entry of ``spec``.

    ``spec`` is a list of ``(spins, orders)`` pairs, e.g.
    ``[("13C", [1, -1]), ("1H", -1)]``: the total coherence order over the
    named spins (``'all'``, ``'electrons'``, ``'nuclei'``, an isotope, or
    0-based spin numbers) must be one of ``orders``.
    """
    _check(spin_system, rho)
    _, projections = lin2lm(spin_system.basis.basis)
    mask = np.ones(spin_system.basis.nstates, dtype=bool)
    for entry in spec:
        if len(entry) != 2:
            raise ValueError("each element of spec must be a (spins, orders) pair.")
        selected = _spins(spin_system, entry[0])
        orders = np.atleast_1d(np.asarray(entry[1]))
        if np.any(orders != orders.astype(int)):
            raise ValueError("coherence orders in spec must be integers.")
        mask &= np.isin(projections[:, selected].sum(axis=1), orders)
    return _apply(rho, mask)


def _check(spin_system: SpinSystem, rho: np.ndarray) -> None:
    if spin_system.basis is None:
        raise ValueError("basis set information is missing, run basis() before calling this function.")
    if spin_system.basis.formalism != "sphten-liouv":
        raise NotImplementedError("analytical order selection is only ported for the sphten-liouv formalism.")
    if np.shape(rho)[-1] != spin_system.basis.nstates:
        raise ValueError("the last dimension of rho must be the dimension of the spin state space.")


def _apply(rho: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.array(rho, copy=True)
    out[..., ~mask] = 0
    return out


def _spins(spin_system: SpinSystem, spins: str | Sequence[int]) -> list[int]:
    isotopes = spin_system.isotopes
    if isinstance(spins, str):
        if spins == "all":
            return list(range(spin_system.nspins))
        if spins == "electrons":
            return [n for n, iso in enumerate(isotopes) if iso.startswith("E")]
        if spins == "nuclei":
            return [n for n, iso in enumerate(isotopes) if not iso.startswith("E")]
        if spins not in isotopes:
            raise ValueError("spins must be 'all', 'electrons', 'nuclei', an isotope present in the system, or spin numbers.")
        return [n for n, iso in enumerate(isotopes) if iso == spins]
    numbers = [int(n) for n in np.atleast_1d(spins)]
    if any(n < 0 or n >= spin_system.nspins for n in numbers):
        raise ValueError("spin numbers must be within the system bounds.")
    return numbers
