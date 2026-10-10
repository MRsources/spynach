"""Basis set selection (port of ``basis.m``).

Ported: the ``'zeeman-hilb'`` formalism (Hilbert space, Zeeman product basis,
no truncation). Permutation symmetry, ``'zeeman-liouv'``, and
``'sphten-liouv'`` are not ported yet.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from spinpy.system import Basis, SpinSystem

BAS_FIELDS = {"formalism", "approximation"}


def basis(spin_system: SpinSystem, bas: dict[str, Any]) -> SpinSystem:
    """Return a copy of ``spin_system`` with the basis described by ``bas``.

    ``bas['formalism']`` must be ``'zeeman-hilb'`` and ``bas['approximation']``
    must be ``'none'``.
    """
    unported = set(bas) - BAS_FIELDS
    if unported:
        raise NotImplementedError(f"bas field(s) {sorted(unported)} not ported yet")
    if "formalism" not in bas:
        raise ValueError("basis formalism must be specified in bas['formalism'].")
    if bas["formalism"] != "zeeman-hilb":
        raise NotImplementedError(f"formalism {bas['formalism']!r} not ported yet; only 'zeeman-hilb'")
    if "approximation" not in bas:
        raise ValueError("approximation level must be specified in bas['approximation'] for zeeman-hilb formalism.")
    if bas["approximation"] != "none":
        raise ValueError("bas['approximation'] should be set to 'none' in zeeman-hilb formalism.")
    return replace(
        spin_system,
        basis=Basis(formalism="zeeman-hilb", approximation="none", levels=_zeeman_levels(spin_system.mults)),
    )


def _zeeman_levels(mults: np.ndarray) -> np.ndarray:
    """Level number of every spin in every Zeeman product state; spin 0 varies slowest."""
    grids = np.meshgrid(*(np.arange(1, m + 1) for m in mults), indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)
