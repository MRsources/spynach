"""Spin states (port of the ``'zeeman-hilb'`` branch of ``state.m``)."""

from __future__ import annotations

from collections.abc import Sequence

import scipy.sparse as sps

from spinpy.operators import operator
from spinpy.system import SpinSystem


def state(
    spin_system: SpinSystem, states: str | Sequence[str], spins: str | int | Sequence[int]
) -> sps.csr_array:
    """Density matrix from a human-readable description, same forms as :func:`spinpy.operator`.

    In Hilbert space the state is the (unnormalised) operator itself, e.g.
    ``state(ss, 'L+', '1H')`` is the sum of L+ over all protons.
    """
    if spin_system.basis is None:
        raise ValueError("basis set information is missing, run basis() before calling this function.")
    if spin_system.basis.formalism != "zeeman-hilb":
        raise NotImplementedError(f"formalism {spin_system.basis.formalism!r} not ported yet.")
    return operator(spin_system, states, spins)
