"""Spin states (port of ``state.m``).

In ``zeeman-hilb`` a state is the operator itself (a density matrix). In
``sphten-liouv`` it is a dense 1-D complex vector of coefficients in the
orthonormal basis of :mod:`spinpy.basis`; its first element is the
coefficient of the unit state ``E / sqrt(dim)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sps

from spinpy.operators import human2opspec, operator, superop
from spinpy.system import SpinSystem

METHODS = ("exact", "cheap", "chem")


def state(
    spin_system: SpinSystem,
    states: str | Sequence[str],
    spins: str | int | Sequence[int],
    method: str = "exact",
) -> sps.csr_array | np.ndarray:
    """State from a human-readable description, same forms as :func:`spinpy.operator`.

    ``zeeman-hilb``: the (unnormalised) operator, e.g. ``state(ss, 'L+', '1H')``
    is the sum of L+ over all protons; ``method`` is ignored.

    ``sphten-liouv``: a vector, by one of three methods as in Spinach:

    * ``'exact'`` (default): the operator applied to the unit state, i.e. the
      expansion of ``Op / sqrt(dim)`` in the orthonormal basis (a spin-1/2
      ``Lz`` has coefficient 1/2 on its ``T(1,0)`` state);
    * ``'cheap'``: the coefficient of each operator placed directly on its
      basis state (``Lz`` gives 1, ``L+`` gives -sqrt(2)); errors if the
      state is not in the basis;
    * ``'chem'``: ``'exact'`` with each term multiplied by the concentration
      of the chemical subsystem its spins belong to.
    """
    if spin_system.basis is None:
        raise ValueError("basis set information is missing, run basis() before calling this function.")
    if method not in METHODS:
        raise ValueError(f"unknown state generation method; use one of {METHODS}.")
    formalism = spin_system.basis.formalism
    if formalism == "zeeman-hilb":
        return operator(spin_system, states, spins)
    if formalism != "sphten-liouv":
        raise NotImplementedError(f"formalism {formalism!r} not ported yet.")

    basis = spin_system.basis.basis
    nstates = basis.shape[0]
    if method == "exact":
        return operator(spin_system, states, spins, "left")[:, [0]].toarray().ravel()

    opspecs, coeffs = human2opspec(spin_system, states, spins)
    rho = np.zeros(nstates, dtype=complex)
    if method == "cheap":
        for spec, coeff in zip(opspecs, coeffs):
            matches = np.flatnonzero((basis == spec).all(axis=1))
            if matches.size == 0:
                raise ValueError("the requested state is not present in the basis.")
            rho[matches[0]] += coeff
        return rho
    for spec, coeff in zip(opspecs, coeffs):
        active = set(np.flatnonzero(spec))
        species = [s for s, part in enumerate(spin_system.chem.parts) if active <= set(part)]
        if len(species) != 1:
            raise ValueError("the spin state requested crosses chemical species boundaries.")
        rows, cols, vals = superop(spin_system, spec, "left")
        on_unit = cols == 0
        np.add.at(rho, rows[on_unit], coeff * spin_system.chem.concs[species[0]] * vals[on_unit])
    return rho
