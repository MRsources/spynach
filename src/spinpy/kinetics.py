"""Chemical kinetics superoperator (port of the exchange branch of ``kinetics.m``).

Ported: first-order exchange between chemical subsystems (``inter['chem']
['rates']``). Magnetisation flux and radical-pair kinetics are refused by
:func:`spinpy.create`. The superoperator is dissipative and enters the
Liouvillian as ``L = H + 1j * R + 1j * K``, as in Spinach.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from spinpy.system import SpinSystem


def kinetics(spin_system: SpinSystem) -> sps.csr_array:
    """Exchange superoperator ``K`` in the ``sphten-liouv`` basis, in Hz.

    For every non-zero rate ``rates[i, j]`` (flow from part ``j`` into part
    ``i``) the states living on part ``j`` are mapped onto the identical
    states of part ``i``: ``K[states_i, states_j] += rates[i, j]``. The
    diagonal rates give the drain on each part's own states. The unit
    state is never touched. With no rates the result is the zero matrix.
    """
    if spin_system.basis is None:
        raise ValueError("basis set information is missing, run basis() before calling this function.")
    if spin_system.basis.formalism != "sphten-liouv":
        raise NotImplementedError("chemical reaction treatment is only available for sphten-liouv formalism.")
    basis = spin_system.basis.basis
    nstates = basis.shape[0]
    rates = spin_system.chem.rates
    if rates is None:
        return sps.csr_array((nstates, nstates), dtype=complex)
    rows, cols, vals = [], [], []
    for i, j in zip(*np.nonzero(rates)):
        part_i, part_j = list(spin_system.chem.parts[i]), list(spin_system.chem.parts[j])
        states_i = np.flatnonzero(basis[:, part_i].any(axis=1))
        states_j = np.flatnonzero(basis[:, part_j].any(axis=1))
        if not np.array_equal(basis[np.ix_(states_i, part_i)], basis[np.ix_(states_j, part_j)]):
            raise ValueError("spin systems on either side of the reaction arrow have different topologies or basis sets.")
        rows.append(states_i)
        cols.append(states_j)
        vals.append(np.full(len(states_i), rates[i, j], dtype=complex))
    return sps.csr_array(
        sps.coo_array((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nstates, nstates))
    )
