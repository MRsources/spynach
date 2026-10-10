"""Time propagation (ports of ``propagator.m`` and the ``'zeeman-hilb'`` part of ``evolution.m``)."""

from __future__ import annotations

import logging
import math
from dataclasses import replace

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sps

from spinpy.system import SpinSystem
from spinpy.utilities import Matrix, clean_up, dagger, hdot, nnz

log = logging.getLogger("spinpy")


def propagator(spin_system: SpinSystem, L: Matrix, timestep: float) -> Matrix:
    """Exponential propagator ``expm(-1j * L * timestep)``.

    Below ``tols.small_matrix`` this is a dense Pade exponential. Larger
    generators use Spinach's scaled and squared Taylor series, rounding each
    term to ``tols.prop_chop`` so that sparsity is kept where it exists.
    """
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L argument must be a square matrix.")
    if not np.isfinite(timestep):
        raise ValueError("timestep must be a finite scalar.")
    tols = spin_system.tols
    if L.shape[0] < tols.small_matrix:
        generator = L.toarray() if sps.issparse(L) else np.asarray(L)
        return sla.expm((-1j * timestep) * generator)

    A = clean_up(tols, (-1j * timestep) * L, tols.prop_chop)
    mat_norm = _norm1(A)
    if mat_norm > 1e9:
        raise ValueError("norm of -i*L*dt exceeds 1e9, check your L and your dt.")
    if mat_norm > 1024:
        log.warning("time step greatly exceeds the timescale of system dynamics; "
                    "exponentiation tolerance tightened up to eps('double').")
        tols = replace(tols, prop_chop=np.finfo(float).eps)
    elif mat_norm > 16:
        log.warning("time step exceeds the timescale of system dynamics.")

    n_squarings = max(0, math.ceil(math.log2(mat_norm))) if mat_norm > 0 else 0
    if n_squarings > 0:
        A = (1 / 2**n_squarings) * A

    P: Matrix = sps.eye_array(A.shape[0], dtype=complex, format="csr")
    next_term: Matrix = P
    n = 1
    while nnz(next_term) > 0:
        next_term = ((1 / n) * A) @ next_term if sps.issparse(A) else ((1 / n) * next_term) @ A
        next_term = clean_up(tols, next_term, tols.prop_chop)
        P = _add(P, next_term)
        n += 1
    log.debug("Taylor series converged in %d iterations.", n)

    P = clean_up(tols, P, tols.prop_chop)
    for _ in range(n_squarings):
        P = clean_up(tols, P @ P, tols.prop_chop)
    return P


def evolution(
    spin_system: SpinSystem,
    L: Matrix,
    coil: Matrix,
    rho: Matrix,
    timestep: float,
    nsteps: int,
    output: str,
) -> np.ndarray:
    """Time evolution under a constant generator.

    Ported: ``output='observable'`` in the ``'zeeman-hilb'`` formalism, which
    returns ``hdot(coil, rho(t))`` at ``nsteps + 1`` equally spaced time points,
    starting at t = 0, as a 1-D complex array.

    Spinach parallelises this case with the Kuprov-Edwards split over the
    columns of rho; the serial loop used here computes the same quantity.
    """
    if spin_system.basis is None or spin_system.basis.formalism != "zeeman-hilb":
        raise NotImplementedError("evolution() is only ported for the 'zeeman-hilb' formalism.")
    if output != "observable":
        raise NotImplementedError(f"evolution output {output!r} not ported yet; only 'observable'.")
    if isinstance(rho, (list, tuple)):
        raise NotImplementedError("state stacks are not ported yet.")
    if coil.shape != rho.shape:
        raise ValueError("coil and rho must have identical dimensions.")
    P = propagator(spin_system, L, timestep)
    P_dag = dagger(P)
    answer = np.empty(nsteps + 1, dtype=complex)
    for n in range(nsteps + 1):
        answer[n] = hdot(coil, rho)
        rho = P @ rho @ P_dag
    return answer


def _norm1(matrix: Matrix) -> float:
    """Maximum absolute column sum (Spinach cheap_norm for non-polyadic input)."""
    column_sums = abs(matrix).sum(axis=0)
    return float(np.max(column_sums)) if matrix.shape[1] else 0.0


def _add(a: Matrix, b: Matrix) -> Matrix:
    """Sum of two matrices that may be dense or sparse; dense if either is dense."""
    if sps.issparse(a) and sps.issparse(b):
        return sps.csr_array(a + b)
    return np.asarray((a.toarray() if sps.issparse(a) else a) + (b.toarray() if sps.issparse(b) else b))
