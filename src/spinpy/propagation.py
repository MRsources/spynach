"""Time propagation (ports of ``propagator.m``, ``step.m``, ``krylov.m`` and ``evolution.m``).

Liouville-space states are dense vectors ``(nstates,)`` or stacks
``(nstates, k)``; ``step`` applies ``expm(-1j * L * dt)`` to them without
forming the exponential, and ``evolution`` reduces the problem with
:mod:`spinpy.reduction` before propagating each subspace.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from spinpy.backend import get_backend
from spinpy.system import SpinSystem
from spinpy.utilities import Matrix, clean_up, dagger, hdot, matlab_round, nnz

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
    be = get_backend()
    if L.shape[0] < tols.small_matrix:
        return be.expm((-1j * timestep) * be.dense(L))   # GPU: small matrices stay on the host, as in Spinach

    # GPU: Spinach moves A to the device above 500 states and runs the Taylor loop and the
    # squarings there (sparse-sparse products); clean_up must then work on device arrays
    A = be.to_device(clean_up(tols, (-1j * timestep) * L, tols.prop_chop))
    mat_norm = be.norm1(A)
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

    P: Matrix = be.eye(A.shape[0])
    next_term: Matrix = P
    n = 1
    while be.nnz(next_term) > 0:
        next_term = ((1 / n) * A) @ next_term if be.issparse(A) else ((1 / n) * next_term) @ A
        next_term = clean_up(tols, next_term, tols.prop_chop)
        P = _add(P, next_term)
        n += 1
    log.debug("Taylor series converged in %d iterations.", n)

    P = clean_up(tols, P, tols.prop_chop)
    for _ in range(n_squarings):
        P = clean_up(tols, P @ P, tols.prop_chop)
    return be.to_host(P)   # GPU: propagators are returned to the host, as Spinach gathers them


def step(spin_system: SpinSystem, L: Matrix | Sequence[Matrix], rho: Matrix, timestep: float) -> Matrix:
    """One time step under a constant generator (port of ``step.m``).

    In the Liouville formalisms ``rho`` is a vector or a stack of column
    vectors and the result is ``expm(-1j * L * dt) @ rho`` computed as a
    Taylor series in substeps of ``2 / ||L||_1``. ``L`` may also be a list
    of two or three generators sampled across the step (Lie-group
    quadratures of Rasulov et al. 2023). In ``zeeman-hilb`` ``rho`` is a
    density matrix, propagated by a dense exponential below
    ``tols.small_matrix`` and by the commutator series above it.
    """
    if not np.isscalar(timestep) or not np.isfinite(timestep):
        raise ValueError("timestep must be a finite scalar.")
    generators = list(L) if isinstance(L, (list, tuple)) else [L]
    if len(generators) not in (1, 2, 3):
        raise ValueError("L must be a matrix or a list of two or three matrices.")
    if timestep == 0:
        return rho
    if spin_system.basis is None:
        raise ValueError("basis set information is missing, run basis() before calling this function.")
    be = get_backend()
    norm = max(be.norm1(G) for G in generators) * abs(timestep)
    nsteps = max(1, math.ceil(norm / 2))
    if nsteps > 1e4:
        raise ValueError("either dt is too long, or L is too big: |L*dt|>1e4, check both.")
    if nsteps > 100:
        log.warning("%d substeps required, consider using evolution() here.", nsteps)

    if spin_system.basis.formalism == "zeeman-hilb":
        if len(generators) > 1:
            raise NotImplementedError("Lie-group quadratures are not ported for Hilbert space yet.")
        H = generators[0]
        if H.shape[0] < spin_system.tols.small_matrix:
            P = be.expm((-1j * timestep) * be.dense(H))
            return P @ rho @ dagger(P)
        return _commutator_series(H, rho, timestep, nsteps)

    # GPU: step.m moves L and rho to the device whenever 'gpu' is enabled; the Taylor
    # loop below is pure matvec and runs unchanged on device arrays
    rho = be.dense(rho, dtype=complex)
    scaling = float(np.abs(rho).max()) if rho.size else 0.0
    if scaling == 0:
        log.warning("all-zero state received")
        return rho
    generators = [be.to_device(G) for G in generators]
    return be.to_host(scaling * _reordered_taylor(generators, be.to_device(rho / scaling), timestep, nsteps))


def _reordered_taylor(generators: list[Matrix], rho: np.ndarray, t: float, nsteps: int) -> np.ndarray:
    """Taylor series of expm(-1j*L*t) acting on rho, in nsteps substeps (step.m ``reordered_taylor``)."""
    eps = np.finfo(float).eps
    dt = t / nsteps
    for _ in range(nsteps):
        next_term, k = rho, 1
        while np.any(np.abs(next_term) > eps):
            if len(generators) == 1:
                next_term = (-1j / k) * dt * (generators[0] @ next_term)
            elif len(generators) == 2:
                a, b = generators[0] @ next_term, generators[1] @ next_term
                next_term = (-1j / 2) * (1 / k) * dt * (a + b) + (1 / 6) * (1 / k) * (t**2 / nsteps) * (
                    generators[0] @ b - generators[1] @ a
                )
            else:
                a, b, c = generators[0] @ next_term, generators[1] @ next_term, generators[2] @ next_term
                next_term = (-1j / 6) * (1 / k) * dt * (a + 4 * b + c) + (1 / 12) * (1 / k) * (t**2 / nsteps) * (
                    generators[0] @ c - generators[2] @ a
                )
            rho = rho + next_term
            k += 1
        if k > 32:
            log.warning("loss of accuracy in the Taylor series, k=%d, use evolution() here instead of step()", k)
    return rho


def _commutator_series(H: Matrix, rho: Matrix, t: float, nsteps: int) -> Matrix:
    """Hilbert-space step by the nested commutator series (step.m ``comm_series``)."""
    eps = np.finfo(float).eps
    scaling = get_backend().norm1(rho)
    if scaling == 0:
        log.warning("all-zero state received")
        return rho
    rho = rho / scaling
    dt = t / nsteps
    for _ in range(nsteps):
        next_term, k = rho, 1
        while nnz(next_term) > 0:
            next_term = (-1j * dt / k) * (H @ next_term - next_term @ H)
            next_term = _chop(next_term, eps)
            rho = _add(rho, next_term)
            k += 1
        if k > 32:
            log.warning("loss of accuracy in the commutator series, iter=%d, use evolution() here instead of step()", k)
        rho = _chop(rho, eps)
    return scaling * rho


def krylov(
    spin_system: SpinSystem, L: Matrix, coil: np.ndarray | None, rho: np.ndarray, timestep: float, nsteps: int, output: str
) -> np.ndarray:
    """Propagation by repeated :func:`step` calls, for generators too large to exponentiate (port of ``krylov.m``)."""
    if spin_system.basis is None or spin_system.basis.formalism != "sphten-liouv":
        raise NotImplementedError("krylov() is only ported for the sphten-liouv formalism.")
    rho = get_backend().dense(rho, dtype=complex)   # GPU: krylov.m keeps L, rho and coil on the device across steps
    if output == "final":
        return step(spin_system, L, rho, timestep * nsteps)
    if output == "trajectory":
        answer = np.zeros((rho.shape[0], nsteps + 1), dtype=complex)
        answer[:, 0] = rho.reshape(-1)
        for n in range(nsteps):
            rho = step(spin_system, L, rho, timestep)
            answer[:, n + 1] = rho.reshape(-1)
        return answer
    if output == "observable":
        answer = np.zeros((nsteps + 1,) + rho.shape[1:], dtype=complex)
        answer[0] = _detect(coil, rho)
        for n in range(nsteps):
            rho = step(spin_system, L, rho, timestep)
            answer[n + 1] = _detect(coil, rho)
        return answer
    if output == "multichannel":
        answer = np.zeros((coil.shape[1], nsteps + 1, rho.shape[1]), dtype=complex)
        answer[:, 0, :] = coil.conj().T @ rho
        for n in range(nsteps):
            rho = step(spin_system, L, rho, timestep)
            answer[:, n + 1, :] = coil.conj().T @ rho
        return answer
    if output == "refocus":
        rho = rho.copy()
        for n in range(1, rho.shape[1]):
            rho[:, n:] = step(spin_system, L, rho[:, n:], timestep)
        return rho
    raise NotImplementedError(f"krylov output {output!r} not ported yet.")


def evolution(
    spin_system: SpinSystem,
    L: Matrix,
    coil: Matrix | None,
    rho: Matrix,
    timestep: float,
    nsteps: int,
    output: str,
    destination: np.ndarray | None = None,
) -> np.ndarray:
    """Time evolution under a constant generator (port of ``evolution.m``).

    ``zeeman-hilb``: ``output='observable'`` only; returns ``hdot(coil, rho(t))``
    at ``nsteps + 1`` equally spaced time points starting at t = 0, as a 1-D
    complex array. Spinach parallelises this case with the Kuprov-Edwards
    split over the columns of rho; the serial loop computes the same thing.

    ``sphten-liouv``: ``rho`` is a vector or a stack of column vectors. The
    problem is first reduced with :func:`spinpy.reduction.reduce`, screened
    against ``destination`` or, for ``'observable'``, the coil state
    (destination state screening, unless ``'dss'`` is disabled), then each
    subspace is propagated with the exponential propagator, or with
    :func:`krylov` above ``tols.krylov_tol`` states. Outputs: ``'observable'``
    (``coil^H rho(t)``, shape ``(nsteps + 1,)`` or ``(nsteps + 1, k)``),
    ``'final'`` (the state after ``nsteps`` steps), ``'trajectory'``
    (``(nstates, nsteps + 1)``, single vector only), ``'multichannel'``
    (several coils stacked as the columns of ``coil``: ``(ncoils, nsteps + 1,
    k)``, as in Spinach), ``'refocus'`` (column ``k`` of the stack propagated
    for ``k`` steps, the second half of a constant-time or refocused
    indirect dimension; ``nsteps`` is ignored).
    """
    if spin_system.basis is None:
        raise ValueError("basis set information is missing, run basis() before calling this function.")
    if isinstance(rho, (list, tuple)):
        raise NotImplementedError("state stacks as lists are not ported yet; stack column vectors in an array.")
    if spin_system.basis.formalism == "zeeman-hilb":
        if output != "observable":
            raise NotImplementedError(f"evolution output {output!r} not ported yet in zeeman-hilb; only 'observable'.")
        if coil.shape != rho.shape:
            raise ValueError("coil and rho must have identical dimensions.")
        P = propagator(spin_system, L, timestep)
        P_dag = dagger(P)
        answer = np.empty(nsteps + 1, dtype=complex)
        for n in range(nsteps + 1):
            answer[n] = hdot(coil, rho)
            rho = P @ rho @ P_dag
        return answer
    if spin_system.basis.formalism != "sphten-liouv":
        raise NotImplementedError(f"formalism {spin_system.basis.formalism!r} not ported yet.")
    if output not in ("observable", "final", "trajectory", "multichannel", "refocus"):
        raise NotImplementedError(f"evolution output {output!r} not ported yet.")

    from spinpy.reduction import reduce  # noqa: PLC0415  (reduction imports step from this module)

    tols = spin_system.tols
    be = get_backend()
    rho = be.dense(rho, dtype=complex)
    single = rho.ndim == 1
    rho = rho.reshape(rho.shape[0], -1)
    if output == "trajectory" and rho.shape[1] > 1:
        raise ValueError("trajectory output is not available for state vector stacks.")
    if coil is not None:
        coil = be.dense(coil, dtype=complex).reshape(-1, 1) if be.dense(coil).ndim == 1 else be.dense(coil, dtype=complex)
    if "dss" in spin_system.disable:
        projectors = reduce(spin_system, L, rho)
    elif destination is not None:
        projectors = reduce(spin_system, dagger(L), destination)
    elif output in ("observable", "multichannel"):
        if coil is None:
            raise ValueError("coil state must be given for the observable and multichannel outputs.")
        projectors = reduce(spin_system, dagger(L), coil)
    else:
        projectors = reduce(spin_system, L, rho)

    use_krylov = "krylov" in spin_system.enable
    if output == "observable":
        answer = np.zeros((nsteps + 1, rho.shape[1]), dtype=complex)
    elif output == "multichannel":
        answer = np.zeros((coil.shape[1], nsteps + 1, rho.shape[1]), dtype=complex)
    elif output in ("final", "refocus"):
        answer = np.zeros_like(rho)
    else:
        answer = np.zeros((rho.shape[0], nsteps + 1), dtype=complex)
    for P in projectors:
        P_dag = P.conj().T
        L_loc = clean_up(tols, _as_matrix(P_dag @ L @ P), tols.liouv_zero)
        rho_loc = _chop(P_dag @ rho, tols.zte_tol)
        dim = L_loc.shape[0]
        krylov_path = use_krylov or (dim >= tols.krylov_tol and "krylov" not in spin_system.disable)
        # GPU: evolution.m moves P, rho_loc and coil_loc to the device for the loops below
        # and gathers the observables; the propagator itself is built as above
        if output == "observable":
            coil_loc = _chop(P_dag @ coil, tols.zte_tol)
            if krylov_path:
                answer += krylov(spin_system, L_loc, coil_loc, rho_loc, timestep, nsteps, "observable")
            else:
                prop = propagator(spin_system, L_loc, timestep)
                for n in range(nsteps + 1):
                    answer[n] += _detect(coil_loc, rho_loc)
                    rho_loc = prop @ rho_loc
        elif output == "multichannel":
            coil_loc = _chop(P_dag @ coil, tols.zte_tol)
            if krylov_path:
                answer += krylov(spin_system, L_loc, coil_loc, rho_loc, timestep, nsteps, "multichannel")
            else:
                prop = propagator(spin_system, L_loc, timestep)
                for n in range(nsteps + 1):
                    answer[:, n, :] += coil_loc.conj().T @ rho_loc
                    rho_loc = prop @ rho_loc
        elif output == "refocus":
            if krylov_path:
                rho_loc = krylov(spin_system, L_loc, None, rho_loc, timestep, nsteps, "refocus")
            else:
                prop = propagator(spin_system, L_loc, timestep)
                rho_loc = rho_loc.copy()
                for n in range(1, rho_loc.shape[1]):
                    rho_loc[:, n:] = prop @ rho_loc[:, n:]
            answer += P @ _chop(rho_loc, tols.zte_tol)
        elif output == "final":
            if krylov_path:
                rho_loc = krylov(spin_system, L_loc, None, rho_loc, timestep, nsteps, "final")
            else:
                # Optimal stepping: fewer, longer steps when the propagator is cheap relative to the stack
                nsteps_opt = max(1, math.ceil(math.log2(dim / (rho_loc.shape[1] * math.log(2)))))
                prop = propagator(spin_system, L_loc, timestep * nsteps / nsteps_opt)
                for _ in range(nsteps_opt):
                    rho_loc = prop @ rho_loc
            answer += P @ _chop(rho_loc, tols.zte_tol)
        else:
            if krylov_path:
                traj = krylov(spin_system, L_loc, None, rho_loc, timestep, nsteps, "trajectory")
            else:
                prop = propagator(spin_system, L_loc, timestep)
                traj = np.zeros((dim, nsteps + 1), dtype=complex)
                for n in range(nsteps + 1):
                    traj[:, n] = rho_loc[:, 0]
                    rho_loc = prop @ rho_loc
            answer += P @ _chop(traj, tols.zte_tol)
    if output in ("final", "refocus"):
        answer = _chop(answer, tols.zte_tol)
    if single and output in ("observable", "final", "refocus"):
        answer = answer[:, 0]
    elif single and output == "multichannel":
        answer = answer[:, :, 0]
    return answer


def _detect(coil: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """``coil' * rho`` for a coil column and a state stack ``(nstates, k)``, giving ``(k,)``."""
    return (coil.conj().T @ rho).reshape(-1)


def _chop(x: Matrix, tol: float) -> Matrix:
    """Round the elements of a vector, stack, or matrix to a grid of ``tol`` (clean_up without storage choice)."""
    be = get_backend()
    if be.issparse(x):
        x = be.csr(x, copy=True)
        x.data = tol * matlab_round(x.data / tol)
        x.eliminate_zeros()
        return x
    return tol * matlab_round(be.dense(x) / tol)


def _as_matrix(x: Matrix) -> Matrix:
    be = get_backend()
    return be.csr(x) if be.issparse(x) else be.dense(x)


def _norm1(matrix: Matrix) -> float:
    """Maximum absolute column sum (Spinach cheap_norm for non-polyadic input)."""
    return get_backend().norm1(matrix)


def _add(a: Matrix, b: Matrix) -> Matrix:
    """Sum of two matrices that may be dense or sparse; dense if either is dense."""
    be = get_backend()
    if be.issparse(a) and be.issparse(b):
        return be.csr(a + b)
    return be.dense(a) + be.dense(b)
